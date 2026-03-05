'use strict';

// =============================================================================
// Fluid Noise Warp — WebGL 2 fluid simulation + GPU noise advection
//
// Architecture:
//   1. Navier-Stokes fluid sim on GPU (velocity + dye FBOs)
//   2. Gaussian noise advected through velocity field on GPU (fragment shader)
//   3. Noise normalization pass restores mean=0, std=1 each frame
//   4. Display shader handles all 6 modes
//   Zero CPU-GPU transfers per frame. Everything stays on GPU.
//
// Based on PavelDoGreat's WebGL Fluid Simulation (MIT) + ICLR 2025 noise warp
// =============================================================================

// ---------------------------------------------------------------------------
// Config & Constants
// ---------------------------------------------------------------------------

const NOISE_RESOLUTIONS = [2048, 1024, 512, 256];
const BN_ITERS_OPTIONS = [2, 5, 10];
const MODE_NAMES = ['noise', 'scene', 'scene+noise', 'dither', 'motion', 'raw'];
const ROUND_MODES = ['None', 'All', '>1'];
const STORAGE_KEY = 'fluid_warp_settings';
const SETTINGS_VERSION = 1;

const DEFAULTS = {
    noiseResIdx: 1,    // 1024
    blueNoise: false,
    bnItersIdx: 0,
    greyscale: true,
    uniformDisplay: true,
    retina: false,
    bilinear: false,
    threshOn: false,
    threshSlider: 500,
    roundMode: 0,
    noiseOpacity: 25,
    simRes: 128,
    dyeRes: 1024,
    curl: 30,
    dissipation: 100,  // /100 = 1.0
    pressure: 80,      // /100 = 0.8
    noiseDissipation: 0, // 0 = no dissipation for noise
};

function loadSettings() {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    const saved = JSON.parse(raw);
    if (saved._version !== SETTINGS_VERSION) return { ...DEFAULTS };
    return { ...DEFAULTS, ...saved };
}

function saveSettings(s) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...s, _version: SETTINGS_VERSION }));
}

// ---------------------------------------------------------------------------
// WebGL 2 Setup
// ---------------------------------------------------------------------------

const canvas = document.getElementById('canvas');
const params = { alpha: true, depth: false, stencil: false, antialias: false, preserveDrawingBuffer: false };
const gl = canvas.getContext('webgl2', params);
if (!gl) { document.body.innerHTML = '<h1>WebGL 2 not supported</h1>'; throw new Error('No WebGL2'); }

gl.getExtension('EXT_color_buffer_float');
gl.getExtension('OES_texture_float_linear');
gl.getExtension('EXT_color_buffer_half_float');
const supportsLinear = !!gl.getExtension('OES_texture_float_linear');

const halfFloatTexType = gl.HALF_FLOAT;
const formatRGBA = { internalFormat: gl.RGBA16F, format: gl.RGBA };
const formatRG   = { internalFormat: gl.RG16F,   format: gl.RG };
const formatR    = { internalFormat: gl.R16F,     format: gl.RED };

// ---------------------------------------------------------------------------
// Shader compilation
// ---------------------------------------------------------------------------

function compileShader(type, source, keywords) {
    if (keywords) {
        let defs = '';
        for (const k of keywords) defs += '#define ' + k + '\n';
        source = defs + source;
    }
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        console.error('Shader:', gl.getShaderInfoLog(shader));
    return shader;
}

function createProgramGL(vs, fs) {
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
        console.error('Link:', gl.getProgramInfoLog(prog));
    return prog;
}

function getUniforms(program) {
    const u = {};
    const n = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
        const name = gl.getActiveUniform(program, i).name;
        u[name] = gl.getUniformLocation(program, name);
    }
    return u;
}

class Program {
    constructor(vs, fs) { this.program = createProgramGL(vs, fs); this.uniforms = getUniforms(this.program); }
    bind() { gl.useProgram(this.program); }
}

function hashCode(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
    return h;
}

class Material {
    constructor(vs, fragSrc) {
        this.vertexShader = vs;
        this.fragSrc = fragSrc;
        this.programs = {};
        this.activeProgram = null;
        this.uniforms = {};
    }
    setKeywords(keywords) {
        let h = 0; for (const k of keywords) h += hashCode(k);
        let prog = this.programs[h];
        if (!prog) {
            const fs = compileShader(gl.FRAGMENT_SHADER, this.fragSrc, keywords);
            prog = createProgramGL(this.vertexShader, fs);
            this.programs[h] = prog;
        }
        if (prog === this.activeProgram) return;
        this.uniforms = getUniforms(prog);
        this.activeProgram = prog;
    }
    bind() { gl.useProgram(this.activeProgram); }
}

// ---------------------------------------------------------------------------
// Vertex Shaders
// ---------------------------------------------------------------------------

const baseVertexShader = compileShader(gl.VERTEX_SHADER, `#version 300 es
precision highp float;
in vec2 aPosition;
out vec2 vUv, vL, vR, vT, vB;
uniform vec2 texelSize;
void main() {
    vUv = aPosition * 0.5 + 0.5;
    vL = vUv - vec2(texelSize.x, 0.0);
    vR = vUv + vec2(texelSize.x, 0.0);
    vT = vUv + vec2(0.0, texelSize.y);
    vB = vUv - vec2(0.0, texelSize.y);
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`);

// ---------------------------------------------------------------------------
// Fragment Shaders: Fluid Simulation
// ---------------------------------------------------------------------------

const copyShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uTexture;
out vec4 fc;
void main() { fc = texture(uTexture, vUv); }
`);

const clearShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uTexture;
uniform float value;
out vec4 fc;
void main() { fc = value * texture(uTexture, vUv); }
`);

const splatShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTarget;
uniform float aspectRatio;
uniform vec3 color;
uniform vec2 point;
uniform float radius;
out vec4 fc;
void main() {
    vec2 p = vUv - point.xy;
    p.x *= aspectRatio;
    fc = vec4(texture(uTarget, vUv).xyz + exp(-dot(p, p) / radius) * color, 1.0);
}
`);

const advectionShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 texelSize;
uniform vec2 dyeTexelSize;
uniform float dt;
uniform float dissipation;
out vec4 fc;
void main() {
    vec2 coord = vUv - dt * texture(uVelocity, vUv).xy * texelSize;
    fc = texture(uSource, coord) / (1.0 + dissipation * dt);
}
`);

const divergenceShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision mediump float;
in vec2 vUv, vL, vR, vT, vB;
uniform sampler2D uVelocity;
out vec4 fc;
void main() {
    float L = texture(uVelocity, vL).x;
    float R = texture(uVelocity, vR).x;
    float T = texture(uVelocity, vT).y;
    float B = texture(uVelocity, vB).y;
    vec2 C = texture(uVelocity, vUv).xy;
    if (vL.x < 0.0) L = -C.x;
    if (vR.x > 1.0) R = -C.x;
    if (vT.y > 1.0) T = -C.y;
    if (vB.y < 0.0) B = -C.y;
    fc = vec4(0.5 * (R - L + T - B), 0.0, 0.0, 1.0);
}
`);

const curlShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision mediump float;
in vec2 vUv, vL, vR, vT, vB;
uniform sampler2D uVelocity;
out vec4 fc;
void main() {
    float L = texture(uVelocity, vL).y;
    float R = texture(uVelocity, vR).y;
    float T = texture(uVelocity, vT).x;
    float B = texture(uVelocity, vB).x;
    fc = vec4(0.5 * (R - L - T + B), 0.0, 0.0, 1.0);
}
`);

const vorticityShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
in vec2 vUv, vL, vR, vT, vB;
uniform sampler2D uVelocity;
uniform sampler2D uCurl;
uniform float curl;
uniform float dt;
out vec4 fc;
void main() {
    float L = texture(uCurl, vL).x;
    float R = texture(uCurl, vR).x;
    float T = texture(uCurl, vT).x;
    float B = texture(uCurl, vB).x;
    float C = texture(uCurl, vUv).x;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force /= length(force) + 0.0001;
    force *= curl * C;
    force.y *= -1.0;
    vec2 v = texture(uVelocity, vUv).xy + force * dt;
    fc = vec4(clamp(v, vec2(-1000.0), vec2(1000.0)), 0.0, 1.0);
}
`);

const pressureShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision mediump float;
in vec2 vUv, vL, vR, vT, vB;
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
out vec4 fc;
void main() {
    float L = texture(uPressure, vL).x;
    float R = texture(uPressure, vR).x;
    float T = texture(uPressure, vT).x;
    float B = texture(uPressure, vB).x;
    float d = texture(uDivergence, vUv).x;
    fc = vec4((L + R + B + T - d) * 0.25, 0.0, 0.0, 1.0);
}
`);

const gradientSubtractShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision mediump float;
in vec2 vUv, vL, vR, vT, vB;
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
out vec4 fc;
void main() {
    float L = texture(uPressure, vL).x;
    float R = texture(uPressure, vR).x;
    float T = texture(uPressure, vT).x;
    float B = texture(uPressure, vB).x;
    vec2 v = texture(uVelocity, vUv).xy - vec2(R - L, T - B);
    fc = vec4(v, 0.0, 1.0);
}
`);

// ---------------------------------------------------------------------------
// Fragment Shaders: Noise Warp (GPU-only)
// ---------------------------------------------------------------------------

// Initialize noise FBO with GPU-generated Gaussian noise (PCG hash + Box-Muller)
const noiseInitShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
in vec2 vUv;
uniform vec2 resolution;
uniform float seed;
out vec4 fc;

uint pcg(uint x) {
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand(uint s) { return float(pcg(s)) / 4294967296.0; }

vec2 boxMuller(uint s1, uint s2) {
    float u1 = max(rand(s1), 1e-10);
    float u2 = rand(s2);
    float r = sqrt(-2.0 * log(u1));
    return vec2(r * cos(6.2831853 * u2), r * sin(6.2831853 * u2));
}

void main() {
    ivec2 px = ivec2(vUv * resolution);
    uint base = uint(px.y) * uint(resolution.x) + uint(px.x);
    base += uint(seed * 1000.0) * 999983u;
    vec2 n1 = boxMuller(base * 4u + 0u, base * 4u + 1u);
    vec2 n2 = boxMuller(base * 4u + 2u, base * 4u + 3u);
    fc = vec4(n1.x, n1.y, n2.x, n2.y);
}
`);

// Advect noise through velocity field + add fresh noise to uncovered areas.
// Uses the velocity field (at sim resolution) to push the noise texture forward.
// The fresh noise injection prevents variance collapse in stretched regions.
const noiseAdvectShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uNoise;       // previous frame noise
uniform sampler2D uVelocity;    // fluid velocity field
uniform vec2 velTexelSize;      // 1/simRes for velocity conversion
uniform float dt;
uniform vec2 noiseTexelSize;    // 1/noiseRes
uniform float seed;
out vec4 fc;

uint pcg(uint x) {
    uint state = x * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand(uint s) { return float(pcg(s)) / 4294967296.0; }

vec2 boxMuller(uint s1, uint s2) {
    float u1 = max(rand(s1), 1e-10);
    float u2 = rand(s2);
    float r = sqrt(-2.0 * log(u1));
    return vec2(r * cos(6.2831853 * u2), r * sin(6.2831853 * u2));
}

void main() {
    // Sample velocity at this position
    vec2 vel = texture(uVelocity, vUv).xy;
    // Convert velocity to UV displacement: vel is in grid cells/sec
    vec2 uvDisp = vel * dt * velTexelSize;
    // Trace back: where did this pixel come from?
    vec2 srcUv = vUv - uvDisp;
    // Sample noise at source position (bilinear)
    vec4 advected = texture(uNoise, srcUv);

    // Detect out-of-bounds or very stretched regions: inject fresh noise
    // If source UV is outside [0,1], use fresh noise
    float oob = step(0.0, srcUv.x) * step(srcUv.x, 1.0)
              * step(0.0, srcUv.y) * step(srcUv.y, 1.0);

    // Generate fresh noise for this pixel
    ivec2 px = ivec2(vUv / noiseTexelSize);
    uint base = uint(px.y * 4096 + px.x) + uint(seed);
    vec2 n1 = boxMuller(base * 4u + 0u, base * 4u + 1u);
    vec2 n2 = boxMuller(base * 4u + 2u, base * 4u + 3u);
    vec4 fresh = vec4(n1.x, n1.y, n2.x, n2.y);

    // Mix: use advected where in bounds, fresh where out of bounds
    fc = mix(fresh, advected, oob);
}
`);

// Normalize noise: rescale to restore mean=0, std=1.
// Uses stats computed from GPU reduction.
const noiseNormalizeShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uNoise;
uniform vec4 stats; // x=mean, y=std, z=0, w=0
out vec4 fc;
void main() {
    vec4 n = texture(uNoise, vUv);
    // Rescale: (n - mean) / std → gives mean=0, std=1
    float invStd = 1.0 / max(stats.y, 0.001);
    fc = (n - stats.x) * invStd;
}
`);

// GPU reduction: compute sum and sum-of-squares for noise statistics.
// Each texel computes local stats for a tile and writes to a smaller FBO.
const noiseReduceShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uNoise;
uniform vec2 srcResolution;  // resolution of source texture
out vec4 fc;
void main() {
    // Each output pixel covers a tile of source pixels
    // Compute sum and sum² over a 2x2 block of source texels
    vec2 ts = 1.0 / srcResolution;
    vec2 base = floor(vUv * srcResolution * 0.5) * 2.0 * ts + ts * 0.5;
    float sum = 0.0;
    float sum2 = 0.0;
    float count = 0.0;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            vec2 uv = base + vec2(float(dx), float(dy)) * ts;
            vec4 v = texture(uNoise, uv);
            // Average all 4 channels for a single stat
            float val = (v.r + v.g + v.b + v.a) * 0.25;
            sum += val;
            sum2 += val * val;
            count += 1.0;
        }
    }
    fc = vec4(sum, sum2, count, 0.0);
}
`);

// Second reduction stage: accumulate stats from previous reduction
const noiseReduceAccumShader = compileShader(gl.FRAGMENT_SHADER, `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uStats;
uniform vec2 srcResolution;
out vec4 fc;
void main() {
    vec2 ts = 1.0 / srcResolution;
    vec2 base = floor(vUv * srcResolution * 0.5) * 2.0 * ts + ts * 0.5;
    float sum = 0.0;
    float sum2 = 0.0;
    float count = 0.0;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            vec4 v = texture(uStats, base + vec2(float(dx), float(dy)) * ts);
            sum += v.r;
            sum2 += v.g;
            count += v.b;
        }
    }
    fc = vec4(sum, sum2, count, 0.0);
}
`);

// ---------------------------------------------------------------------------
// Fragment Shader: Display (noise warp modes)
// ---------------------------------------------------------------------------

const displayShaderSource = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uDye;
uniform sampler2D uNoise;
uniform sampler2D uVelocity;
uniform int uMode;
uniform int uGreyscale;
uniform int uUniform;
uniform int uThreshOn;
uniform float uThreshVal;
uniform float uNoiseOpacity;
out vec4 fc;

float erf_approx(float x) {
    float ax = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * ax);
    float poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    float r = 1.0 - poly * exp(-ax * ax);
    return x >= 0.0 ? r : -r;
}
float normal_cdf(float x) { return 0.5 * (1.0 + erf_approx(x * 0.7071067812)); }
float luminance(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

vec3 noiseToDisplay(vec3 n) {
    vec3 v = n;
    if (uUniform == 1) v = vec3(normal_cdf(v.x), normal_cdf(v.y), normal_cdf(v.z));
    if (uGreyscale == 1) v = vec3(v.x, v.x, v.x);
    if (uUniform == 1) return v;
    return v / 5.0 + 0.5;
}

vec4 applyThreshold(vec4 c) {
    if (uThreshOn == 0) return c;
    return c.r > uThreshVal ? vec4(1.0) : vec4(vec3(0.0), 1.0);
}

void main() {
    vec3 n = texture(uNoise, vUv).rgb;
    if (uMode == 0) {  // noise
        fc = applyThreshold(vec4(noiseToDisplay(n), 1.0));
    } else if (uMode == 1) {  // scene (fluid dye)
        fc = vec4(texture(uDye, vUv).rgb, 1.0);
    } else if (uMode == 2) {  // scene + noise
        vec3 scene = texture(uDye, vUv).rgb;
        vec3 nc;
        if (uUniform == 1) nc = (vec3(normal_cdf(n.x), normal_cdf(n.y), normal_cdf(n.z)) - 0.5) * uNoiseOpacity;
        else nc = n * uNoiseOpacity;
        fc = vec4(clamp(scene + nc, 0.0, 1.0), 1.0);
    } else if (uMode == 3) {  // dither
        vec3 scene = texture(uDye, vUv).rgb;
        float lum = luminance(scene);
        float thr = normal_cdf(n.x);
        fc = lum > thr ? vec4(1.0) : vec4(vec3(0.0), 1.0);
    } else if (uMode == 4) {  // motion (velocity)
        vec2 vel = texture(uVelocity, vUv).rg;
        fc = vec4(vel * 0.001 + 0.5, 0.5, 1.0);
    } else {  // raw
        fc = applyThreshold(vec4(noiseToDisplay(n), 1.0));
    }
}
`;

// ---------------------------------------------------------------------------
// Build Programs
// ---------------------------------------------------------------------------

const copyProgram = new Program(baseVertexShader, copyShader);
const clearProgram = new Program(baseVertexShader, clearShader);
const splatProgram = new Program(baseVertexShader, splatShader);
const advectionProgram = new Program(baseVertexShader, advectionShader);
const divergenceProgram = new Program(baseVertexShader, divergenceShader);
const curlProgram = new Program(baseVertexShader, curlShader);
const vorticityProgram = new Program(baseVertexShader, vorticityShader);
const pressureProgram = new Program(baseVertexShader, pressureShader);
const gradientSubtractProgram = new Program(baseVertexShader, gradientSubtractShader);
const displayProgram = new Program(baseVertexShader,
    compileShader(gl.FRAGMENT_SHADER, displayShaderSource));
const noiseInitProgram = new Program(baseVertexShader, noiseInitShader);
const noiseAdvectProgram = new Program(baseVertexShader, noiseAdvectShader);
const noiseNormalizeProgram = new Program(baseVertexShader, noiseNormalizeShader);
const noiseReduceProgram = new Program(baseVertexShader, noiseReduceShader);
const noiseReduceAccumProgram = new Program(baseVertexShader, noiseReduceAccumShader);

// ---------------------------------------------------------------------------
// Geometry: fullscreen quad
// ---------------------------------------------------------------------------

gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, -1,1, 1,1, 1,-1]), gl.STATIC_DRAW);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0,1,2, 0,2,3]), gl.STATIC_DRAW);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
gl.enableVertexAttribArray(0);

function blit(target, clear) {
    if (!target) {
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    } else {
        gl.viewport(0, 0, target.width, target.height);
        gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    }
    if (clear) { gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT); }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
}

// ---------------------------------------------------------------------------
// FBO helpers
// ---------------------------------------------------------------------------

function createFBO(w, h, internalFormat, format, type, param) {
    gl.activeTexture(gl.TEXTURE0);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);
    return {
        texture, fbo, width: w, height: h,
        texelSizeX: 1.0 / w, texelSizeY: 1.0 / h,
        attach(id) { gl.activeTexture(gl.TEXTURE0 + id); gl.bindTexture(gl.TEXTURE_2D, texture); return id; }
    };
}

function createDoubleFBO(w, h, internalFormat, format, type, param) {
    let fbo1 = createFBO(w, h, internalFormat, format, type, param);
    let fbo2 = createFBO(w, h, internalFormat, format, type, param);
    return {
        width: w, height: h, texelSizeX: fbo1.texelSizeX, texelSizeY: fbo1.texelSizeY,
        get read() { return fbo1; }, set read(v) { fbo1 = v; },
        get write() { return fbo2; }, set write(v) { fbo2 = v; },
        swap() { const t = fbo1; fbo1 = fbo2; fbo2 = t; }
    };
}

function resizeFBO(target, w, h, iF, f, t, p) {
    const n = createFBO(w, h, iF, f, t, p);
    copyProgram.bind();
    gl.uniform2f(copyProgram.uniforms.texelSize, target.texelSizeX, target.texelSizeY);
    gl.uniform1i(copyProgram.uniforms.uTexture, target.attach(0));
    blit(n);
    return n;
}

function resizeDoubleFBO(target, w, h, iF, f, t, p) {
    if (target.width === w && target.height === h) return target;
    target.read = resizeFBO(target.read, w, h, iF, f, t, p);
    target.write = createFBO(w, h, iF, f, t, p);
    target.width = w; target.height = h;
    target.texelSizeX = 1.0 / w; target.texelSizeY = 1.0 / h;
    return target;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let dye, velocity, divergenceFBO, curlFBO, pressureFBO;
let noiseFBO;          // double FBO for noise ping-pong (RGBA32F at noiseRes)
let reduceChain = [];  // chain of FBOs for GPU reduction
const settings = loadSettings();
let displayMode = 3;   // dither
let paused = false;
let noiseLocked = false;
let noiseRes = NOISE_RESOLUTIONS[settings.noiseResIdx];
let noiseSeed = 42;
let noiseStats = { mean: 0, std: 1 };  // tracked from GPU reduction
const PRESSURE_ITERATIONS = 20;
const SPLAT_FORCE = 6000;
const SPLAT_RADIUS = 0.25;

function getResolution(resolution) {
    let ar = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (ar < 1) ar = 1.0 / ar;
    const min = Math.round(resolution);
    const max = Math.round(resolution * ar);
    return gl.drawingBufferWidth > gl.drawingBufferHeight
        ? { width: max, height: min }
        : { width: min, height: max };
}

function initFramebuffers() {
    const simRes = getResolution(settings.simRes);
    const dyeRes = getResolution(settings.dyeRes);
    const filtering = supportsLinear ? gl.LINEAR : gl.NEAREST;

    gl.disable(gl.BLEND);

    if (!dye)
        dye = createDoubleFBO(dyeRes.width, dyeRes.height, formatRGBA.internalFormat, formatRGBA.format, halfFloatTexType, filtering);
    else
        dye = resizeDoubleFBO(dye, dyeRes.width, dyeRes.height, formatRGBA.internalFormat, formatRGBA.format, halfFloatTexType, filtering);

    if (!velocity)
        velocity = createDoubleFBO(simRes.width, simRes.height, formatRG.internalFormat, formatRG.format, halfFloatTexType, filtering);
    else
        velocity = resizeDoubleFBO(velocity, simRes.width, simRes.height, formatRG.internalFormat, formatRG.format, halfFloatTexType, filtering);

    divergenceFBO = createFBO(simRes.width, simRes.height, formatR.internalFormat, formatR.format, halfFloatTexType, gl.NEAREST);
    curlFBO = createFBO(simRes.width, simRes.height, formatR.internalFormat, formatR.format, halfFloatTexType, gl.NEAREST);
    pressureFBO = createDoubleFBO(simRes.width, simRes.height, formatR.internalFormat, formatR.format, halfFloatTexType, gl.NEAREST);
}

function initNoiseFBOs() {
    // Noise uses RGBA32F for full precision Gaussian values
    noiseFBO = createDoubleFBO(noiseRes, noiseRes, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.LINEAR);

    // Initialize with GPU-generated Gaussian noise
    noiseInitProgram.bind();
    gl.uniform2f(noiseInitProgram.uniforms.texelSize, 1.0/noiseRes, 1.0/noiseRes);
    gl.uniform2f(noiseInitProgram.uniforms.resolution, noiseRes, noiseRes);
    gl.uniform1f(noiseInitProgram.uniforms.seed, Math.random() * 1000);
    blit(noiseFBO.write);
    noiseFBO.swap();

    // Build reduction chain: noiseRes → noiseRes/2 → ... → 1
    reduceChain = [];
    let s = Math.ceil(noiseRes / 2);
    while (s >= 1) {
        reduceChain.push(createFBO(s, s, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST));
        if (s === 1) break;
        s = Math.ceil(s / 2);
    }
}

// ---------------------------------------------------------------------------
// Pointer / splat
// ---------------------------------------------------------------------------

class Pointer {
    constructor() {
        this.id = -1; this.down = false; this.moved = false;
        this.texcoordX = 0; this.texcoordY = 0;
        this.prevTexcoordX = 0; this.prevTexcoordY = 0;
        this.deltaX = 0; this.deltaY = 0;
        this.color = { r: 0, g: 0, b: 0 };
    }
}

const pointers = [new Pointer()];
let splatStack = [];

function scaleByPixelRatio(input) {
    return Math.floor(input * (window.devicePixelRatio || 1));
}

function correctRadius(radius) {
    const ar = canvas.width / canvas.height;
    return ar > 1 ? radius * ar : radius;
}

function HSVtoRGB(h, s, v) {
    let r, g, b;
    const i = Math.floor(h * 6), f = h * 6 - i;
    const p = v * (1-s), q = v * (1-f*s), t = v * (1-(1-f)*s);
    switch (i % 6) {
        case 0: r=v;g=t;b=p; break; case 1: r=q;g=v;b=p; break;
        case 2: r=p;g=v;b=t; break; case 3: r=p;g=q;b=v; break;
        case 4: r=t;g=p;b=v; break; case 5: r=v;g=p;b=q; break;
    }
    return { r, g, b };
}

function generateColor() {
    const c = HSVtoRGB(Math.random(), 1.0, 1.0);
    c.r *= 0.15; c.g *= 0.15; c.b *= 0.15;
    return c;
}

function splat(x, y, dx, dy, color) {
    splatProgram.bind();
    gl.uniform2f(splatProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(splatProgram.uniforms.uTarget, velocity.read.attach(0));
    gl.uniform1f(splatProgram.uniforms.aspectRatio, canvas.width / canvas.height);
    gl.uniform2f(splatProgram.uniforms.point, x, y);
    gl.uniform3f(splatProgram.uniforms.color, dx, dy, 0.0);
    gl.uniform1f(splatProgram.uniforms.radius, correctRadius(SPLAT_RADIUS / 100.0));
    blit(velocity.write);
    velocity.swap();

    gl.uniform1i(splatProgram.uniforms.uTarget, dye.read.attach(0));
    gl.uniform3f(splatProgram.uniforms.color, color.r, color.g, color.b);
    blit(dye.write);
    dye.swap();
}

function multipleSplats(amount) {
    for (let i = 0; i < amount; i++) {
        const color = generateColor();
        color.r *= 10; color.g *= 10; color.b *= 10;
        splat(Math.random(), Math.random(),
              1000 * (Math.random() - 0.5), 1000 * (Math.random() - 0.5), color);
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

canvas.addEventListener('mousedown', e => {
    const p = pointers[0];
    p.down = true; p.moved = false;
    p.texcoordX = scaleByPixelRatio(e.offsetX) / canvas.width;
    p.texcoordY = 1.0 - scaleByPixelRatio(e.offsetY) / canvas.height;
    p.prevTexcoordX = p.texcoordX; p.prevTexcoordY = p.texcoordY;
    p.deltaX = 0; p.deltaY = 0; p.color = generateColor();
});

canvas.addEventListener('mousemove', e => {
    const p = pointers[0];
    if (!p.down) return;
    p.prevTexcoordX = p.texcoordX; p.prevTexcoordY = p.texcoordY;
    p.texcoordX = scaleByPixelRatio(e.offsetX) / canvas.width;
    p.texcoordY = 1.0 - scaleByPixelRatio(e.offsetY) / canvas.height;
    let dx = p.texcoordX - p.prevTexcoordX, dy = p.texcoordY - p.prevTexcoordY;
    const ar = canvas.width / canvas.height;
    if (ar < 1) dx *= ar; else dy /= ar;
    p.deltaX = dx; p.deltaY = dy;
    p.moved = Math.abs(dx) > 0 || Math.abs(dy) > 0;
});

window.addEventListener('mouseup', () => { pointers[0].down = false; });

canvas.addEventListener('touchstart', e => {
    e.preventDefault();
    const t = e.targetTouches;
    while (t.length >= pointers.length) pointers.push(new Pointer());
    for (let i = 0; i < t.length; i++) {
        const p = pointers[i + 1] || pointers[0];
        p.id = t[i].identifier; p.down = true; p.moved = false;
        p.texcoordX = scaleByPixelRatio(t[i].pageX) / canvas.width;
        p.texcoordY = 1.0 - scaleByPixelRatio(t[i].pageY) / canvas.height;
        p.prevTexcoordX = p.texcoordX; p.prevTexcoordY = p.texcoordY;
        p.deltaX = 0; p.deltaY = 0; p.color = generateColor();
    }
});

canvas.addEventListener('touchmove', e => {
    e.preventDefault();
    const t = e.targetTouches;
    for (let i = 0; i < t.length; i++) {
        const p = pointers[i + 1] || pointers[0];
        if (!p.down) continue;
        p.prevTexcoordX = p.texcoordX; p.prevTexcoordY = p.texcoordY;
        p.texcoordX = scaleByPixelRatio(t[i].pageX) / canvas.width;
        p.texcoordY = 1.0 - scaleByPixelRatio(t[i].pageY) / canvas.height;
        let dx = p.texcoordX - p.prevTexcoordX, dy = p.texcoordY - p.prevTexcoordY;
        const ar = canvas.width / canvas.height;
        if (ar < 1) dx *= ar; else dy /= ar;
        p.deltaX = dx; p.deltaY = dy;
        p.moved = Math.abs(dx) > 0 || Math.abs(dy) > 0;
    }
}, false);

window.addEventListener('touchend', e => {
    for (const t of e.changedTouches) {
        const p = pointers.find(pp => pp.id === t.identifier);
        if (p) p.down = false;
    }
});

// ---------------------------------------------------------------------------
// Fluid simulation step
// ---------------------------------------------------------------------------

function fluidStep(dt) {
    gl.disable(gl.BLEND);

    curlProgram.bind();
    gl.uniform2f(curlProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(curlProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(curlFBO);

    vorticityProgram.bind();
    gl.uniform2f(vorticityProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(vorticityProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(vorticityProgram.uniforms.uCurl, curlFBO.attach(1));
    gl.uniform1f(vorticityProgram.uniforms.curl, settings.curl);
    gl.uniform1f(vorticityProgram.uniforms.dt, dt);
    blit(velocity.write);
    velocity.swap();

    divergenceProgram.bind();
    gl.uniform2f(divergenceProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(divergenceProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(divergenceFBO);

    clearProgram.bind();
    gl.uniform2f(clearProgram.uniforms.texelSize, pressureFBO.texelSizeX, pressureFBO.texelSizeY);
    gl.uniform1i(clearProgram.uniforms.uTexture, pressureFBO.read.attach(0));
    gl.uniform1f(clearProgram.uniforms.value, settings.pressure / 100);
    blit(pressureFBO.write);
    pressureFBO.swap();

    pressureProgram.bind();
    gl.uniform2f(pressureProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(pressureProgram.uniforms.uDivergence, divergenceFBO.attach(0));
    for (let i = 0; i < PRESSURE_ITERATIONS; i++) {
        gl.uniform1i(pressureProgram.uniforms.uPressure, pressureFBO.read.attach(1));
        blit(pressureFBO.write);
        pressureFBO.swap();
    }

    gradientSubtractProgram.bind();
    gl.uniform2f(gradientSubtractProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(gradientSubtractProgram.uniforms.uPressure, pressureFBO.read.attach(0));
    gl.uniform1i(gradientSubtractProgram.uniforms.uVelocity, velocity.read.attach(1));
    blit(velocity.write);
    velocity.swap();

    advectionProgram.bind();
    gl.uniform2f(advectionProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(advectionProgram.uniforms.uSource, velocity.read.attach(0));
    gl.uniform1f(advectionProgram.uniforms.dt, dt);
    gl.uniform1f(advectionProgram.uniforms.dissipation, settings.dissipation / 100);
    blit(velocity.write);
    velocity.swap();

    gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, dye.texelSizeX, dye.texelSizeY);
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(advectionProgram.uniforms.uSource, dye.read.attach(1));
    gl.uniform1f(advectionProgram.uniforms.dissipation, settings.dissipation / 100);
    blit(dye.write);
    dye.swap();
}

// ---------------------------------------------------------------------------
// Noise warp step (all GPU — no readback)
// ---------------------------------------------------------------------------

function noiseWarpStep(dt) {
    if (noiseLocked) return;
    gl.disable(gl.BLEND);

    // Advect noise through velocity field
    noiseAdvectProgram.bind();
    gl.uniform2f(noiseAdvectProgram.uniforms.texelSize, 1.0/noiseRes, 1.0/noiseRes);
    gl.uniform1i(noiseAdvectProgram.uniforms.uNoise, noiseFBO.read.attach(0));
    gl.uniform1i(noiseAdvectProgram.uniforms.uVelocity, velocity.read.attach(1));
    gl.uniform2f(noiseAdvectProgram.uniforms.velTexelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1f(noiseAdvectProgram.uniforms.dt, dt);
    gl.uniform2f(noiseAdvectProgram.uniforms.noiseTexelSize, 1.0/noiseRes, 1.0/noiseRes);
    noiseSeed = (noiseSeed + 1) | 0;
    gl.uniform1f(noiseAdvectProgram.uniforms.seed, noiseSeed * 999983.0);
    blit(noiseFBO.write);
    noiseFBO.swap();

    // GPU reduction to compute noise mean/std (every 4 frames for perf)
    if (noiseSeed % 4 === 0) {
        computeNoiseStats();
    }

    // Normalize noise to restore mean=0, std=1
    if (noiseStats.std > 0.01) {
        noiseNormalizeProgram.bind();
        gl.uniform2f(noiseNormalizeProgram.uniforms.texelSize, 1.0/noiseRes, 1.0/noiseRes);
        gl.uniform1i(noiseNormalizeProgram.uniforms.uNoise, noiseFBO.read.attach(0));
        gl.uniform4f(noiseNormalizeProgram.uniforms.stats, noiseStats.mean, noiseStats.std, 0, 0);
        blit(noiseFBO.write);
        noiseFBO.swap();
    }
}

function computeNoiseStats() {
    if (reduceChain.length === 0) return;

    gl.disable(gl.BLEND);

    // First reduction: noise → reduceChain[0]
    noiseReduceProgram.bind();
    gl.uniform2f(noiseReduceProgram.uniforms.texelSize,
        reduceChain[0].texelSizeX, reduceChain[0].texelSizeY);
    gl.uniform2f(noiseReduceProgram.uniforms.srcResolution, noiseRes, noiseRes);
    gl.uniform1i(noiseReduceProgram.uniforms.uNoise, noiseFBO.read.attach(0));
    blit(reduceChain[0]);

    // Cascade reductions
    for (let i = 1; i < reduceChain.length; i++) {
        const src = reduceChain[i - 1];
        const dst = reduceChain[i];
        noiseReduceAccumProgram.bind();
        gl.uniform2f(noiseReduceAccumProgram.uniforms.texelSize, dst.texelSizeX, dst.texelSizeY);
        gl.uniform2f(noiseReduceAccumProgram.uniforms.srcResolution, src.width, src.height);
        gl.uniform1i(noiseReduceAccumProgram.uniforms.uStats, src.attach(0));
        blit(dst);
    }

    // Read back the 1x1 final result
    const last = reduceChain[reduceChain.length - 1];
    gl.bindFramebuffer(gl.FRAMEBUFFER, last.fbo);
    const px = new Float32Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);

    const sum = px[0], sum2 = px[1], count = px[2];
    if (count > 0) {
        const mean = sum / count;
        const variance = sum2 / count - mean * mean;
        noiseStats.mean = mean;
        noiseStats.std = Math.sqrt(Math.max(0, variance));
    }
}

// ---------------------------------------------------------------------------
// Render display
// ---------------------------------------------------------------------------

function render() {
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.enable(gl.BLEND);

    displayProgram.bind();
    gl.uniform2f(displayProgram.uniforms.texelSize, 1.0/gl.drawingBufferWidth, 1.0/gl.drawingBufferHeight);
    gl.uniform1i(displayProgram.uniforms.uDye, dye.read.attach(0));
    gl.uniform1i(displayProgram.uniforms.uNoise, noiseFBO.read.attach(1));
    gl.uniform1i(displayProgram.uniforms.uVelocity, velocity.read.attach(2));
    gl.uniform1i(displayProgram.uniforms.uMode, displayMode);
    gl.uniform1i(displayProgram.uniforms.uGreyscale, settings.greyscale ? 1 : 0);
    gl.uniform1i(displayProgram.uniforms.uUniform, settings.uniformDisplay ? 1 : 0);
    gl.uniform1i(displayProgram.uniforms.uThreshOn, settings.threshOn ? 1 : 0);
    gl.uniform1f(displayProgram.uniforms.uThreshVal, settings.threshSlider / 1000);
    gl.uniform1f(displayProgram.uniforms.uNoiseOpacity, settings.noiseOpacity / 100);
    blit(null);
}

// ---------------------------------------------------------------------------
// Canvas resize
// ---------------------------------------------------------------------------

function resizeCanvas() {
    const dpr = settings.retina ? (window.devicePixelRatio || 1) : 1;
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

let lastUpdateTime = Date.now();
let frameCount = 0;
let lastFPSTime = Date.now();
let currentFPS = 0;
let lastWarpMs = 0;

function update() {
    let now = Date.now();
    let dt = (now - lastUpdateTime) / 1000;
    dt = Math.min(dt, 0.016666);
    lastUpdateTime = now;

    if (resizeCanvas()) initFramebuffers();

    if (splatStack.length > 0) multipleSplats(splatStack.pop());
    pointers.forEach(p => {
        if (p.moved) {
            p.moved = false;
            splat(p.texcoordX, p.texcoordY,
                  p.deltaX * SPLAT_FORCE, p.deltaY * SPLAT_FORCE, p.color);
        }
    });

    if (!paused) {
        fluidStep(dt);
        const t0 = performance.now();
        noiseWarpStep(dt);
        lastWarpMs = performance.now() - t0;
    }

    render();

    frameCount++;
    now = Date.now();
    if (now - lastFPSTime >= 1000) {
        currentFPS = frameCount;
        frameCount = 0;
        lastFPSTime = now;
    }

    if (frameCount % 30 === 0) {
        document.getElementById('stats').textContent =
            `FPS: ${currentFPS} | noise: ${noiseRes}x${noiseRes} | ` +
            `sim: ${velocity ? velocity.width : '?'} | ` +
            `mean: ${noiseStats.mean.toFixed(3)} | std: ${noiseStats.std.toFixed(3)} | ` +
            `mode: ${MODE_NAMES[displayMode]} | warp: ${lastWarpMs.toFixed(1)}ms`;
    }

    requestAnimationFrame(update);
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

function syncUI() {
    const s = settings;
    const el = id => document.getElementById(id);
    el('resBtn').textContent = `[P] ${NOISE_RESOLUTIONS[s.noiseResIdx]}`;
    el('greyBtn').textContent = `[G] Grey: ${s.greyscale ? 'ON' : 'OFF'}`;
    el('greyBtn').classList.toggle('on', s.greyscale);
    el('uniformBtn').textContent = `[U] Uniform: ${s.uniformDisplay ? 'ON' : 'OFF'}`;
    el('uniformBtn').classList.toggle('on', s.uniformDisplay);
    el('retinaBtn').textContent = `[T] Retina: ${s.retina ? 'ON' : 'OFF'}`;
    el('retinaBtn').classList.toggle('on', s.retina);
    el('interpBtn').textContent = `[I] Interp: ${s.bilinear ? 'Bilinear' : 'Nearest'}`;
    el('interpBtn').classList.toggle('on', !s.bilinear);
    el('roundBtn').textContent = `[O] Round: ${ROUND_MODES[s.roundMode]}`;
    el('roundBtn').classList.toggle('on', s.roundMode > 0);
    el('blueNoiseBtn').textContent = `[B] Blue: ${s.blueNoise ? 'ON' : 'OFF'}`;
    el('blueNoiseBtn').classList.toggle('on', s.blueNoise);
    el('bnItersBtn').textContent = `[N] BN\u00d7${BN_ITERS_OPTIONS[s.bnItersIdx]}`;
    el('threshBtn').textContent = `[H] Thresh: ${s.threshOn ? 'ON' : 'OFF'}`;
    el('threshBtn').classList.toggle('on', s.threshOn);
    el('threshSlider').value = s.threshSlider;
    el('threshSlider').disabled = !s.threshOn;
    el('threshLabel').textContent = s.threshOn ? (s.threshSlider / 1000).toFixed(2) : '';
    el('lockNoiseBtn').textContent = `[L] Lock: ${noiseLocked ? 'ON' : 'OFF'}`;
    el('lockNoiseBtn').classList.toggle('on', noiseLocked);
    el('pauseBtn').textContent = `[Space] Pause: ${paused ? 'ON' : 'OFF'}`;
    el('pauseBtn').classList.toggle('on', paused);
    el('modeSettings').style.display = displayMode === 2 ? 'inline' : 'none';
    el('noiseOpacitySlider').value = s.noiseOpacity;
    el('noiseOpacityLabel').textContent = (s.noiseOpacity / 100).toFixed(2);
    el('curlSlider').value = s.curl;
    el('curlLabel').textContent = s.curl;
    el('dissipSlider').value = s.dissipation;
    el('dissipLabel').textContent = (s.dissipation / 100).toFixed(2);
    el('pressureSlider').value = s.pressure;
    el('pressureLabel').textContent = (s.pressure / 100).toFixed(2);
    el('simResSelect').value = s.simRes;
    el('dyeResSelect').value = s.dyeRes;
    canvas.style.imageRendering = s.bilinear ? 'auto' : 'pixelated';

    document.querySelectorAll('.modeBtn').forEach(btn => {
        btn.classList.toggle('on', parseInt(btn.dataset.mode) === displayMode);
    });
}

function persist() { saveSettings(settings); }

function setupUI() {
    const el = id => document.getElementById(id);

    el('resBtn').addEventListener('click', () => {
        settings.noiseResIdx = (settings.noiseResIdx + 1) % NOISE_RESOLUTIONS.length;
        noiseRes = NOISE_RESOLUTIONS[settings.noiseResIdx];
        initNoiseFBOs();
        persist(); syncUI();
    });

    el('greyBtn').addEventListener('click', () => { settings.greyscale = !settings.greyscale; persist(); syncUI(); });
    el('uniformBtn').addEventListener('click', () => { settings.uniformDisplay = !settings.uniformDisplay; persist(); syncUI(); });
    el('retinaBtn').addEventListener('click', () => {
        settings.retina = !settings.retina; persist(); syncUI();
        resizeCanvas(); initFramebuffers();
    });
    el('interpBtn').addEventListener('click', () => { settings.bilinear = !settings.bilinear; persist(); syncUI(); });
    el('roundBtn').addEventListener('click', () => {
        settings.roundMode = (settings.roundMode + 1) % ROUND_MODES.length; persist(); syncUI();
    });
    el('blueNoiseBtn').addEventListener('click', () => { settings.blueNoise = !settings.blueNoise; persist(); syncUI(); });
    el('bnItersBtn').addEventListener('click', () => {
        settings.bnItersIdx = (settings.bnItersIdx + 1) % BN_ITERS_OPTIONS.length; persist(); syncUI();
    });
    el('threshBtn').addEventListener('click', () => { settings.threshOn = !settings.threshOn; persist(); syncUI(); });
    el('threshSlider').addEventListener('input', function() { settings.threshSlider = parseInt(this.value); persist(); syncUI(); });
    el('lockNoiseBtn').addEventListener('click', () => { noiseLocked = !noiseLocked; syncUI(); });
    el('pauseBtn').addEventListener('click', () => { paused = !paused; syncUI(); });
    el('splatBtn').addEventListener('click', () => { splatStack.push(Math.floor(Math.random() * 20) + 5); });
    el('noiseOpacitySlider').addEventListener('input', function() { settings.noiseOpacity = parseInt(this.value); persist(); syncUI(); });
    el('curlSlider').addEventListener('input', function() { settings.curl = parseInt(this.value); persist(); syncUI(); });
    el('dissipSlider').addEventListener('input', function() { settings.dissipation = parseInt(this.value); persist(); syncUI(); });
    el('pressureSlider').addEventListener('input', function() { settings.pressure = parseInt(this.value); persist(); syncUI(); });
    el('simResSelect').addEventListener('change', function() {
        settings.simRes = parseInt(this.value); persist(); syncUI(); initFramebuffers();
    });
    el('dyeResSelect').addEventListener('change', function() {
        settings.dyeRes = parseInt(this.value); persist(); syncUI(); initFramebuffers();
    });
    el('resetBtn').addEventListener('click', () => {
        localStorage.removeItem(STORAGE_KEY);
        Object.assign(settings, DEFAULTS);
        noiseRes = NOISE_RESOLUTIONS[settings.noiseResIdx];
        paused = false; noiseLocked = false;
        initNoiseFBOs(); initFramebuffers();
        multipleSplats(Math.floor(Math.random() * 20) + 5);
        persist(); syncUI();
    });

    document.querySelectorAll('.modeBtn').forEach(btn => {
        btn.addEventListener('click', () => { displayMode = parseInt(btn.dataset.mode); syncUI(); });
    });

    document.addEventListener('keydown', e => {
        if (e.metaKey || e.ctrlKey) return;
        if (e.code >= 'Digit1' && e.code <= 'Digit6') { displayMode = parseInt(e.code.slice(5)) - 1; syncUI(); }
        if (e.code === 'KeyP') el('resBtn').click();
        if (e.code === 'KeyG') el('greyBtn').click();
        if (e.code === 'KeyU') el('uniformBtn').click();
        if (e.code === 'KeyT') el('retinaBtn').click();
        if (e.code === 'KeyI') el('interpBtn').click();
        if (e.code === 'KeyO') el('roundBtn').click();
        if (e.code === 'KeyB') el('blueNoiseBtn').click();
        if (e.code === 'KeyN') el('bnItersBtn').click();
        if (e.code === 'KeyH') el('threshBtn').click();
        if (e.code === 'KeyL') el('lockNoiseBtn').click();
        if (e.code === 'Space') { e.preventDefault(); el('pauseBtn').click(); }
    });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

resizeCanvas();
initFramebuffers();
initNoiseFBOs();
setupUI();
syncUI();
multipleSplats(Math.floor(Math.random() * 20) + 5);
update();
