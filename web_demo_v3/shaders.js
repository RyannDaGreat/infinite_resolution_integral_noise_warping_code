/**
 * WGSL shader sources for V3.
 * Scene: instanced MRT (color + motion + depth) with storage buffer of transforms.
 * Warp + display + blue noise: identical to V2.
 */

// ---------------------------------------------------------------------------
// Scene: instanced vertex + fragment for MRT
// ---------------------------------------------------------------------------

export const sceneWGSL = /* wgsl */`
struct CameraUniforms {
    viewProj:     mat4x4f,
    prevViewProj: mat4x4f,
}

struct InstanceData {
    model:     mat4x4f,
    prevModel: mat4x4f,
    color:     vec4f,
}

@group(0) @binding(0) var<uniform> cam: CameraUniforms;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;

struct VsIn {
    @location(0) position: vec3f,
    @location(1) normal:   vec3f,
    @builtin(instance_index) instanceIdx: u32,
}

struct VsOut {
    @builtin(position) position: vec4f,
    @location(0) color:      vec3f,
    @location(1) currClip:   vec4f,
    @location(2) prevClip:   vec4f,
    @location(3) normal:     vec3f,
    @location(4) localPos:   vec3f,
    @location(5) worldPos:   vec3f,
    @location(6) @interpolate(flat) instanceId: u32,
    @location(7) localNormal: vec3f,
}

@vertex fn vs(in: VsIn) -> VsOut {
    let inst = instances[in.instanceIdx];
    var out: VsOut;
    let wp = inst.model * vec4f(in.position, 1.0);
    let prevWp = inst.prevModel * vec4f(in.position, 1.0);
    out.currClip   = cam.viewProj * wp;
    out.prevClip   = cam.prevViewProj * prevWp;
    out.color      = inst.color.rgb;
    out.normal     = (inst.model * vec4f(in.normal, 0.0)).xyz;
    out.localPos   = in.position;
    out.worldPos   = wp.xyz;
    out.instanceId  = in.instanceIdx;
    out.localNormal = in.normal;
    out.position    = out.currClip;
    return out;
}

// ---------- Procedural helpers ----------

fn hash2(p: vec2f) -> vec2f {
    let q = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

fn hash1(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

// Returns (F1 distance, F2 distance, cell ID)
fn voronoi(uv: vec2f) -> vec3f {
    let i = floor(uv);
    let f = fract(uv);
    var f1: f32 = 8.0;
    var f2: f32 = 8.0;
    var cellId: f32 = 0.0;
    for (var y: i32 = -1; y <= 1; y++) {
        for (var x: i32 = -1; x <= 1; x++) {
            let neighbor = vec2f(f32(x), f32(y));
            let point = hash2(i + neighbor);
            let d = length(neighbor + point - f);
            if (d < f1) {
                f2 = f1;
                f1 = d;
                cellId = dot(i + neighbor, vec2f(7.0, 157.0));
            } else if (d < f2) {
                f2 = d;
            }
        }
    }
    return vec3f(f1, f2, cellId);
}

// Procedural stone/rock floor texture with mortar lines at cell borders
fn floorTexture(worldXZ: vec2f) -> vec3f {
    let scale = 0.4;
    let v = voronoi(worldXZ * scale);
    // Mortar lines where F2 - F1 is small (cell borders)
    let edge = smoothstep(0.03, 0.08, v.y - v.x);
    let cellShade = hash1(v.z) * 0.12 + 0.38;
    let mortar = vec3f(0.25);
    return mix(mortar, vec3f(cellShade), edge);
}

// Procedural domino texture: pips on Z-faces (two halves like real dominoes).
// Bevel is handled by geometry (smooth normals on chamfered edges).
fn dominoTexture(localPos: vec3f, localNormal: vec3f, instanceId: u32) -> vec3f {
    let base = vec3f(0.92, 0.90, 0.85);

    // Pips only on Z-faces (the large flat faces). Use LOCAL normal so pips
    // stay visible regardless of domino orientation in world space.
    let isZFace = abs(localNormal.z) > 0.99;
    var pattern: f32 = 1.0;
    if (isZFace) {
        // Two halves per face like real dominoes: top half + bottom half
        let pipTop = (instanceId % 6u) + 1u;
        let pipBot = ((instanceId * 7u + 3u) % 6u) + 1u;

        // Center divider line
        let divider = 1.0 - smoothstep(0.0, 0.02, abs(localPos.y));
        pattern -= divider * 0.5;

        let halfU = localPos.x + 0.5;
        if (localPos.y > 0.0) {
            // Top half: map y [0, 0.5] → [0, 1]
            pattern = min(pattern, drawPips(vec2f(halfU, localPos.y * 2.0), pipTop));
        } else {
            // Bottom half: map y [-0.5, 0] → [0, 1]
            pattern = min(pattern, drawPips(vec2f(halfU, (localPos.y + 0.5) * 2.0), pipBot));
        }
    }

    return base * pattern;
}

// Draw domino pips (dots) for a half-face. Standard domino pip patterns.
fn drawPips(faceUV: vec2f, count: u32) -> f32 {
    let pipR: f32 = 0.055;
    var result: f32 = 1.0;
    // Pip center positions for counts 1-6 (in 0..1 UV space, centered on face)
    let cx = 0.5;
    let lx = 0.25;
    let rx = 0.75;
    let ty = 0.8;
    let my = 0.5;
    let by = 0.2;
    // Center pip
    if (count == 1u || count == 3u || count == 5u) {
        result = min(result, pipDot(faceUV, vec2f(cx, my), pipR));
    }
    // Top-right + bottom-left (2, 3, 4, 5, 6)
    if (count >= 2u) {
        result = min(result, pipDot(faceUV, vec2f(rx, ty), pipR));
        result = min(result, pipDot(faceUV, vec2f(lx, by), pipR));
    }
    // Top-left + bottom-right (4, 5, 6)
    if (count >= 4u) {
        result = min(result, pipDot(faceUV, vec2f(lx, ty), pipR));
        result = min(result, pipDot(faceUV, vec2f(rx, by), pipR));
    }
    // Middle-left + middle-right (6)
    if (count == 6u) {
        result = min(result, pipDot(faceUV, vec2f(lx, my), pipR));
        result = min(result, pipDot(faceUV, vec2f(rx, my), pipR));
    }
    return result;
}

fn pipDot(uv: vec2f, center: vec2f, radius: f32) -> f32 {
    let d = length(uv - center);
    return smoothstep(radius - 0.01, radius + 0.01, d);
}

// Soccer ball pattern: approximate with icosahedron-like regions
fn soccerBallTexture(localPos: vec3f) -> vec3f {
    let n = normalize(localPos);
    // Use spherical voronoi approximation: 12 pentagon centers (icosahedron vertices)
    let phi = 1.618034;
    var pentagonCenters = array<vec3f, 12>(
        normalize(vec3f( 0,  1,  phi)), normalize(vec3f( 0, -1,  phi)),
        normalize(vec3f( 0,  1, -phi)), normalize(vec3f( 0, -1, -phi)),
        normalize(vec3f( 1,  phi,  0)), normalize(vec3f(-1,  phi,  0)),
        normalize(vec3f( 1, -phi,  0)), normalize(vec3f(-1, -phi,  0)),
        normalize(vec3f( phi,  0,  1)), normalize(vec3f(-phi,  0,  1)),
        normalize(vec3f( phi,  0, -1)), normalize(vec3f(-phi,  0, -1)),
    );
    var minDot: f32 = -1.0;
    for (var i: u32 = 0u; i < 12u; i++) {
        let d = dot(n, pentagonCenters[i]);
        minDot = max(minDot, d);
    }
    // Pentagons are black, rest is white with seam lines
    let isPentagon = step(0.89, minDot);
    let seamDist = abs(minDot - 0.80);
    let seam = 1.0 - smoothstep(0.0, 0.025, seamDist);
    let baseColor = mix(0.95, 0.08, isPentagon);
    return vec3f(baseColor - seam * 0.3);
}

// ---------- Fragment ----------

struct FsOut {
    @location(0) color:  vec4f,
    @location(1) motion: vec4f,
}

@fragment fn fs(in: VsOut) -> FsOut {
    var out: FsOut;
    let n = normalize(in.normal);

    // Determine material color from procedural textures
    var albedo = in.color;
    let isFloor = (in.instanceId == 0u);
    let isBox = (in.color.g > 0.8 && in.color.b < 0.9); // domino detection via color
    let isSphere = (in.color.b > 0.9 && in.color.r < 0.3); // blue sphere detection

    if (isFloor) {
        albedo = floorTexture(in.worldPos.xz);
    } else if (isSphere) {
        albedo = soccerBallTexture(in.localPos);
    } else if (!isFloor && !isSphere) {
        albedo = dominoTexture(in.localPos, in.localNormal, in.instanceId);
    }

    // Blinn-Phong with two lights
    let viewDir = normalize(-in.currClip.xyz / in.currClip.w);
    let ambient = vec3f(0.15);
    let light0 = normalize(vec3f(1.0, 3.0, 2.0));
    let light1 = normalize(vec3f(-2.0, 1.0, -1.0));
    let light0col = vec3f(1.0, 0.95, 0.9);
    let light1col = vec3f(0.3, 0.35, 0.5);
    let shininess = select(32.0, 24.0, isFloor);

    var diffuse = vec3f(0.0);
    var specular = vec3f(0.0);

    let d0 = max(dot(n, light0), 0.0);
    let h0 = normalize(light0 + viewDir);
    let s0 = pow(max(dot(n, h0), 0.0), shininess);
    diffuse += light0col * d0;
    specular += light0col * s0 * 0.3;

    let d1 = max(dot(n, light1), 0.0);
    let h1 = normalize(light1 + viewDir);
    let s1 = pow(max(dot(n, h1), 0.0), shininess);
    diffuse += light1col * d1;
    specular += light1col * s1 * 0.15;

    out.color = vec4f(albedo * (ambient + diffuse) + specular, 1.0);

    let currNDC = in.currClip.xy / in.currClip.w;
    let prevNDC = in.prevClip.xy / in.prevClip.w;
    out.motion = vec4f((currNDC - prevNDC) * 0.5, 0.0, 1.0);
    return out;
}
`;

// ---------------------------------------------------------------------------
// Display: fullscreen quad with 5 display modes (identical to V2)
// ---------------------------------------------------------------------------

export const displayWGSL = /* wgsl */`
struct VsOut {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
}

@vertex fn vs(@location(0) pos: vec2f, @location(1) uv: vec2f) -> VsOut {
    var out: VsOut;
    out.position = vec4f(pos, 0.0, 1.0);
    out.texcoord = uv;
    return out;
}

@group(0) @binding(0) var<storage, read> noiseData: array<f32>;
@group(0) @binding(1) var colorTex:  texture_2d<f32>;
@group(0) @binding(2) var motionTex: texture_2d<f32>;

struct DisplayUniforms {
    mode:  u32,
    W:     u32,
    H:     u32,
    flags: u32,
    thresholdOn:    u32,
    thresholdValue: f32,
    noiseOpacity:   f32,
}
@group(0) @binding(3) var<uniform> disp: DisplayUniforms;

fn readNoise(col: u32, row: u32) -> vec4f {
    let idx = (row * disp.W + col) * 4u;
    return vec4f(noiseData[idx], noiseData[idx+1u], noiseData[idx+2u], noiseData[idx+3u]);
}

fn erf_approx(x: f32) -> f32 {
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * exp(-ax * ax);
    return select(-result, result, x >= 0.0);
}

fn normal_cdf(x: f32) -> f32 {
    return 0.5 * (1.0 + erf_approx(x * 0.7071067811865476));
}

fn applyFlags(n: vec4f) -> vec4f {
    var v = n;
    if ((disp.flags & 2u) != 0u) {
        v = vec4f(normal_cdf(v.x), normal_cdf(v.y), normal_cdf(v.z), v.w);
    }
    if ((disp.flags & 1u) != 0u) {
        v = vec4f(v.x, v.x, v.x, v.w);
    }
    return v;
}

fn noiseToDisplay(col: u32, row: u32) -> vec4f {
    let n = readNoise(col, row);
    let v = applyFlags(n);
    if ((disp.flags & 2u) != 0u) {
        return vec4f(v.rgb, 1.0);
    }
    return vec4f(v.rgb / 5.0 + 0.5, 1.0);
}

fn applyThreshold(color: vec4f) -> vec4f {
    if (disp.thresholdOn == 0u) { return color; }
    let val = color.r;
    return select(vec4f(0.0, 0.0, 0.0, 1.0), vec4f(1.0, 1.0, 1.0, 1.0), val > disp.thresholdValue);
}

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.299, 0.587, 0.114));
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let uv = in.texcoord;
    let W = disp.W;
    let H = disp.H;
    let col = min(u32(uv.x * f32(W)), W - 1u);
    let row = min(u32(uv.y * f32(H)), H - 1u);

    // Mode 0: Noise
    if (disp.mode == 0u) {
        return applyThreshold(noiseToDisplay(col, row));
    }
    // Mode 1: Scene
    if (disp.mode == 1u) {
        return textureLoad(colorTex, vec2u(col, row), 0);
    }
    // Mode 2: Scene + Noise (opacity controlled by uniform)
    if (disp.mode == 2u) {
        let scene = textureLoad(colorTex, vec2u(col, row), 0).rgb;
        let n = readNoise(col, row);
        var noiseContrib: vec3f;
        if ((disp.flags & 2u) != 0u) {
            // Uniform: subtract 0.5 to center, then scale by opacity
            noiseContrib = vec3f(normal_cdf(n.x) - 0.5, normal_cdf(n.y) - 0.5, normal_cdf(n.z) - 0.5) * disp.noiseOpacity;
        } else {
            // Gaussian: scale by opacity
            noiseContrib = n.rgb * disp.noiseOpacity;
        }
        return vec4f(clamp(scene + noiseContrib, vec3f(0.0), vec3f(1.0)), 1.0);
    }
    // Mode 3: Dither (threshold scene luminance with noise)
    if (disp.mode == 3u) {
        let scene = textureLoad(colorTex, vec2u(col, row), 0).rgb;
        let lum = luminance(scene);
        let n = readNoise(col, row);
        // Use noise as threshold: uniform CDF gives [0,1]
        let threshold = normal_cdf(n.x);
        return select(vec4f(0.0, 0.0, 0.0, 1.0), vec4f(1.0, 1.0, 1.0, 1.0), lum > threshold);
    }
    // Mode 4: Motion vectors
    if (disp.mode == 4u) {
        let mv = textureLoad(motionTex, vec2u(col, row), 0).rg;
        return vec4f(mv * 5.0 + 0.5, 0.5, 1.0);
    }
    // Mode 5: Raw noise
    return applyThreshold(noiseToDisplay(col, row));
}
`;

// ---------------------------------------------------------------------------
// Build Deformation (identical to V2)
// ---------------------------------------------------------------------------

export const buildDeformWGSL = /* wgsl */`
struct Uniforms {
    H:         u32,
    W:         u32,
    frameSeed: u32,
    roundMode: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var motionTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> deformation: array<f32>;

fn applyRounding(val: f32) -> f32 {
    if (u.roundMode == 1u) {
        return round(val);
    } else if (u.roundMode == 2u) {
        if (abs(val) > 1.0) { return round(val); }
    }
    return val;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= u.H * u.W) { return; }

    let row = idx / u.W;
    let col = idx % u.W;

    let motion = textureLoad(motionTex, vec2u(col, row), 0);
    let dxPx = applyRounding(motion.r * f32(u.W));
    let dyPx = applyRounding(motion.g * f32(u.H));

    deformation[idx * 2u]      = f32(row) + 0.5 + dyPx;
    deformation[idx * 2u + 1u] = f32(col) + 0.5 - dxPx;
}
`;

// ---------------------------------------------------------------------------
// Backward Map (identical to V2)
// ---------------------------------------------------------------------------

export const backwardMapWGSL = /* wgsl */`
const MAX_TICKETS: u32 = 24u;

struct Uniforms {
    H: u32,
    W: u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read>       deformation: array<f32>;
@group(0) @binding(2) var<storage, read_write> ticketCount: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> masterField: array<i32>;
@group(0) @binding(4) var<storage, read_write> areaField:   array<f32>;

fn addTicket(row: i32, col: i32, destIdx: u32, weight: f32) {
    if (weight <= 0.0) { return; }
    if (row < 0 || row >= i32(u.H) || col < 0 || col >= i32(u.W)) { return; }
    let srcIdx = u32(row) * u.W + u32(col);
    let t = atomicAdd(&ticketCount[srcIdx], 1);
    if (t >= i32(MAX_TICKETS)) {
        atomicSub(&ticketCount[srcIdx], 1);
        return;
    }
    let slot = srcIdx * MAX_TICKETS + u32(t);
    masterField[slot] = i32(destIdx);
    areaField[slot]   = weight;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= u.H * u.W) { return; }

    let wpRow = deformation[idx * 2u]      - 0.5;
    let wpCol = deformation[idx * 2u + 1u] - 0.5;
    let lRow = i32(floor(wpRow));
    let lCol = i32(floor(wpCol));
    let fRow = wpRow - f32(lRow);
    let fCol = wpCol - f32(lCol);
    let uRow = lRow + 1;
    let uCol = lCol + 1;

    addTicket(lRow, lCol, idx, (1.0 - fRow) * (1.0 - fCol));
    addTicket(uRow, uCol, idx, fRow * fCol);
    addTicket(lRow, uCol, idx, (1.0 - fRow) * fCol);
    addTicket(uRow, lCol, idx, fRow * (1.0 - fCol));
}
`;

// ---------------------------------------------------------------------------
// Brownian Bridge (identical to V2)
// ---------------------------------------------------------------------------

export const brownianWGSL = /* wgsl */`
const MAX_TICKETS: u32 = 24u;
const EPS: f32 = 1e-6;
const TWO_PI: f32 = 6.283185307;

struct Uniforms {
    H:         u32,
    W:         u32,
    frameSeed: u32,
    _pad:      u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read>       ticketCount:    array<i32>;
@group(0) @binding(2) var<storage, read>       masterField:    array<i32>;
@group(0) @binding(3) var<storage, read>       areaField:      array<f32>;
@group(0) @binding(4) var<storage, read>       noise:          array<f32>;
@group(0) @binding(5) var<storage, read_write> bufferAtomic:   array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> totalRequest:   array<f32>;

fn pcg(v: u32) -> u32 {
    var state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn randn4(state: ptr<function, u32>) -> vec4f {
    *state = pcg(*state);
    let h1a = pcg(*state);
    *state = pcg(*state + 1u);
    let h2a = pcg(*state);
    let u1a = max(f32(h1a) / 4294967296.0, 1e-10);
    let u2a = f32(h2a) / 4294967296.0;
    let ra = sqrt(-2.0 * log(u1a));
    let th_a = TWO_PI * u2a;

    *state = pcg(*state + 1u);
    let h1b = pcg(*state);
    *state = pcg(*state + 1u);
    let h2b = pcg(*state);
    let u1b = max(f32(h1b) / 4294967296.0, 1e-10);
    let u2b = f32(h2b) / 4294967296.0;
    let rb = sqrt(-2.0 * log(u1b));
    let th_b = TWO_PI * u2b;

    return vec4f(ra * cos(th_a), ra * sin(th_a), rb * cos(th_b), rb * sin(th_b));
}

fn atomicAddF32(addr: ptr<storage, atomic<u32>, read_write>, val: f32) {
    var old = atomicLoad(addr);
    loop {
        let newVal = bitcast<u32>(bitcast<f32>(old) + val);
        let result = atomicCompareExchangeWeak(addr, old, newVal);
        if (result.exchanged) { break; }
        old = result.old_value;
    }
}

override WG_SIZE: u32 = 256;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let srcIdx = gid.x;
    if (srcIdx >= u.H * u.W) { return; }

    let nTickets = ticketCount[srcIdx];
    if (nTickets <= 0) { return; }

    let ticketBase = srcIdx * MAX_TICKETS;
    var totalReq: f32 = 0.0;
    for (var k: i32 = 0; k < nTickets; k = k + 1) {
        totalReq += areaField[ticketBase + u32(k)];
    }
    if (totalReq <= 0.0) { return; }

    totalRequest[srcIdx] = totalReq;

    let srcBase = srcIdx * 4u;
    let srcNoise = vec4f(noise[srcBase], noise[srcBase+1u], noise[srcBase+2u], noise[srcBase+3u]);

    if (nTickets == 1) {
        let destIdx = u32(masterField[ticketBase]);
        let destBase = destIdx * 4u;
        atomicAddF32(&bufferAtomic[destBase],      srcNoise.x);
        atomicAddF32(&bufferAtomic[destBase + 1u], srcNoise.y);
        atomicAddF32(&bufferAtomic[destBase + 2u], srcNoise.z);
        atomicAddF32(&bufferAtomic[destBase + 3u], srcNoise.w);
        return;
    }

    var pastRange: f32 = 0.0;
    var pv = vec4f(0.0);
    var randState: u32 = pcg(u.frameSeed * 104729u + srcIdx);

    for (var k: i32 = 0; k < nTickets; k = k + 1) {
        let w = areaField[ticketBase + u32(k)];
        let destIdx = u32(masterField[ticketBase + u32(k)]);
        let normalizedW = w / totalReq;
        let nextRange = pastRange + normalizedW;

        var nv: vec4f;

        if (nextRange >= 1.0 || (1.0 - pastRange) <= EPS) {
            nv = srcNoise;
        } else {
            let denom = 1.0 - pastRange;
            let a = (1.0 - nextRange) / denom;
            let b = (nextRange - pastRange) / denom;
            let mu4 = a * pv + b * srcNoise;
            let variance = (nextRange - pastRange) * (1.0 - nextRange) / denom;
            let stddev = sqrt(max(0.0, variance));

            let r4 = randn4(&randState);
            nv = r4 * stddev + mu4;
        }

        let cv = nv - pv;
        let destBase = destIdx * 4u;
        atomicAddF32(&bufferAtomic[destBase],      cv.x);
        atomicAddF32(&bufferAtomic[destBase + 1u], cv.y);
        atomicAddF32(&bufferAtomic[destBase + 2u], cv.z);
        atomicAddF32(&bufferAtomic[destBase + 3u], cv.w);

        pv = nv;
        pastRange = nextRange;
    }
}
`;

// ---------------------------------------------------------------------------
// Normalize (identical to V2)
// ---------------------------------------------------------------------------

export const normalizeWGSL = /* wgsl */`
const C: u32 = 4u;

struct Uniforms {
    H:         u32,
    W:         u32,
    frameSeed: u32,
    _pad:      u32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read>       buffer:       array<f32>;
@group(0) @binding(2) var<storage, read>       deformation:  array<f32>;
@group(0) @binding(3) var<storage, read>       totalRequest: array<f32>;
@group(0) @binding(4) var<storage, read_write> noise:        array<f32>;

fn pcg(v: u32) -> u32 {
    var state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn randn(seed: u32) -> f32 {
    let h1 = pcg(seed);
    let h2 = pcg(seed ^ 0xDEADBEEFu);
    let u1 = max(f32(h1) / 4294967296.0, 1e-10);
    let u2 = f32(h2) / 4294967296.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
}

fn computePixelArea(idx: u32) -> f32 {
    let wpRow = deformation[idx * 2u] - 0.5;
    let wpCol = deformation[idx * 2u + 1u] - 0.5;
    let lRow = i32(floor(wpRow));
    let lCol = i32(floor(wpCol));
    let fRow = wpRow - f32(lRow);
    let fCol = wpCol - f32(lCol);

    var area: f32 = 0.0;
    let w00 = (1.0 - fRow) * (1.0 - fCol);
    let w11 = fRow * fCol;
    let w01 = (1.0 - fRow) * fCol;
    let w10 = fRow * (1.0 - fCol);
    let uRow = lRow + 1;
    let uCol = lCol + 1;

    if (lRow >= 0 && lRow < i32(u.H) && lCol >= 0 && lCol < i32(u.W) && w00 > 0.0) {
        let tr = totalRequest[u32(lRow) * u.W + u32(lCol)];
        if (tr > 0.0) { area += w00 / tr; }
    }
    if (uRow >= 0 && uRow < i32(u.H) && uCol >= 0 && uCol < i32(u.W) && w11 > 0.0) {
        let tr = totalRequest[u32(uRow) * u.W + u32(uCol)];
        if (tr > 0.0) { area += w11 / tr; }
    }
    if (lRow >= 0 && lRow < i32(u.H) && uCol >= 0 && uCol < i32(u.W) && w01 > 0.0) {
        let tr = totalRequest[u32(lRow) * u.W + u32(uCol)];
        if (tr > 0.0) { area += w01 / tr; }
    }
    if (uRow >= 0 && uRow < i32(u.H) && lCol >= 0 && lCol < i32(u.W) && w10 > 0.0) {
        let tr = totalRequest[u32(uRow) * u.W + u32(lCol)];
        if (tr > 0.0) { area += w10 / tr; }
    }
    return area;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= u.H * u.W) { return; }

    let area = computePixelArea(idx);
    let base = idx * C;

    if (area > 0.0) {
        let scale = 1.0 / sqrt(area);
        for (var c: u32 = 0u; c < C; c = c + 1u) {
            noise[base + c] = scale * buffer[base + c];
        }
    } else {
        for (var c: u32 = 0u; c < C; c = c + 1u) {
            let seed = u.frameSeed * 7919u + 31337u + idx * C + c;
            noise[base + c] = randn(seed);
        }
    }
}
`;

// ---------------------------------------------------------------------------
// Blue Noise Blur (identical to V2)
// ---------------------------------------------------------------------------

export const blueNoiseBlurWGSL = /* wgsl */`
struct Uniforms {
    H:           u32,
    W:           u32,
    sigma:       f32,
    direction:   u32,
    invSigmaHp:  f32,
    greyscale:   u32,
}

const RADIUS: i32 = 4;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read>       blurInput: array<f32>;
@group(0) @binding(2) var<storage, read_write> noise:     array<f32>;

fn readVec4Input(idx: u32) -> vec4f {
    let b = idx * 4u;
    return vec4f(blurInput[b], blurInput[b+1u], blurInput[b+2u], blurInput[b+3u]);
}

fn readVec4Noise(idx: u32) -> vec4f {
    let b = idx * 4u;
    return vec4f(noise[b], noise[b+1u], noise[b+2u], noise[b+3u]);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= u.H * u.W) { return; }

    let row = idx / u.W;
    let col = idx % u.W;
    let inv2s2 = 1.0 / (2.0 * u.sigma * u.sigma);

    if (u.greyscale == 1u) {
        var sum1: f32 = 0.0;
        var wSum1: f32 = 0.0;
        for (var d: i32 = -RADIUS; d <= RADIUS; d++) {
            var srcIdx: u32;
            if (u.direction == 0u) {
                let c = u32((i32(col) + d + i32(u.W)) % i32(u.W));
                srcIdx = row * u.W + c;
            } else {
                let r = u32((i32(row) + d + i32(u.H)) % i32(u.H));
                srcIdx = r * u.W + col;
            }
            let w = exp(-f32(d * d) * inv2s2);
            sum1 += w * blurInput[srcIdx * 4u];
            wSum1 += w;
        }
        var r1 = sum1 / wSum1;
        if (u.direction == 1u) {
            r1 = (noise[idx * 4u] - r1) * u.invSigmaHp;
        }
        noise[idx * 4u] = r1;
        return;
    }

    var sum = vec4f(0.0);
    var wSum: f32 = 0.0;

    for (var d: i32 = -RADIUS; d <= RADIUS; d++) {
        var srcIdx: u32;
        if (u.direction == 0u) {
            let c = u32((i32(col) + d + i32(u.W)) % i32(u.W));
            srcIdx = row * u.W + c;
        } else {
            let r = u32((i32(row) + d + i32(u.H)) % i32(u.H));
            srcIdx = r * u.W + col;
        }
        let w = exp(-f32(d * d) * inv2s2);
        sum += w * readVec4Input(srcIdx);
        wSum += w;
    }

    var result = sum / wSum;

    if (u.direction == 1u) {
        result = (readVec4Noise(idx) - result) * u.invSigmaHp;
    }

    let b = idx * 4u;
    noise[b]      = result.x;
    noise[b + 1u] = result.y;
    noise[b + 2u] = result.z;
    noise[b + 3u] = result.w;
}
`;
