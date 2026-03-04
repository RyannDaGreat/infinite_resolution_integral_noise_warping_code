/**
 * WGSL shader sources for WebGPU V2.
 * Warp shaders: scene (MRT), display, build_deform, backward_map, brownian, normalize.
 * Blue noise shaders: blur, histCount, prefixSum, scatterSort, scatterMatch.
 */

// ---------------------------------------------------------------------------
// Scene: vertex + fragment for MRT (color + motion + depth)
// ---------------------------------------------------------------------------

export const sceneWGSL = /* wgsl */`
struct Uniforms {
    model:         mat4x4f,
    viewProj:      mat4x4f,
    prevModel:     mat4x4f,
    prevViewProj:  mat4x4f,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsIn {
    @location(0) position: vec3f,
    @location(1) normal:   vec3f,
    @location(2) color:    vec3f,
}

struct VsOut {
    @builtin(position) position: vec4f,
    @location(0) color:    vec3f,
    @location(1) currClip: vec4f,
    @location(2) prevClip: vec4f,
    @location(3) normal:   vec3f,
}

@vertex fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    let worldPos     = u.model * vec4f(in.position, 1.0);
    let prevWorldPos = u.prevModel * vec4f(in.position, 1.0);
    out.currClip = u.viewProj * worldPos;
    out.prevClip = u.prevViewProj * prevWorldPos;
    out.color    = in.color;
    out.normal   = (u.model * vec4f(in.normal, 0.0)).xyz;
    out.position = out.currClip;
    return out;
}

struct FsOut {
    @location(0) color:  vec4f,
    @location(1) motion: vec4f,
}

@fragment fn fs(in: VsOut) -> FsOut {
    var out: FsOut;
    let lightDir = normalize(vec3f(1.0, 1.0, 1.0));
    let diff = max(dot(normalize(in.normal), lightDir), 0.3);
    out.color = vec4f(in.color * diff, 1.0);

    let currNDC = in.currClip.xy / in.currClip.w;
    let prevNDC = in.prevClip.xy / in.prevClip.w;
    out.motion = vec4f((currNDC - prevNDC) * 0.5, 0.0, 1.0);
    return out;
}
`;

// ---------------------------------------------------------------------------
// Display: fullscreen quad with 5 display modes
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

// Noise read directly from storage buffer (no texture copy needed)
@group(0) @binding(0) var<storage, read> noiseData: array<f32>;
@group(0) @binding(1) var colorTex:  texture_2d<f32>;
@group(0) @binding(2) var motionTex: texture_2d<f32>;

struct DisplayUniforms {
    mode:  u32,
    W:     u32,
    H:     u32,
    flags: u32,  // bit 0: greyscale, bit 1: Gaussian→uniform (normal CDF)
    thresholdOn:    u32,  // 0=off, 1=on
    thresholdValue: f32,  // [0,1] cutoff — pixels with display value > threshold are white
}
@group(0) @binding(3) var<uniform> disp: DisplayUniforms;

fn readNoise(col: u32, row: u32) -> vec4f {
    let idx = (row * disp.W + col) * 4u;
    return vec4f(noiseData[idx], noiseData[idx+1u], noiseData[idx+2u], noiseData[idx+3u]);
}

// Abramowitz & Stegun erf approximation 7.1.26 — error < 1.5e-7
fn erf_approx(x: f32) -> f32 {
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let result = 1.0 - poly * exp(-ax * ax);
    return select(-result, result, x >= 0.0);
}

// Normal CDF: maps Gaussian → Uniform [0,1]
fn normal_cdf(x: f32) -> f32 {
    return 0.5 * (1.0 + erf_approx(x * 0.7071067811865476));
}

// Apply display flags (greyscale, Gaussian→uniform)
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

// Compute the displayed pixel value for noise (applies flags + scaling)
fn noiseToDisplay(col: u32, row: u32) -> vec4f {
    let n = readNoise(col, row);
    let v = applyFlags(n);
    if ((disp.flags & 2u) != 0u) {
        return vec4f(v.rgb, 1.0);  // uniform mode: already [0,1]
    }
    return vec4f(v.rgb / 5.0 + 0.5, 1.0);  // Gaussian: scale to ~[0,1] for display
}

// Apply threshold: pixels above threshold → white, below → black
fn applyThreshold(color: vec4f) -> vec4f {
    if (disp.thresholdOn == 0u) { return color; }
    let val = color.r;  // threshold on displayed channel 0
    return select(vec4f(0.0, 0.0, 0.0, 1.0), vec4f(1.0, 1.0, 1.0, 1.0), val > disp.thresholdValue);
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
    let uv = in.texcoord;
    let W = disp.W;
    let H = disp.H;
    let col = min(u32(uv.x * f32(W)), W - 1u);
    let row = min(u32(uv.y * f32(H)), H - 1u);

    if (disp.mode == 0u) {
        return applyThreshold(noiseToDisplay(col, row));
    } else if (disp.mode == 1u) {
        return textureLoad(colorTex, vec2u(col, row), 0);
    } else if (disp.mode == 2u) {
        let mv = textureLoad(motionTex, vec2u(col, row), 0).rg;
        return vec4f(mv * 5.0 + 0.5, 0.5, 1.0);
    } else if (disp.mode == 3u) {
        if (uv.x < 0.5) {
            let sc = min(u32(uv.x * 2.0 * f32(W)), W - 1u);
            return textureLoad(colorTex, vec2u(sc, row), 0);
        } else {
            let sc = min(u32((uv.x - 0.5) * 2.0 * f32(W)), W - 1u);
            return applyThreshold(noiseToDisplay(sc, row));
        }
    } else {
        return applyThreshold(noiseToDisplay(col, row));
    }
}
`;

// ---------------------------------------------------------------------------
// Build Deformation: motion texture → deformation buffer
// ---------------------------------------------------------------------------

export const buildDeformWGSL = /* wgsl */`
struct Uniforms {
    H:         u32,
    W:         u32,
    frameSeed: u32,
    roundMode: u32,  // 0=none, 1=round all motion to nearest pixel, 2=round only if |motion| > 1px
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var motionTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> deformation: array<f32>;

// Round a pixel-space motion value based on roundMode:
// 0: no rounding — fractional pixel motion preserved
// 1: round to nearest integer — snaps all motion to whole pixels
// 2: round only if |val| > 1 — sub-pixel motion preserved, large motion quantized
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
    // Convert UV-space motion to pixel-space, apply rounding, then compute source position
    let dxPx = applyRounding(motion.r * f32(u.W));  // horizontal pixels
    let dyPx = applyRounding(motion.g * f32(u.H));  // vertical pixels

    // Backward deformation: dest pixel → source position (row, col)
    // Identity cell center = (row + 0.5, col + 0.5)
    // mvY > 0 means object moved up → source was below → larger row
    // mvX > 0 means object moved right → source was left → smaller col
    deformation[idx * 2u]      = f32(row) + 0.5 + dyPx;
    deformation[idx * 2u + 1u] = f32(col) + 0.5 - dxPx;
}
`;

// ---------------------------------------------------------------------------
// Backward Map: deformation → bilinear tickets via atomicAdd
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
// Brownian Bridge: per-source pixel, sequential ticket processing
// Uses float atomics (CAS loop) for scatter to dest pixels.
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

// Generate 4 normal samples via 2 Box-Muller pairs (uses both sin+cos)
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

    // Store per-source totalRequest so normalize can compute pixelArea without atomics
    totalRequest[srcIdx] = totalReq;

    let srcBase = srcIdx * 4u;
    let srcNoise = vec4f(noise[srcBase], noise[srcBase+1u], noise[srcBase+2u], noise[srcBase+3u]);

    // Fast path: 1 ticket = direct copy, no Brownian bridge / randn needed
    if (nTickets == 1) {
        let destIdx = u32(masterField[ticketBase]);
        let destBase = destIdx * 4u;
        atomicAddF32(&bufferAtomic[destBase],      srcNoise.x);
        atomicAddF32(&bufferAtomic[destBase + 1u], srcNoise.y);
        atomicAddF32(&bufferAtomic[destBase + 2u], srcNoise.z);
        atomicAddF32(&bufferAtomic[destBase + 3u], srcNoise.w);
        return;
    }

    // Vectorized Brownian bridge — operate on all 4 channels as vec4f
    var pastRange: f32 = 0.0;
    var pv = vec4f(0.0);  // pastValue for all 4 channels
    var randState: u32 = pcg(u.frameSeed * 104729u + srcIdx);

    for (var k: i32 = 0; k < nTickets; k = k + 1) {
        let w = areaField[ticketBase + u32(k)];
        let destIdx = u32(masterField[ticketBase + u32(k)]);
        let normalizedW = w / totalReq;
        let nextRange = pastRange + normalizedW;

        var nv: vec4f;  // nextValue

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

        let cv = nv - pv;  // currValue
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
// Normalize: scale by 1/sqrt(area) or fill fresh noise
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

// PCG hash for PRNG
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

// Compute pixelArea for dest pixel by recomputing bilinear weights from deformation
// and reading totalRequest per source pixel. Avoids atomic scatter in brownian pass.
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

    // For each bilinear neighbor: area += weight / totalRequest[neighbor]
    // (only if neighbor is in bounds and has nonzero totalRequest)
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
        // Fresh randn for unassigned pixels
        for (var c: u32 = 0u; c < C; c = c + 1u) {
            let seed = u.frameSeed * 7919u + 31337u + idx * C + c;
            noise[base + c] = randn(seed);
        }
    }
}
`;

// ===========================================================================
// Blue Noise: alternating projections (spatial Gaussian high-pass + histogram match)
// Equivalent to rp.convert_to_blue_noise but GPU-accelerated.
// high_pass(x) = x - gaussian_blur(x, sigma) where sigma = N/(2*pi*D0)
// ===========================================================================

// ---------------------------------------------------------------------------
// Gaussian blur (separable, 9-tap radius-4, all 4 channels as vec4f)
// direction=0: horizontal, direction=1: vertical + subtract original (= high-pass)
// ---------------------------------------------------------------------------

export const blueNoiseBlurWGSL = /* wgsl */`
struct Uniforms {
    H:           u32,
    W:           u32,
    sigma:       f32,
    direction:   u32,  // 0=horizontal, 1=vertical+subtract+rescale
    invSigmaHp:  f32,  // 1/σ_hp: rescales high-pass to unit variance
    greyscale:   u32,  // 1=scalar path (channel 0 only), 0=vec4 path
}

const RADIUS: i32 = 4;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read>       blurInput: array<f32>;  // source for blur samples
@group(0) @binding(2) var<storage, read_write> noise:     array<f32>;  // dir=0: write temp; dir=1: read original + write result

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

    // Greyscale scalar path: only process channel 0 (~75% less memory bandwidth)
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

    // Vec4 path: process all 4 channels
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

    // Vertical pass: high-pass = (original - blurred) * invSigmaHp
    // For Gaussian inputs, histogram matching is equivalent to dividing by σ_hp.
    // Safe to read+write noise at same pixel: each thread touches only its own index.
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

// Sorting shaders removed — replaced by analytical σ_hp rescaling in blur pass.
// For Gaussian inputs, histogram matching is equivalent to dividing by σ_hp.
