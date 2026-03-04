/**
 * WGSL shader sources for V3.
 * Scene: instanced MRT (color + motion + depth) with storage buffer of transforms.
 * Sky: fullscreen atmosphere pass (Rayleigh + Mie single-scatter).
 * Warp + display + blue noise: identical to V2.
 */

// ---------------------------------------------------------------------------
// Shared: atmosphere scattering (used by sky pass and scene lighting)
// ---------------------------------------------------------------------------

const atmosphereWGSL = /* wgsl */`
// Single-scatter atmospheric scattering (Rayleigh + Mie), Nishita-style.
// All lengths in megameters (Mm). Earth radius ≈ 6.371 Mm.
//
// Rayleigh scattering: wavelength-dependent (shorter λ scatters more).
//   → blue sky overhead; orange/red at sunset (long slant path exhausts blue).
// Mie scattering: nearly wavelength-independent, forms bright halo around sun.

const PLANET_R: f32 = 6.371;       // Earth surface radius (Mm)
const ATMOS_R:  f32 = 6.471;       // Top of atmosphere (Mm)
const CLOUD_R:  f32 = 6.373;       // Cloud layer sphere radius (~2 km altitude)
const H_R:      f32 = 0.008;       // Rayleigh scale height (Mm)
const H_M:      f32 = 0.0012;      // Mie scale height (Mm)
// β_R per wavelength in Mm⁻¹: blue (0.45µm) >> green >> red, drives color separation at sunset
// Standard sea-level values: 5.8e-6 m⁻¹ × 1e6 m/Mm = 5.8 Mm⁻¹
const BETA_R: vec3f = vec3f(5.8, 13.5, 33.1);
const BETA_M: f32   = 21.0;        // Mie β in Mm⁻¹ (wavelength-neutral)
const MIE_G:  f32   = 0.82;        // Mie asymmetry: closer to 1 = tighter forward peak
const SUN_INTENSITY: f32 = 40.0;
const VIEW_SAMPLES: i32 = 8;
const SUN_SAMPLES:  i32 = 4;

// Returns distance along dir to far sphere intersection, or -1 on miss.
fn raySphereIntersect(origin: vec3f, dir: vec3f, radius: f32) -> f32 {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) { return -1.0; }
    return -b + sqrt(disc);
}

// Returns near sphere intersection (first hit), or -1 on miss.
fn raySphereNear(origin: vec3f, dir: vec3f, radius: f32) -> f32 {
    let b = dot(origin, dir);
    let c = dot(origin, origin) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) { return -1.0; }
    let sq = sqrt(disc);
    let t0 = -b - sq;
    let t1 = -b + sq;
    if (t1 < 0.0) { return -1.0; }
    return select(t1, t0, t0 >= 0.0);
}

// Atmospheric scattering integral along viewDir from sea-level camera.
fn atmosphere(viewDir: vec3f, sunDir: vec3f) -> vec3f {
    let origin = vec3f(0.0, PLANET_R + 0.0002, 0.0);
    let dir = normalize(viewDir);

    var tMax = raySphereIntersect(origin, dir, ATMOS_R);
    if (tMax < 0.0) { return vec3f(0.0); }

    // Clip at planet surface (don't integrate underground)
    let bv  = dot(origin, dir);
    let cv  = dot(origin, origin) - PLANET_R * PLANET_R;
    let dv  = bv * bv - cv;
    if (dv > 0.0) {
        let tPlanet = -bv - sqrt(dv);
        if (tPlanet > 0.0) { tMax = min(tMax, tPlanet); }
    }

    let stepLen = tMax / f32(VIEW_SAMPLES);
    let mu  = dot(dir, sunDir);
    let mu2 = mu * mu;
    let g2  = MIE_G * MIE_G;

    // Rayleigh phase: 3/(16π) * (1 + cos²θ)
    let phaseR = 0.05968310365 * (1.0 + mu2);
    // Mie phase (Henyey-Greenstein)
    let denomM = max(1.0 + g2 - 2.0 * MIE_G * mu, 0.0001);
    let phaseM = 0.07957747154 * (1.0 - g2) / (denomM * sqrt(denomM));

    var optDepthR: f32 = 0.0;
    var optDepthM: f32 = 0.0;
    var sumR = vec3f(0.0);
    var sumM = vec3f(0.0);

    for (var i: i32 = 0; i < VIEW_SAMPLES; i++) {
        let t   = (f32(i) + 0.5) * stepLen;
        let pos = origin + dir * t;
        let h   = length(pos) - PLANET_R;

        let densR = exp(-h / H_R) * stepLen;
        let densM = exp(-h / H_M) * stepLen;
        optDepthR += densR;
        optDepthM += densM;

        // Sun ray: skip if sun below horizon from this point
        let tSun = raySphereIntersect(pos, sunDir, ATMOS_R);
        if (tSun < 0.0) { continue; }

        let sunStep = tSun / f32(SUN_SAMPLES);
        var sunOptR: f32 = 0.0;
        var sunOptM: f32 = 0.0;
        for (var j: i32 = 0; j < SUN_SAMPLES; j++) {
            let ts    = (f32(j) + 0.5) * sunStep;
            let sPos  = pos + sunDir * ts;
            let sH    = length(sPos) - PLANET_R;
            sunOptR += exp(-sH / H_R) * sunStep;
            sunOptM += exp(-sH / H_M) * sunStep;
        }

        // Beer-Lambert extinction along view + sun path
        let tau  = BETA_R * (optDepthR + sunOptR) + BETA_M * (optDepthM + sunOptM);
        let attn = exp(-tau);
        sumR += densR * attn;
        sumM += densM * attn;
    }

    return (sumR * BETA_R * phaseR + sumM * BETA_M * phaseM) * SUN_INTENSITY;
}

// ---------- Cloud helpers ----------

// Hash 2D → [0,1)
fn cloudHash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453123);
}

// Bilinear value noise
fn cloudNoise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = cloudHash(i);
    let b = cloudHash(i + vec2f(1.0, 0.0));
    let c = cloudHash(i + vec2f(0.0, 1.0));
    let d = cloudHash(i + vec2f(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// FBM cloud density in [0,1] — 5 octaves with light domain warp.
fn cloudFBM(p: vec2f) -> f32 {
    var v:    f32 = 0.0;
    var amp:  f32 = 0.5;
    var freq: f32 = 1.0;
    var q = p;
    for (var i: i32 = 0; i < 5; i++) {
        v += amp * cloudNoise(q * freq);
        let warpFreq = freq * 2.07;  // slightly irrational to break tiling
        // Domain warp: displace the sample point by lower-frequency noise
        q += (vec2f(cloudNoise(q * warpFreq + vec2f(1.7, 9.2)),
                    cloudNoise(q * warpFreq + vec2f(8.3, 2.8))) - 0.5) * 0.25;
        freq *= 2.07;
        amp  *= 0.45;
    }
    return v;
}

// ---------- Star helpers ----------

// Scatter sparse stars by hashing a coarse direction cell.
// Returns brightness [0,1] — mostly 0.
fn starField(dir: vec3f) -> f32 {
    // Quantize the unit sphere into ~8000 cells for star placement
    let cellScale = 90.0;
    let cell = floor(dir * cellScale);
    let h = fract(sin(dot(cell, vec3f(127.1, 311.7, 74.7))) * 43758.5453);
    // Only ~0.5% of cells get a star
    if (h > 0.995) {
        // Sub-cell position for a sharp dot
        let frac = fract(dir * cellScale);
        let d = length(frac - vec3f(0.5)) * cellScale * 0.5;
        return max(0.0, 1.0 - d) * (h - 0.995) * 1000.0;
    }
    return 0.0;
}
`;

// ---------------------------------------------------------------------------
// Sky: fullscreen quad at far plane, atmosphere background
// ---------------------------------------------------------------------------

export const skyWGSL = /* wgsl */`
struct SkyUniforms {
    invViewProj: mat4x4f,
    sunDir:      vec4f,   // xyz = normalized sun direction, w = unused
    time:        vec4f,   // x = elapsed seconds for cloud/star animation
}

@group(0) @binding(0) var<uniform> sky: SkyUniforms;

${atmosphereWGSL}

struct VsOut {
    @builtin(position) position: vec4f,
    @location(0) clipXY: vec2f,
}

@vertex fn vs(@location(0) pos: vec2f, @location(1) uv: vec2f) -> VsOut {
    var out: VsOut;
    // z=1.0, w=1.0 places the quad at the far plane
    out.position = vec4f(pos, 1.0, 1.0);
    out.clipXY = pos;
    return out;
}

struct FsOut {
    @location(0) color:  vec4f,
    @location(1) motion: vec4f,
}

@fragment fn fs(in: VsOut) -> FsOut {
    var out: FsOut;

    // Reconstruct world-space view direction from clip coords.
    // WebGPU NDC: depth 0=near, 1=far.
    let nearClip  = vec4f(in.clipXY, 0.0, 1.0);
    let farClip   = vec4f(in.clipXY, 1.0, 1.0);
    let nearWorld = sky.invViewProj * nearClip;
    let farWorld  = sky.invViewProj * farClip;
    let viewDir   = normalize(farWorld.xyz / farWorld.w - nearWorld.xyz / nearWorld.w);

    let sunDir    = normalize(sky.sunDir.xyz);
    let elapsedT  = sky.time.x;

    // Moon is opposite the sun.
    let moonDir = -sunDir;

    // How far below the horizon is the sun? Used to blend day/night lighting.
    // nightBlend: 0 = full day, 1 = full night.
    let nightBlend = smoothstep(0.05, -0.1, sunDir.y);
    // moonBlend: 1 when moon is above horizon, 0 when below.
    let moonBlend  = smoothstep(-0.05, 0.05, moonDir.y);

    // --- Atmosphere (daylight) ---
    var skyColor = atmosphere(viewDir, sunDir);

    // --- Moonlight atmospheric scattering ---
    // At night, the moon acts as a dim bluish light source.
    // Scale down by ~0.02x relative to sun intensity and add a cool blue tint.
    let moonAtmos = atmosphere(viewDir, moonDir)
                    * 0.02                            // ~50x dimmer than sun
                    * vec3f(0.55, 0.65, 1.0)          // cool blue-white moonlight
                    * nightBlend * moonBlend;
    skyColor += moonAtmos;

    // --- Sun disc ---
    // Only draw when sun is above horizon (sunDir.y > 0).
    let sunVisible = smoothstep(-0.02, 0.02, sunDir.y);
    let sunDot  = dot(viewDir, sunDir);
    // Chromatic sun: white-hot core, warm halo tinting toward orange
    let sunCore = smoothstep(0.9998, 1.0,   sunDot) * sunVisible;
    let sunHalo = smoothstep(0.9990, 0.9998, sunDot) * sunVisible;
    // Halo color tints with sunDir.y: warm orange near horizon, white overhead
    let sunHaloColor = mix(vec3f(1.0, 0.55, 0.15), vec3f(1.0, 0.95, 0.85),
                           saturate(sunDir.y * 3.0));
    skyColor += sunHaloColor * sunHalo * 1.5;
    skyColor += vec3f(2.5, 2.4, 2.0) * sunCore;

    // --- Moon disc (crescent) ---
    // Visible when moon is above the horizon and sun is below.
    // Crescent shape: full moon disc minus a shadow disc offset perpendicular to moonDir.
    // Since moonDir = -sunDir exactly, we can't derive a sun-side tangent from sunDir alone.
    // Instead, use the world-up axis to build a stable disc tangent for the crescent tilt.
    let moonDot = dot(viewDir, moonDir);
    // Angular size: slightly larger than real (~0.5°) to be clearly visible.
    let moonDiscEdge  = 0.99993;   // outer edge of moon disc
    let moonDiscCore  = 0.99997;   // bright interior of moon disc

    // Build a tangent in the moon's disc plane for the crescent shadow direction.
    // Use world-up unless moonDir is nearly vertical, then fall back to world-forward.
    let upRef     = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(moonDir.y) > 0.99);
    let moonUp    = normalize(upRef - moonDir * dot(moonDir, upRef));
    let moonRight = normalize(cross(moonDir, moonUp));
    // Shadow disc center: offset the shadow to one side to carve a crescent.
    let shadowCenter = normalize(moonDir + moonRight * 0.00010);
    let shadowDot    = dot(viewDir, shadowCenter);

    // Moon face: inside moon disc, outside shadow disc
    let inMoonDisc   = smoothstep(moonDiscEdge, moonDiscCore, moonDot);
    let inShadowDisc = smoothstep(moonDiscCore, moonDiscEdge, shadowDot);  // 1 inside shadow
    let moonFace     = inMoonDisc * (1.0 - inShadowDisc);

    // Moon color: cool white with a slight warm inner glow
    let moonColor = mix(vec3f(0.8, 0.85, 0.95), vec3f(1.0, 0.98, 0.92), inMoonDisc * 0.5);
    skyColor += moonColor * moonFace * nightBlend * moonBlend * 1.2;

    // --- Filmic tone mapping (ACES approximation) ---
    // Better than Reinhard: preserves warm hues and avoids washing out orange/pink.
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    skyColor = clamp((skyColor * (a * skyColor + b)) /
                     (skyColor * (c * skyColor + d) + e), vec3f(0.0), vec3f(1.0));

    // --- Stars (night only) ---
    // Fade stars in as sun goes below horizon. Stars are static (hash-based).
    if (nightBlend > 0.001 && viewDir.y > -0.1) {
        let starBrightness = starField(viewDir) * nightBlend;
        // Stars twinkle faintly via time-based modulation on their hash
        let twinkle = 0.85 + 0.15 * sin(elapsedT * 3.7 + dot(viewDir, vec3f(7.3, 13.1, 4.7)) * 50.0);
        skyColor = max(skyColor, vec3f(starBrightness * twinkle));
    }

    // --- Clouds ---
    // Cast the view ray to the cloud layer sphere (CLOUD_R from atmosphere constants).
    let cloudOrigin = vec3f(0.0, PLANET_R + 0.0002, 0.0);
    let tCloud = raySphereNear(cloudOrigin, viewDir, CLOUD_R);
    if (tCloud > 0.0 && viewDir.y > 0.0) {
        let cloudPos = cloudOrigin + viewDir * tCloud;

        // Project to a flat UV for noise (stereographic-ish: divide by altitude)
        // Scale controls cloud feature size: larger = bigger clouds
        let cloudScale = 80.0;
        // Animate: clouds drift visibly across the sky (~0.8 UV units/s at this scale)
        let driftSpeed = 0.16;
        let cloudUV = vec2f(cloudPos.x, cloudPos.z) * cloudScale + elapsedT * driftSpeed;

        let density = cloudFBM(cloudUV);

        // Threshold: values above ~0.58 become cloud (adjusts coverage)
        let cloudCoverage: f32 = 0.58;
        let cloudOpacity = smoothstep(cloudCoverage, cloudCoverage + 0.15, density);

        if (cloudOpacity > 0.001) {
            // --- Daytime cloud lighting ---
            let sunLift = saturate(sunDir.y);
            let cloudSunFactor = saturate(dot(vec3f(0.0, 1.0, 0.0), sunDir));

            // Cloud color: white on top, greyish underneath; warm tint at sunrise/set
            let sunsetTint = mix(vec3f(1.0, 1.0, 1.0),
                                 mix(vec3f(1.0, 0.6, 0.3), vec3f(1.0, 1.0, 1.0),
                                     saturate(sunDir.y * 5.0)),
                                 saturate(1.0 - sunDir.y * 3.0));
            let cloudLit   = sunsetTint * (0.7 + 0.3 * cloudSunFactor);
            let cloudShade = sunsetTint * vec3f(0.4, 0.42, 0.5) * (0.5 + 0.5 * sunLift);
            // Mix lit/shaded based on density (denser → more shaded interior)
            let dayCloudColor = mix(cloudLit, cloudShade, smoothstep(0.0, 0.5, density - cloudCoverage));

            // --- Nighttime cloud lighting (moonlit) ---
            // Cool blue-grey lit by moonlight; darker underside.
            let moonLift = saturate(moonDir.y) * moonBlend;
            let nightCloudLit   = vec3f(0.30, 0.33, 0.45) * (0.5 + 0.5 * moonLift);
            let nightCloudShade = vec3f(0.05, 0.06, 0.10);
            let nightCloudColor = mix(nightCloudLit, nightCloudShade,
                                      smoothstep(0.0, 0.5, density - cloudCoverage));

            // Blend day/night cloud colors based on time of day
            let cloudColor = mix(dayCloudColor, nightCloudColor, nightBlend);

            // Fade clouds out toward horizon (atmospheric haze)
            let horizonFade = smoothstep(0.0, 0.15, viewDir.y);
            skyColor = mix(skyColor, cloudColor, cloudOpacity * horizonFade);
        }
    }

    out.color = vec4f(skyColor, 1.0);
    out.motion = vec4f(0.0, 0.0, 0.0, 1.0);
    return out;
}
`;

// ---------------------------------------------------------------------------
// Shadow: depth-only vertex pass for directional shadow mapping
// ---------------------------------------------------------------------------

export const shadowWGSL = /* wgsl */`
struct ShadowUniforms {
    lightSpaceMatrix: mat4x4f,
}

struct InstanceData {
    model:     mat4x4f,
    prevModel: mat4x4f,
    color:     vec4f,
}

@group(0) @binding(0) var<uniform>       shadow:    ShadowUniforms;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;

@vertex fn vs(
    @location(0) position: vec3f,
    @location(1) normal:   vec3f,
    @builtin(instance_index) instanceIdx: u32,
) -> @builtin(position) vec4f {
    let wp = instances[instanceIdx].model * vec4f(position, 1.0);
    return shadow.lightSpaceMatrix * wp;
}
`;

// ---------------------------------------------------------------------------
// Scene: instanced vertex + fragment for MRT
// ---------------------------------------------------------------------------

export const sceneWGSL = /* wgsl */`
struct CameraUniforms {
    viewProj:         mat4x4f,
    prevViewProj:     mat4x4f,
    sunDir:           vec4f,   // xyz = normalized sun direction, w = unused
    lightSpaceMatrix: mat4x4f, // for shadow map lookup (offset 144 bytes)
    eyePos:           vec4f,   // xyz = camera world position (offset 208 bytes)
    eyeDir:           vec4f,   // xyz = camera forward unit vector (offset 224 bytes)
}

struct InstanceData {
    model:     mat4x4f,
    prevModel: mat4x4f,
    color:     vec4f,
}

// Point light for spheres/mushrooms — emitted light illuminates nearby surfaces.
struct PointLight {
    posAndRadius: vec4f,  // xyz = world position, w = influence radius
    color:        vec4f,  // rgb = light color, a = intensity
}

struct LightBuffer {
    count: u32,
    _pad1: u32, _pad2: u32, _pad3: u32,
    lights: array<PointLight, 32>,
}

@group(0) @binding(0) var<uniform>       cam:          CameraUniforms;
@group(0) @binding(1) var<storage, read> instances:    array<InstanceData>;
@group(0) @binding(2) var               shadowMap:     texture_depth_2d;
@group(0) @binding(3) var               shadowSampler: sampler_comparison;
@group(0) @binding(4) var<uniform>       sceneLights:  LightBuffer;

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
    @location(8) tangentWorld: vec3f,
    @location(9) bitangentWorld: vec3f,
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
    // Tangent frame for normal perturbation (local X,Y axes in world space)
    out.tangentWorld   = (inst.model * vec4f(1.0, 0.0, 0.0, 0.0)).xyz;
    out.bitangentWorld = (inst.model * vec4f(0.0, 1.0, 0.0, 0.0)).xyz;
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

// Normal perturbation for the stone floor crevices.
// Returns a (dx, dz) offset in world XZ to tilt the floor normal toward crevice walls,
// making mortar lines appear indented. Uses finite differences on Voronoi F1 distance.
// Only perturbs near crevices (small F2-F1 gap); stone face centers stay flat.
// Returns vec2f(dx, dz) — apply as: n' = normalize(n + strength*(dx*worldX + dz*worldZ))
fn floorNormalPerturbation(worldXZ: vec2f) -> vec2f {
    let scale = 0.4;
    let eps: f32 = 0.04;  // finite-difference step in world units

    // Sample voronoi at center (gives F1, F2) and two offset points for gradient
    let vc = voronoi(worldXZ * scale);
    let f1x = voronoi((worldXZ + vec2f(eps, 0.0)) * scale).x;
    let f1z = voronoi((worldXZ + vec2f(0.0, eps)) * scale).x;

    // Gradient of F1 points away from nearest cell center (toward crevice)
    let grad = vec2f((f1x - vc.x) / eps, (f1z - vc.x) / eps);

    // Crevice proximity: F2-F1 small → near border. Remap to [0,1] weight.
    let creviceWeight = 1.0 - smoothstep(0.0, 0.12, vc.y - vc.x);

    // Tilt normal along gradient direction, scaled by crevice proximity
    return grad * creviceWeight;
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

// Compute pip indentation depth at a point, returning (dx, dy) normal offset in face-local space.
// Pips are concave bowls: the normal tilts away from pip center inside the bowl.
fn pipNormalOffset(faceUV: vec2f, center: vec2f, radius: f32) -> vec2f {
    let d = length(faceUV - center);
    if (d >= radius) { return vec2f(0.0); }
    // Smooth bowl: offset proportional to position within pip, fading at edges
    let t = 1.0 - d / radius;  // 1 at center, 0 at rim
    let dir = select((faceUV - center) / d, vec2f(0.0), d < 0.001);
    // Parabolic bowl profile: offset strongest at ~half radius
    return dir * t * (1.0 - t) * 4.0;
}

// Accumulate pip normal offsets for a half-face. Mirrors drawPips layout exactly.
fn pipsPerturbation(faceUV: vec2f, count: u32) -> vec2f {
    let pipR: f32 = 0.055;
    let cx = 0.5; let lx = 0.25; let rx = 0.75;
    let ty = 0.8; let my = 0.5;  let by = 0.2;
    var offset = vec2f(0.0);
    if (count == 1u || count == 3u || count == 5u) {
        offset += pipNormalOffset(faceUV, vec2f(cx, my), pipR);
    }
    if (count >= 2u) {
        offset += pipNormalOffset(faceUV, vec2f(rx, ty), pipR);
        offset += pipNormalOffset(faceUV, vec2f(lx, by), pipR);
    }
    if (count >= 4u) {
        offset += pipNormalOffset(faceUV, vec2f(lx, ty), pipR);
        offset += pipNormalOffset(faceUV, vec2f(rx, by), pipR);
    }
    if (count == 6u) {
        offset += pipNormalOffset(faceUV, vec2f(lx, my), pipR);
        offset += pipNormalOffset(faceUV, vec2f(rx, my), pipR);
    }
    return offset;
}

// Normal perturbation for domino face in local tangent space (dx, dy).
// Returns a 2D offset to be applied as: n' = normalize(n + depth*(dx*T + dy*B))
fn dominoNormalPerturbation(localPos: vec3f, localNormal: vec3f, instanceId: u32) -> vec2f {
    let isZFace = abs(localNormal.z) > 0.99;
    if (!isZFace) { return vec2f(0.0); }

    var offset = vec2f(0.0);

    // Center groove: V-groove at localPos.y = 0
    let grooveWidth: f32 = 0.03;
    let grooveDist = abs(localPos.y);
    if (grooveDist < grooveWidth) {
        let tilt = sign(localPos.y) * (1.0 - grooveDist / grooveWidth);
        offset.y += tilt;
    }

    // Pip indentation
    let pipTop = (instanceId % 6u) + 1u;
    let pipBot = ((instanceId * 7u + 3u) % 6u) + 1u;
    let halfU = localPos.x + 0.5;
    if (localPos.y > 0.0) {
        offset += pipsPerturbation(vec2f(halfU, localPos.y * 2.0), pipTop);
    } else {
        offset += pipsPerturbation(vec2f(halfU, (localPos.y + 0.5) * 2.0), pipBot);
    }

    return offset;
}

// Soccer ball (truncated icosahedron): 12 black pentagons + 20 white hexagons.
// Pentagon centers = 12 icosahedron vertices. Each pentagon has 5-fold symmetry
// with corners pointing toward neighboring vertices. Seams run along pentagon
// boundaries and hex-hex Voronoi edges between adjacent icosahedron vertices.
fn soccerBallTexture(localPos: vec3f) -> vec3f {
    let p = normalize(localPos);

    // 12 icosahedron vertices (normalized), serving as pentagon centers.
    // Coordinates: permutations of (0, +-1, +-phi) / sqrt(1 + phi^2).
    let phi = 1.618034;
    var ico = array<vec3f, 12>(
        normalize(vec3f( 0.0,  1.0,  phi)), normalize(vec3f( 0.0, -1.0,  phi)),
        normalize(vec3f( 0.0,  1.0, -phi)), normalize(vec3f( 0.0, -1.0, -phi)),
        normalize(vec3f( 1.0,  phi,  0.0)), normalize(vec3f(-1.0,  phi,  0.0)),
        normalize(vec3f( 1.0, -phi,  0.0)), normalize(vec3f(-1.0, -phi,  0.0)),
        normalize(vec3f( phi,  0.0,  1.0)), normalize(vec3f(-phi,  0.0,  1.0)),
        normalize(vec3f( phi,  0.0, -1.0)), normalize(vec3f(-phi,  0.0, -1.0)),
    );

    // Find closest (d1) and second-closest (d2) icosahedron vertex by dot product.
    var d1: f32 = -2.0;
    var d2: f32 = -2.0;
    var closest: vec3f = ico[0];
    var second:  vec3f = ico[0];
    for (var i: u32 = 0u; i < 12u; i++) {
        let d = dot(p, ico[i]);
        if (d > d1) {
            d2 = d1; second = closest;
            d1 = d;  closest = ico[i];
        } else if (d > d2) {
            d2 = d; second = ico[i];
        }
    }

    // Pentagon geometry on the unit sphere (derived from truncated icosahedron):
    // Edge angle between adjacent icosahedron vertices = arccos(1/sqrt(5)) ~ 63.43 deg.
    // Truncation at 1/3 gives pentagon circumradius angle ~ 21.15 deg.
    // Pentagon apothem angle = circumradius * cos(36 deg) ~ 17.11 deg.
    let PENT_CIRCUM: f32 = 0.36905;  // 21.145 deg in radians
    let PENT_APOTHEM: f32 = 0.29862; // 17.107 deg in radians
    let SEAM_W: f32 = 0.015;         // seam half-width in dot-product space

    // Compute pentagonal boundary: modulate threshold with 5-fold symmetry.
    // Project p and second-closest vertex into tangent plane of closest vertex.
    let tang_p   = p - d1 * closest;
    let tang_ref = second - dot(second, closest) * closest;
    let tp_len = length(tang_p);
    let tr_len = length(tang_ref);

    // Azimuthal angle relative to second-closest vertex direction.
    // Pentagon corners point toward neighbors; edges are at 36 deg offsets.
    var pentBoundaryDot: f32 = cos(PENT_CIRCUM);  // fallback: circular
    if (tp_len > 1e-6 && tr_len > 1e-6) {
        let tp_n = tang_p / tp_len;
        let tr_n = tang_ref / tr_len;
        let cos_az = dot(tp_n, tr_n);
        let sin_az = dot(cross(tr_n, tp_n), closest);
        let azimuth = atan2(sin_az, cos_az);

        // Map azimuth into one sector of the pentagon: [-36, 36] degrees.
        // Pentagon has 72 deg per sector; corners at 0, edges at +-36 deg.
        let sector = 1.2566371;  // 72 deg in radians
        let half   = 0.6283185;  // 36 deg in radians
        // fract-based modulo to avoid negative-modulo issues
        let az_mod = (azimuth / sector - floor(azimuth / sector)) * sector - half;

        // Pentagon SDF: boundary distance = apothem / cos(az_mod)
        let boundaryAngle = PENT_APOTHEM / cos(az_mod);
        pentBoundaryDot = cos(clamp(boundaryAngle, 0.0, PENT_CIRCUM * 1.2));
    }

    // Pentagon fill: inside the pentagonal boundary
    let pentFill = smoothstep(pentBoundaryDot - SEAM_W * 0.5,
                              pentBoundaryDot + SEAM_W * 0.5, d1);

    // Pentagon boundary seam: ring where d1 ~ pentBoundaryDot
    let pentSeam = 1.0 - smoothstep(0.0, SEAM_W, abs(d1 - pentBoundaryDot));

    // Hex-hex seam: perpendicular distance to the bisector plane between
    // the two closest icosahedron vertices. dot(p, normalize(c1-c2)) measures
    // signed distance; thin lines even at triple junctions.
    let bisectorRaw = closest - second;
    let bisectorLen = length(bisectorRaw);
    let bisector = select(bisectorRaw / bisectorLen, vec3f(0.0), bisectorLen < 1e-6);
    let bisectorDist = abs(dot(p, bisector));
    let hexEdge = 1.0 - smoothstep(0.0, SEAM_W, bisectorDist);
    // Zero inside pentagon region
    let outsidePent = 1.0 - smoothstep(pentBoundaryDot - SEAM_W,
                                       pentBoundaryDot + SEAM_W, d1);
    let hexSeam = hexEdge * outsidePent;

    let seam = max(pentSeam, hexSeam);

    // Final color: black pentagons (0.06), dark seams (0.10), white hexagons (0.95)
    var color = mix(0.95, 0.06, pentFill);
    color = mix(color, 0.10, seam);
    return vec3f(color);
}

// ---------- Maze textures ----------

// Raw world-space UV for stone wall (no fract — Voronoi handles periodicity
// internally, so using raw world coords makes the pattern tile seamlessly
// across adjacent wall panels that share the same world position values).
fn stoneWallRawUV(worldPos: vec3f, localNormal: vec3f) -> vec2f {
    let absN = abs(localNormal);
    if (absN.y > 0.5) {
        return worldPos.xz;
    } else if (absN.x > absN.z) {
        return vec2f(worldPos.z, worldPos.y);
    } else {
        return vec2f(worldPos.x, worldPos.y);
    }
}

// Shared stone-wall Voronoi sample.
// Returns vec4f(.rgb = final color, .a = mortar weight 0=stone 1=mortar).
fn stoneWallSample(worldPos: vec3f, localNormal: vec3f) -> vec4f {
    let rawUV = stoneWallRawUV(worldPos, localNormal);

    // Higher frequency than floor, smaller stones
    let blockUV = rawUV * vec2f(0.9, 1.2);
    let row = floor(blockUV.y);
    // Running bond: every other row offsets by half a block width
    let shift = select(0.0, 0.5, fract(row * 0.5) > 0.25);
    let shiftedUV = vec2f(blockUV.x + shift, blockUV.y) * 2.4;

    let v = voronoi(shiftedUV);

    // Mortar: thin band where F2-F1 is small (Voronoi cell borders)
    let border = v.y - v.x;
    let mortarW = 1.0 - smoothstep(0.04, 0.10, border);

    // Per-stone color variation: lower contrast, cooler grey-blue tones
    let h1 = hash1(v.z);
    let h2 = hash1(v.z + 73.1);
    let h3 = hash1(v.z + 137.3);

    // Luminance 0.38 – 0.50 (narrower range = lower contrast)
    let lum  = h1 * 0.12 + 0.38;
    let warm = (h2 - 0.5) * 0.03;
    let cool = (h3 - 0.5) * 0.05;

    // Sub-cell roughness
    let rA = hash1(floor(rawUV.x * 5.0) * 7.3  + floor(rawUV.y * 5.0) * 13.7) * 0.02 - 0.01;
    let rB = hash1(floor(rawUV.x * 7.0) * 11.3 + floor(rawUV.y * 7.0) * 17.9) * 0.015 - 0.007;
    let rough = rA + rB;

    // Cool grey-blue stone (distinct from warm floor voronoi)
    let stoneCol = vec3f(
        clamp(lum - 0.02 + warm + rough,         0.0, 1.0),
        clamp(lum + cool + rough * 0.8,           0.0, 1.0),
        clamp(lum + 0.04 - warm * 0.3 + rough,    0.0, 1.0)
    );
    let mortarCol = vec3f(0.22, 0.20, 0.18);

    return vec4f(mix(stoneCol, mortarCol, mortarW), mortarW);
}

// Seamless Voronoi stone wall texture with per-stone color variation and mortar.
fn stoneWallTexture(worldPos: vec3f, localNormal: vec3f) -> vec3f {
    return stoneWallSample(worldPos, localNormal).rgb;
}

// Normal perturbation for stone walls: mortar crevices indented, stone bumps.
// Returns (dU, dV) to add to the surface normal via TBN frame.
fn stoneWallNormalPerturb(worldPos: vec3f, localNormal: vec3f) -> vec2f {
    let rawUV = stoneWallRawUV(worldPos, localNormal);
    let blockUV = rawUV * vec2f(0.55, 0.75);
    let row = floor(blockUV.y);
    let shift = select(0.0, 0.5, fract(row * 0.5) > 0.25);
    let baseUV = vec2f(blockUV.x + shift, blockUV.y) * 1.8;

    // Finite-difference gradient of mortar weight for crevice indent
    let eps = 0.06;
    let vxp = voronoi(baseUV + vec2f(eps, 0.0));
    let vxn = voronoi(baseUV - vec2f(eps, 0.0));
    let vyp = voronoi(baseUV + vec2f(0.0, eps));
    let vyn = voronoi(baseUV - vec2f(0.0, eps));
    let mxp = 1.0 - smoothstep(0.04, 0.10, vxp.y - vxp.x);
    let mxn = 1.0 - smoothstep(0.04, 0.10, vxn.y - vxn.x);
    let myp = 1.0 - smoothstep(0.04, 0.10, vyp.y - vyp.x);
    let myn = 1.0 - smoothstep(0.04, 0.10, vyn.y - vyn.x);

    // Gradient toward mortar = indent (negate to push normal inward)
    let gU = (mxp - mxn) * 0.5;
    let gV = (myp - myn) * 0.5;

    // Stone surface roughness bumps (only on stone, suppress on mortar)
    let mortarW = stoneWallSample(worldPos, localNormal).a;
    let bScale = (1.0 - mortarW) * 0.012;
    let bU = hash1(floor(rawUV.x * 4.0 + 0.5) * 9.1 + floor(rawUV.y * 4.0) * 17.3) * bScale;
    let bV = hash1(floor(rawUV.x * 4.0) * 9.1 + floor(rawUV.y * 4.0 + 0.5) * 17.3) * bScale;

    let mortarDepth = 0.6;
    return vec2f(-gU * mortarDepth + bU, -gV * mortarDepth + bV);
}

// Procedural ivy vines cascading from wall tops.
// Returns ivy color scaled by vine strength (vec3(0) = no ivy).
fn ivyOverlay(worldPos: vec3f, localNormal: vec3f) -> vec3f {
    let wallTop = 3.0;
    let distFromTop = wallTop - worldPos.y;
    if (abs(localNormal.y) > 0.5 || distFromTop < 0.0) {
        return vec3f(0.0);
    }

    let absN = abs(localNormal);
    var u: f32;
    if (absN.x > absN.z) { u = worldPos.z; } else { u = worldPos.x; }

    let n1 = sin(u * 2.5 + hash1(floor(u * 0.8)) * 6.28) * 0.5 + 0.5;
    let n2 = sin(u * 5.3 + 1.7 + hash1(floor(u * 1.3 + 7.0)) * 6.28) * 0.5 + 0.5;
    let n3 = sin(u * 8.1 + 3.2) * 0.5 + 0.5;

    let vine1 = smoothstep(n1 * 1.8 + 0.5, n1 * 1.8 + 0.2, distFromTop) * n1;
    let vine2 = smoothstep(n2 * 1.2 + 0.25, n2 * 1.2 + 0.0, distFromTop) * n2 * 0.7;
    let vine3 = smoothstep(n3 * 0.7 + 0.1, n3 * 0.7 - 0.05, distFromTop) * n3 * 0.4;
    let vineStrength = clamp(vine1 + vine2 + vine3, 0.0, 1.0);

    let leafNoise = hash1(floor(u * 4.0) + floor(worldPos.y * 3.0) * 17.0);
    let leafBright = smoothstep(0.6, 0.8, leafNoise) * vineStrength * 0.3;

    let darkGreen = vec3f(0.12, 0.28, 0.08);
    let brightGreen = vec3f(0.18, 0.42, 0.12);
    return mix(darkGreen, brightGreen, leafBright + vineStrength * 0.3) * vineStrength;
}

// Procedural treasure chest texture: wood planks + gold metal bands.
fn chestTexture(localPos: vec3f, localNormal: vec3f) -> vec3f {
    let absN = abs(localNormal);
    var uv: vec2f;
    if (absN.y > 0.5) {
        uv = localPos.xz;
    } else if (absN.x > absN.z) {
        uv = vec2f(localPos.z, localPos.y);
    } else {
        uv = vec2f(localPos.x, localPos.y);
    }

    let plankY = fract(uv.y * 4.0 + 0.5);
    let plankEdge = smoothstep(0.0, 0.03, plankY) * smoothstep(0.0, 0.03, 1.0 - plankY);
    let woodGrain = hash1(floor(uv.y * 4.0 + 0.5) * 7.0) * 0.1;
    let woodColor = vec3f(0.55 + woodGrain, 0.35 + woodGrain * 0.7, 0.12);

    let bandY = abs(uv.y);
    let isBand = smoothstep(0.3, 0.32, bandY) * (1.0 - smoothstep(0.38, 0.40, bandY));
    let metalColor = vec3f(0.7, 0.6, 0.15);

    return mix(woodColor * plankEdge, metalColor, isBand);
}

// ---------- Terrain biome ----------

// Biome color from world position + normal. Heights are negative (terrain below y=0).
// Bands (y is negative): near 0 = snow/rock, deeper = grass, deepest = dark scree.
fn terrainBiomeColor(worldPos: vec3f, worldNormal: vec3f) -> vec3f {
    let y = worldPos.y;
    // Slope: 0 = flat, 1 = vertical cliff (derived from upward component of normal)
    let slope = 1.0 - abs(worldNormal.y);

    let snowColor  = vec3f(0.90, 0.92, 0.95);
    let rockColor  = vec3f(0.45, 0.38, 0.30);
    let grassColor = vec3f(0.22, 0.34, 0.14);
    let deepColor  = vec3f(0.15, 0.13, 0.12);

    // Heights now go upward: -10 (base) to +440 (peaks)
    var biome: vec3f;
    if (y > 300.0) {
        biome = snowColor;
    } else if (y > 100.0) {
        let t = smoothstep(100.0, 300.0, y);
        biome = mix(rockColor, snowColor, t);
    } else if (y > 0.0) {
        let t = smoothstep(0.0, 100.0, y);
        biome = mix(grassColor, rockColor, t);
    } else {
        let t = smoothstep(-20.0, 0.0, y);
        biome = mix(deepColor, grassColor, t);
    }

    // Steep slopes are always rocky regardless of height
    biome = mix(biome, rockColor, smoothstep(0.4, 0.75, slope));

    // Subtle world-space noise variation via sin harmonics
    let noiseXZ = sin(worldPos.x * 0.07 + 1.3) * sin(worldPos.z * 0.09 + 2.7) * 0.04;
    return clamp(biome + noiseXZ, vec3f(0.0), vec3f(1.0));
}

// ---------- Fragment ----------

struct FsOut {
    @location(0) color:  vec4f,
    @location(1) motion: vec4f,
}

@fragment fn fs(in: VsOut) -> FsOut {
    var out: FsOut;
    var n = normalize(in.normal);

    // Determine material color from procedural textures
    var albedo = in.color;
    let isFloor = (in.instanceId == 0u);
    let isSphere = (in.color.b > 0.9 && in.color.r < 0.3);
    // Maze wall: color ~(0.45, 0.44, 0.42)
    let isMazeWall = (abs(in.color.r - 0.45) < 0.02 && abs(in.color.g - 0.44) < 0.02 && abs(in.color.b - 0.42) < 0.02);
    // Treasure chest: color ~(0.85, 0.65, 0.20)
    let isChest = (abs(in.color.r - 0.85) < 0.02 && abs(in.color.g - 0.65) < 0.02 && abs(in.color.b - 0.20) < 0.02);
    // Domino: color ~(0.92, 0.90, 0.85) — explicit match instead of catch-all
    let isDomino = (abs(in.color.r - 0.92) < 0.02 && abs(in.color.g - 0.90) < 0.02 && abs(in.color.b - 0.85) < 0.02);
    // Terrain mesh: sentinel color ~(0.12, 0.48, 0.08)
    let isTerrain = (abs(in.color.r - 0.12) < 0.02 && abs(in.color.g - 0.48) < 0.02 && abs(in.color.b - 0.08) < 0.02);

    if (isFloor) {
        albedo = floorTexture(in.worldPos.xz);
        // Perturb floor normal to make stone crevices appear indented.
        // Gradient in XZ plane tilts the up-normal toward crevice walls.
        let floorPerturb = floorNormalPerturbation(in.worldPos.xz);
        let indentStrength: f32 = 0.3;
        n = normalize(n + indentStrength * (floorPerturb.x * vec3f(1.0, 0.0, 0.0)
                                          + floorPerturb.y * vec3f(0.0, 0.0, 1.0)));
    } else if (isSphere) {
        albedo = soccerBallTexture(in.localPos);
    } else if (isMazeWall) {
        let stone = stoneWallTexture(in.worldPos, in.localNormal);
        let ivy = ivyOverlay(in.worldPos, in.localNormal);
        let ivyMask = clamp(length(ivy) * 2.5, 0.0, 1.0);
        albedo = mix(stone, ivy / max(ivyMask, 0.001), ivyMask);

        // Perturb normal: mortar crevices indent + stone surface bumps
        let wallPerturb = stoneWallNormalPerturb(in.worldPos, in.localNormal);
        let T = normalize(in.tangentWorld);
        let B = normalize(in.bitangentWorld);
        n = normalize(n + wallPerturb.x * T + wallPerturb.y * B);
    } else if (isChest) {
        albedo = chestTexture(in.localPos, in.localNormal);
    } else if (isDomino) {
        albedo = dominoTexture(in.localPos, in.localNormal, in.instanceId);

        // Perturb normal for pip indentation and center groove
        let perturbXY = dominoNormalPerturbation(in.localPos, in.localNormal, in.instanceId);
        let indentDepth: f32 = 0.4;
        let T = normalize(in.tangentWorld);
        let B = normalize(in.bitangentWorld);
        n = normalize(n + indentDepth * (perturbXY.x * T + perturbXY.y * B));
    } else if (isTerrain) {
        albedo = terrainBiomeColor(in.worldPos, n);
    }

    // Blinn-Phong: sun (warm/time-varying) + cool sky fill + night ambient
    let viewDir = normalize(-in.currClip.xyz / in.currClip.w);
    let sunDir  = normalize(cam.sunDir.xyz);

    // sunElevation in [-1,1]: positive = day, negative = night
    let sunElevation = sunDir.y;

    // Sun color: white at noon, orange at sunset, not shown at night
    let noonColor   = vec3f(1.0,  0.95, 0.88);
    let sunsetColor = vec3f(1.0,  0.55, 0.18);
    // t=0 at horizon, t=1 overhead
    let sunColorT  = saturate(sunElevation * 3.0);
    let sunColor   = mix(sunsetColor, noonColor, sunColorT);
    // Sun dimmer near/below horizon
    let sunStrength = saturate(sunElevation * 4.0 + 0.3);

    // Cool sky fill (simulates indirect sky light from above)
    let skyFillDir   = normalize(vec3f(-0.3, 0.8, -0.5));
    let skyFillColor = mix(
        vec3f(0.05, 0.06, 0.12),  // night: very dark blue
        vec3f(0.22, 0.28, 0.42),  // day: soft blue sky
        saturate(sunElevation * 2.0 + 0.5));

    // Ambient: dark at night, warm dim fill during day
    let ambient = mix(
        vec3f(0.02, 0.02, 0.04),   // night: nearly black
        vec3f(0.10, 0.09, 0.12),   // day: warm dark
        saturate(sunElevation * 3.0 + 0.5));

    // Shadow: PCF 3×3 samples around light-space projected position.
    // Bias scales with slope (NdotL) to suppress shadow acne on angled surfaces.
    // Shadow factor is in [0.3, 1.0] so ambient still shows through.
    let lightClip  = cam.lightSpaceMatrix * vec4f(in.worldPos, 1.0);
    let lightNDC   = lightClip.xyz / lightClip.w;
    // Map NDC [-1,1] → UV [0,1]; WebGPU depth is already [0,1] for lightNDC.z.
    let shadowUV   = vec2f(lightNDC.x * 0.5 + 0.5, -lightNDC.y * 0.5 + 0.5);
    let NdotSun    = max(dot(n, sunDir), 0.0);
    let bias       = mix(0.005, 0.002, NdotSun);
    let shadowDepth = lightNDC.z - bias;

    // 3×3 PCF: average 9 comparison samples over a texel neighbourhood.
    let texelSize = 1.0 / vec2f(textureDimensions(shadowMap));
    var shadowSum: f32 = 0.0;
    for (var sy: i32 = -1; sy <= 1; sy++) {
        for (var sx: i32 = -1; sx <= 1; sx++) {
            let offset = vec2f(f32(sx), f32(sy)) * texelSize;
            shadowSum += textureSampleCompare(shadowMap, shadowSampler,
                                              shadowUV + offset, shadowDepth);
        }
    }
    let shadowFactor = mix(0.1, 1.0, shadowSum / 9.0);

    // No shadow when sun is below horizon or fragment is outside the light frustum.
    let inLightFrustum = (shadowUV.x >= 0.0) && (shadowUV.x <= 1.0)
                      && (shadowUV.y >= 0.0) && (shadowUV.y <= 1.0)
                      && (lightNDC.z >= 0.0) && (lightNDC.z <= 1.0);
    let shadow = select(1.0, shadowFactor, inLightFrustum && sunElevation > 0.0);

    let shininess = select(32.0, 24.0, isFloor);
    var diffuse  = vec3f(0.0);
    var specular = vec3f(0.0);

    // Sun contribution (only when above horizon), attenuated by shadow
    let d0 = max(dot(n, sunDir), 0.0);
    let h0 = normalize(sunDir + viewDir);
    let s0 = pow(max(dot(n, h0), 0.0), shininess);
    diffuse  += sunColor * d0 * sunStrength * shadow;
    specular += sunColor * s0 * 0.3 * sunStrength * shadow;

    // Sky fill contribution (unshadowed — indirect light)
    let d1 = max(dot(n, skyFillDir), 0.0);
    let h1 = normalize(skyFillDir + viewDir);
    let s1 = pow(max(dot(n, h1), 0.0), shininess);
    diffuse  += skyFillColor * d1;
    specular += skyFillColor * s1 * 0.15;

    var color = albedo * (ambient + diffuse) + specular;

    // --- Point lights from glowing objects ---
    for (var li: u32 = 0u; li < sceneLights.count; li++) {
        let light = sceneLights.lights[li];
        let toLight = light.posAndRadius.xyz - in.worldPos;
        let lightDist = length(toLight);
        if (lightDist < light.posAndRadius.w && lightDist > 0.01) {
            let lightDir = toLight / lightDist;
            let falloff = 1.0 - lightDist / light.posAndRadius.w;
            let atten = falloff * falloff * light.color.a / (1.0 + lightDist * lightDist * 0.1);
            let NdotL = max(dot(n, lightDir), 0.0);
            color += albedo * light.color.rgb * NdotL * atten;
            let halfVec = normalize(lightDir + viewDir);
            let spec = pow(max(dot(n, halfVec), 0.0), shininess);
            color += light.color.rgb * spec * 0.3 * atten;
        }
    }

    // --- Half-emissive glow (always visible, retains shading) ---
    if (isSphere) {
        color += vec3f(0.3, 0.5, 1.0) * 0.75;
    }
    let isMushCap = (abs(albedo.r - 0.8) < 0.08 && abs(albedo.g - 0.15) < 0.06);
    if (isMushCap) {
        let hue = fract(sin(dot(floor(in.worldPos.xz * 0.5), vec2f(127.1, 311.7))) * 43758.5);
        let glowColor = mix(vec3f(0.2, 0.8, 0.4), vec3f(0.4, 0.2, 0.9), hue);
        color += glowColor * 0.6;
    }

    // --- Nighttime lighting effects ---
    let nightFactor = smoothstep(0.05, -0.1, sunElevation);

    if (nightFactor > 0.01) {
        // Flashlight: forward-facing spotlight from camera position.
        let toFrag   = normalize(in.worldPos - cam.eyePos.xyz);
        let spotDot  = dot(toFrag, cam.eyeDir.xyz);
        let spotAtten = smoothstep(0.85, 0.95, spotDot);
        let flashDist = length(in.worldPos - cam.eyePos.xyz);
        let distAtten = 1.0 / (1.0 + 0.05 * flashDist * flashDist);
        let flashlight = spotAtten * distAtten * nightFactor * 2.0;
        color += albedo * flashlight * vec3f(1.0, 0.95, 0.85);
    }

    out.color = vec4f(color, 1.0);

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
