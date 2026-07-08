/**
 * WebGPU renderer: instanced MRT scene render, warp compute pipeline, blue noise, display.
 * Zero CPU-GPU copies per frame for the warp — only the instance buffer is uploaded.
 */

import { boxVertices, beveledBoxVertices, sphereVertices, quadVertices, terrainMeshVertices } from './geometry.js';
import {
    sceneWGSL, skyWGSL, shadowWGSL, displayWGSL,
    buildDeformWGSL, backwardMapWGSL, brownianWGSL, normalizeWGSL,
    blueNoiseBlurWGSL,
    starSplatWGSL, starScanRowsWGSL, starScanCdfWGSL, starUpdateWGSL, starRenderWGSL,
    mergeSelectWGSL,
} from './shaders.js';
import { MAX_INSTANCES, FLOATS_PER_INSTANCE, TERRAIN_INSTANCE_IDX } from './scene.js';

const NUM_TIMESTAMPS = 12;

// Star warp mode (display mode index 6, toolbar "7:Stars")
export const STARS_MODE = 6;
export const MAX_STARS = 1 << 20;      // preallocated star capacity; active N is a uniform
const STAR_STATS_SAMPLE = 65536;       // positions read back for headless uniformity stats
// Emoji identity atlas: same 61-glyph palette as the report playground; id ->
// Knuth hash -> cell, so the same identity always renders the same emoji.
const EMOJIS = ['🍎','🍊','🍋','🍉','🍇','🍓','🍒','🥝','🍍','🥑','🌽','🥕','🍄','🌸','🌻',
    '🌷','🍀','🌵','🌲','🍁','🐶','🐱','🐭','🐰','🦊','🐻','🐼','🐨','🐯','🦁','🐮','🐷',
    '🐸','🐵','🐔','🐧','🦆','🦉','🐢','🐍','🐙','🦀','🐠','🐬','🐳','⭐','🌙','⚡','🔥',
    '💧','🌈','🎈','🎲','🎯','🎸','🚀','🔑','💎','🧲','🪐','🫧','🍩'];
const ATLAS_GRID = 8;                  // 8x8 cells
const ATLAS_CELL_PX = 64;              // per-glyph raster size
const GLYPH_HALF_BASE = 8;             // emoji sprite half-extent in texels at 1024 wide

const BN_INV_SIGMA_TABLE = [
    1.083608, 1.035389, 1.023437, 1.017777, 1.014435,
    1.012213, 1.010624, 1.009426, 1.008490, 1.007737,
];

// ---------------------------------------------------------------------------
// PRNG (CPU-side init only)
// ---------------------------------------------------------------------------

function mulberry32(state) {
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) | 0;
    return [(t ^ (t >>> 14)) >>> 0, state];
}

function makeRng(seed) {
    let state = seed | 0;
    return () => { let v; [v, state] = mulberry32(state); return (v >>> 0) / 4294967296; };
}

function makeRandn(seed) {
    const rng = makeRng(seed);
    let spare = null;
    return () => {
        if (spare !== null) { const s = spare; spare = null; return s; }
        let u1 = rng(); while (u1 < 1e-10) u1 = rng();
        const u2 = rng(), r = Math.sqrt(-2 * Math.log(u1)), th = 6.283185307179586 * u2;
        spare = r * Math.sin(th);
        return r * Math.cos(th);
    };
}

// ---------------------------------------------------------------------------
// Shadow map helpers
// ---------------------------------------------------------------------------

/**
 * Pure function. Build an orthographic light-space matrix for a directional
 * shadow map. Looks from sunDir toward origin, covering a square world-space
 * area of ±halfExtent units and a depth range of depthRange units.
 *
 * Returns a column-major Float32Array(16) in WebGPU NDC convention (depth 0→1).
 *
 * Args:
 *   sunDir (number[4]): normalized sun direction [x, y, z, 0]
 *   halfExtent (number): ortho half-width and half-height in world units
 *   depthRange (number): total depth extent of the ortho frustum
 *
 * Returns:
 *   Float32Array(16)
 *
 * Examples:
 *   >>> buildLightSpaceMatrix([0,1,0,0], 100, 300).length
 *   16
 */
function buildLightSpaceMatrix(sunDir, halfExtent, depthRange) {
    const [lx, ly, lz] = sunDir;

    // Build an orthonormal frame with Z pointing along -sunDir (into the scene).
    // Light "looks" from the sun toward origin along -lightZ.
    const lightZ = [-lx, -ly, -lz];

    // Choose a stable world-up reference; fall back to world-forward when sun is near zenith.
    const upRef  = Math.abs(ly) > 0.99 ? [0, 0, 1] : [0, 1, 0];
    const lightX = normalize3(cross3(upRef, lightZ));
    const lightY = cross3(lightZ, lightX);

    // View matrix: world → light space (lookAt from sun position toward origin)
    // We don't need an actual translation since the ortho covers the entire area.
    // The camera sits "depthRange/2" units above the scene along -lightZ.
    const camPos = [lx * depthRange * 0.5, ly * depthRange * 0.5, lz * depthRange * 0.5];
    const tx = -(dot3(lightX, camPos));
    const ty = -(dot3(lightY, camPos));
    const tz = -(dot3(lightZ, camPos));

    // Column-major 4×4 view matrix (rows of the rotation become columns)
    const view = new Float32Array([
        lightX[0], lightY[0], lightZ[0], 0,
        lightX[1], lightY[1], lightZ[1], 0,
        lightX[2], lightY[2], lightZ[2], 0,
        tx,        ty,        tz,        1,
    ]);

    // Orthographic projection: maps ±halfExtent → ±1 in X/Y, 0..depthRange → 0..1 in Z.
    // WebGPU uses depth [0,1], so near maps to 0, far to 1.
    const r = halfExtent;
    const proj = new Float32Array([
        1/r,   0,    0,    0,
        0,     1/r,  0,    0,
        0,     0,    1/depthRange, 0,
        0,     0,    0,    1,
    ]);

    return mat4mul(proj, view);
}

function normalize3(v) {
    const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    return [v[0]/len, v[1]/len, v[2]/len];
}

function cross3(a, b) {
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ];
}

function dot3(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

/** Pure function. Column-major 4×4 matrix multiply: returns A*B. */
function mat4mul(a, b) {
    const m = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
        for (let row = 0; row < 4; row++) {
            let s = 0;
            for (let k = 0; k < 4; k++) s += a[k*4+row] * b[col*4+k];
            m[col*4+row] = s;
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// WebGPU Renderer
// ---------------------------------------------------------------------------

export class WebGPURenderer {
    constructor(canvas, W, H) {
        this.canvas = canvas;
        this.W = W;
        this.H = H;
        this.N = W * H;
        this.C = 4;
        this.frameCount = 0;

        this.hasTimestamps = false;
        this._tsMapping = false;
        this._gpuTimings = null;
        this._gpuTimingHistory = [];

        this._statsMapping = false;
        this.noiseStats = { mean: 0, std: 1 };

        this.blueNoiseEnabled = false;
        this.blueNoiseIterations = 2;
        this.blueNoiseCutoffDivider = 8.0;

        this.greyscaleEnabled = false;
        this.uniformDisplayEnabled = false;
        this.noiseOpacity = 0.25;
        this.noiseLocked = false;
        this._wasLocked = false;
        this.numStars = 10000;
        this.starAAEnabled = true;
        this.starFieldView = 0;   // 0 off, 1 density E, 2 deficit — turbo bg under stars
        this.starEmojiEnabled = false;   // render id-hashed emoji sprites
        this.starColorQEnabled = false;  // tint stars by turbo(strength)
        this.starSizeQEnabled = false;   // scale star footprint by strength
        this.starSizeMaxPx = 8;          // q-size mode: full width of a q~1 star
        this._stereoActive = false;      // stars mode + stereo enabled this frame
        this.stereoSwapEnabled = false;  // anaglyph: swap which eye is red vs blue
        this._starStatsMapping = false;
        this.shadowsEnabled = true;
        this.shadowResolution = 4096;
        this.pointLightsEnabled = true;
        this.terrainEnabled = true;
        this.daySpeedMultiplier = 0.2;  // 5x slower than original (1500-sec cycle)
    }

    async init() {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('WebGPU: no adapter found');

        this.hasTimestamps = adapter.features.has('timestamp-query');
        const features = this.hasTimestamps ? ['timestamp-query'] : [];
        this.device = await adapter.requestDevice({
            requiredFeatures: features,
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                maxBufferSize: adapter.limits.maxBufferSize,
                maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
            },
        });
        this.device.lost.then(info => { throw new Error('WebGPU device lost: ' + info.message); });

        this.ctx = this.canvas.getContext('webgpu');
        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        this.ctx.configure({ device: this.device, format: this.canvasFormat, alphaMode: 'opaque' });

        this._createTextures();
        this._createBuffers();
        this._createPipelines();
        this._createBindGroups();
        this._createVertexBuffers();
        this._initProfiler();
        this._initNoise();
        this._initStars();
    }

    destroy() {
        this.colorTex?.destroy();
        this.motionTex?.destroy();
        this.depthTex?.destroy();
        this.shadowTex?.destroy();
        this.starTex?.destroy();
        this.starTexR?.destroy();
        this.motionTexR?.destroy();
        this.crossTexL?.destroy();
        this.crossTexR?.destroy();
        this.atlasTex?.destroy();

        const bufs = [
            this.noiseBuf, this.bufferBuf, this.totalRequestBuf, this.ticketCountBuf,
            this.masterFieldBuf, this.areaFieldBuf, this.deformationBuf,
            this.cameraUniformBuf, this.skyUniformBuf, this.shadowUniformBuf, this.lightUniformBuf, this.instanceBuf,
            this.computeUniformBuf, this.displayUniformBuf,
            this._statsStagingBuf,
            this.bnBackupBuf,
            this.bnBlurHUniformBuf,
            ...(this.bnBlurVUniformBufs || []),
            this.starBuf, this.starMetaBuf, this.starCountersBuf,
            this.starBufR, this.starMetaBufR, this.starUniformBufR, this.cameraUniformBufR,
            this.mergedPosL, this.mergedMetaL, this.mergedPosR, this.mergedMetaR,
            this.mergeIndirectL, this.mergeIndirectR,
            this.starRowCdfBuf, this.starUniformBuf, this.starRenderUniformBuf, this._starStagingBuf,
            this.starDensityBuf, this.starRowPrefixBuf,
            this.boxVB, this.sphereVB, this.quadVB, this.terrainVB,
        ];
        if (this.hasTimestamps) bufs.push(this.querySet, this.tsResolveBuf, this.tsReadBuf);
        for (const b of bufs) b?.destroy?.();
    }

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------

    _createTextures() {
        const { device, W, H } = this;
        this.colorTex = device.createTexture({
            size: [W, H], format: 'rgba8unorm',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.motionTex = device.createTexture({
            size: [W, H], format: 'rgba32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        // Stereo: right-eye temporal motion + per-eye cross-eye flow targets.
        this.motionTexR = device.createTexture({
            size: [W, H], format: 'rgba32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.crossTexL = device.createTexture({   // written by eye L: flow L->R
            size: [W, H], format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.crossTexR = device.createTexture({   // written by eye R: flow R->L
            size: [W, H], format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.depthTex = device.createTexture({
            size: [W, H], format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        // Shadow map: configurable resolution depth texture, sampled in scene shader for PCF.
        this.shadowTex = device.createTexture({
            size: [this.shadowResolution, this.shadowResolution], format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        // Star coverage accumulator: LINEAR-light tent splats, sRGB-encoded at display.
        // rgba16float so additive blending of fractional coverage doesn't quantize.
        this.starTex = device.createTexture({
            size: [W, H], format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.starTexR = device.createTexture({    // right eye (stereo only)
            size: [W, H], format: 'rgba16float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        // Emoji identity atlas: rasterize the palette once via 2D canvas (emoji
        // fonts come free), upload unpremultiplied; the shader premultiplies.
        const atlasPx = ATLAS_GRID * ATLAS_CELL_PX;
        const atlasCanvas = new OffscreenCanvas(atlasPx, atlasPx);
        const actx = atlasCanvas.getContext('2d');
        actx.font = `${Math.round(ATLAS_CELL_PX * 0.8)}px sans-serif`;
        actx.textAlign = 'center';
        actx.textBaseline = 'middle';
        EMOJIS.forEach((glyph, g) => {
            actx.fillText(glyph,
                (g % ATLAS_GRID + 0.5) * ATLAS_CELL_PX,
                (Math.floor(g / ATLAS_GRID) + 0.5) * ATLAS_CELL_PX);
        });
        this.atlasTex = device.createTexture({
            size: [atlasPx, atlasPx], format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
                   GPUTextureUsage.RENDER_ATTACHMENT,
        });
        device.queue.copyExternalImageToTexture(
            { source: atlasCanvas }, { texture: this.atlasTex }, [atlasPx, atlasPx]);
        this.atlasSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

        this.colorTexView  = this.colorTex.createView();
        this.motionTexView = this.motionTex.createView();
        this.depthTexView  = this.depthTex.createView();
        this.shadowTexView = this.shadowTex.createView({ aspect: 'depth-only' });
        this.starTexView   = this.starTex.createView();
        this.starTexRView  = this.starTexR.createView();
        this.motionTexRView = this.motionTexR.createView();
        this.crossTexLView = this.crossTexL.createView();
        this.crossTexRView = this.crossTexR.createView();
        this.atlasTexView  = this.atlasTex.createView();
    }

    _createBuffers() {
        const { device, N, C } = this;
        const MAX_TICKETS = 24;
        const f4 = 4;

        const storage = (size, extra = 0) => device.createBuffer({
            size, usage: GPUBufferUsage.STORAGE | extra,
        });

        this.noiseBuf = storage(N * C * f4, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        this.bufferBuf       = storage(N * C * f4, GPUBufferUsage.COPY_DST);
        this.totalRequestBuf = storage(N * f4, GPUBufferUsage.COPY_DST);
        this.ticketCountBuf  = storage(N * f4, GPUBufferUsage.COPY_DST);
        this.masterFieldBuf  = storage(N * MAX_TICKETS * f4);
        this.areaFieldBuf    = storage(N * MAX_TICKETS * f4);
        this.deformationBuf  = storage(N * 2 * f4);

        // Camera uniform: viewProj (64) + prevViewProj (64) + sunDir (16) + lightSpaceMatrix (64)
        //                 + eyePos (16) + eyeDir (16) + otherViewProj (64) = 304 bytes
        this.cameraUniformBuf = device.createBuffer({
            size: 304, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.cameraUniformBufR = device.createBuffer({    // right stereo eye
            size: 304, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Shadow uniform: lightSpaceMatrix (64 bytes) for the shadow depth pass
        this.shadowUniformBuf = device.createBuffer({
            size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Sky uniform: invViewProj (64) + sunDir (16) + time vec4 (16) = 96 bytes
        this.skyUniformBuf = device.createBuffer({
            size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Point light buffer: count(u32) + 3 pad(u32) + 32 × PointLight(2 × vec4f = 32 bytes)
        // Total: 16 + 32 × 32 = 1040 bytes
        this.lightUniformBuf = device.createBuffer({
            size: 1040, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Instance storage buffer: MAX_INSTANCES × 144 bytes
        this.instanceBuf = device.createBuffer({
            size: MAX_INSTANCES * FLOATS_PER_INSTANCE * f4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.computeUniformBuf = device.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.displayUniformBuf = device.createBuffer({
            size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._statsStagingBuf = device.createBuffer({
            size: N * C * f4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        // --- Star warp buffers ---
        this.starBuf = device.createBuffer({           // (x, y) per star, pixel coords
            size: MAX_STARS * 2 * f4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        // Star strength q (U[0,1) at birth, eroded by crowding, dies at 1) and
        // persistent identity per star, kept OUT of the stride-2 position
        // buffer so the render and stats paths stay untouched.
        // {q, id} interleaved: ONE buffer, so starUpdate stays at 8 storage
        // buffers — the baseline maxStorageBuffersPerShaderStage.
        this.starMetaBuf = storage(MAX_STARS * 2 * f4, GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        // counters: [0] fresh-birth id mint, [1] cumulative deaths (diagnostics).
        this.starCountersBuf = storage(2 * f4, GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        // Stereo: an independent right-eye stream + per-eye merge outputs.
        this.starBufR = device.createBuffer({
            size: MAX_STARS * 2 * f4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.starMetaBufR = storage(MAX_STARS * 2 * f4, GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
        this.mergedPosL  = storage(2 * MAX_STARS * 2 * f4);
        this.mergedMetaL = storage(2 * MAX_STARS * 2 * f4);
        this.mergedPosR  = storage(2 * MAX_STARS * 2 * f4);
        this.mergedMetaR = storage(2 * MAX_STARS * 2 * f4);
        // drawIndirect args per eye: [vertexCount, 1, 0, 0]; vertexCount grows atomically.
        this.mergeIndirectL = device.createBuffer({
            size: 16, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE |
                             GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.mergeIndirectR = device.createBuffer({
            size: 16, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE |
                             GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.starDensityBuf   = storage(N * f4, GPUBufferUsage.COPY_DST);  // E (CAS-atomic f32)
        this.starRowPrefixBuf = storage(N * f4);       // per-row deficit prefix sums
        this.starRowCdfBuf    = storage(this.H * f4);  // row CDF; last entry = total deficit
        this.starUniformBuf = device.createBuffer({
            size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.starUniformBufR = device.createBuffer({      // right stream: different frameSeed
            size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.starRenderUniformBuf = device.createBuffer({
            size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // Layout: [0, SS*8) positions, [SS*8, SS*16) meta {q,id}, [SS*16, +16) counters.
        this._starStagingBuf = device.createBuffer({
            size: STAR_STATS_SAMPLE * 4 * f4 + 8 * f4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        this.bnBackupBuf = device.createBuffer({
            size: N * C * f4,
            usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.bnBlurHUniformBuf = device.createBuffer({
            size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.bnBlurVUniformBufs = [];
        for (let i = 0; i < BN_INV_SIGMA_TABLE.length; i++) {
            this.bnBlurVUniformBufs.push(device.createBuffer({
                size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
        }
    }

    _createPipelines() {
        const { device } = this;
        const mod = (code) => device.createShaderModule({ code });
        const skyModule       = mod(skyWGSL);
        const sceneModule     = mod(sceneWGSL);
        const shadowModule    = mod(shadowWGSL);
        const displayModule   = mod(displayWGSL);
        const buildDeformMod  = mod(buildDeformWGSL);
        const backwardMapMod  = mod(backwardMapWGSL);
        const brownianMod     = mod(brownianWGSL);
        const normalizeMod    = mod(normalizeWGSL);

        // Shadow pipeline: depth-only, no fragment shader, same vertex layout as scene.
        this.shadowPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shadowModule, entryPoint: 'vs',
                buffers: [{
                    arrayStride: 6 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0,  format: 'float32x3' },
                        { shaderLocation: 1, offset: 12, format: 'float32x3' },
                    ],
                }],
            },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: true,
                depthCompare: 'less',
            },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
        });

        // Scene pipeline: instanced, 6 floats per vertex (position + normal)
        this.scenePipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: sceneModule, entryPoint: 'vs',
                buffers: [{
                    arrayStride: 6 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0,  format: 'float32x3' },
                        { shaderLocation: 1, offset: 12, format: 'float32x3' },
                    ],
                }],
            },
            fragment: {
                module: sceneModule, entryPoint: 'fs',
                targets: [
                    { format: 'rgba8unorm' },
                    { format: 'rgba32float' },
                    { format: 'rgba16float' },   // cross-eye flow (zero in mono)
                ],
            },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: true,
                depthCompare: 'less',
            },
            primitive: { topology: 'triangle-list', cullMode: 'back' },
        });

        // Sky pipeline: fullscreen quad, writes to same MRT as scene, depth = 1.0
        this.skyPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: skyModule, entryPoint: 'vs',
                buffers: [{
                    arrayStride: 4 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x2' },
                    ],
                }],
            },
            fragment: {
                module: skyModule, entryPoint: 'fs',
                targets: [
                    { format: 'rgba8unorm' },
                    { format: 'rgba32float' },
                    { format: 'rgba16float' },   // cross-eye flow (sky: zero disparity)
                ],
            },
            depthStencil: {
                format: 'depth24plus',
                depthWriteEnabled: false,
                depthCompare: 'always',  // always pass; depth already cleared to 1.0
            },
            primitive: { topology: 'triangle-list' },
        });

        // Display pipeline (same as V2)
        this.displayPipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: displayModule, entryPoint: 'vs',
                buffers: [{
                    arrayStride: 4 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: 'float32x2' },
                        { shaderLocation: 1, offset: 8, format: 'float32x2' },
                    ],
                }],
            },
            fragment: {
                module: displayModule, entryPoint: 'fs',
                targets: [{ format: this.canvasFormat }],
            },
            primitive: { topology: 'triangle-list' },
        });

        // Compute pipelines (same as V2)
        const computePipeline = (module, constants) => device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main', ...(constants ? { constants } : {}) },
        });

        this.brownianWG = this.brownianWGOverride || 256;
        this.buildDeformPipeline = computePipeline(buildDeformMod);
        this.backwardMapPipeline = computePipeline(backwardMapMod);
        this.brownianPipeline    = computePipeline(brownianMod, { WG_SIZE: this.brownianWG });
        this.normalizePipeline   = computePipeline(normalizeMod);
        this.bnBlurPipeline      = computePipeline(mod(blueNoiseBlurWGSL));

        // Star warp pipelines
        this.starSplatPipeline   = computePipeline(mod(starSplatWGSL));
        this.starScanRowsPipeline = computePipeline(mod(starScanRowsWGSL));
        this.starScanCdfPipeline = computePipeline(mod(starScanCdfWGSL));
        this.starUpdatePipeline  = computePipeline(mod(starUpdateWGSL));
        this.mergeSelectPipeline = computePipeline(mod(mergeSelectWGSL));

        // Star render: additive blending accumulates LINEAR-light tent coverage.
        const starModule = mod(starRenderWGSL);
        const starRenderPipelineWithBlend = (blend) => device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: starModule, entryPoint: 'vs' },
            fragment: {
                module: starModule, entryPoint: 'fs',
                targets: [{ format: 'rgba16float', blend }],
            },
            primitive: { topology: 'triangle-list' },
        });
        // White tents accumulate additively (linear-light coverage in rgb AND a);
        // emoji sprites composite premultiplied-over so overlaps don't blow out.
        this.starRenderPipeline = starRenderPipelineWithBlend({
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        });
        this.starRenderPipelineEmoji = starRenderPipelineWithBlend({
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        });
    }

    _createBindGroups() {
        const { device } = this;
        const buf = (b) => ({ buffer: b });

        // Sky bind group: sky uniforms
        this.skyBindGroup = device.createBindGroup({
            layout: this.skyPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.skyUniformBuf) },
            ],
        });

        // Shadow comparison sampler — used by scene shader for PCF
        this.shadowSampler = device.createSampler({
            compare: 'less',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // Shadow pass bind group: light-space uniform + instance storage
        this.shadowPassBindGroup = device.createBindGroup({
            layout: this.shadowPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.shadowUniformBuf) },
                { binding: 1, resource: buf(this.instanceBuf) },
            ],
        });

        // Scene bind group: camera uniform + instance storage + shadow map + shadow sampler + lights
        const sceneEntries = (camBuf) => [
            { binding: 0, resource: buf(camBuf) },
            { binding: 1, resource: buf(this.instanceBuf) },
            { binding: 2, resource: this.shadowTexView },
            { binding: 3, resource: this.shadowSampler },
            { binding: 4, resource: buf(this.lightUniformBuf) },
        ];
        this.sceneBindGroup = device.createBindGroup({
            layout: this.scenePipeline.getBindGroupLayout(0),
            entries: sceneEntries(this.cameraUniformBuf),
        });
        this.sceneBindGroupR = device.createBindGroup({
            layout: this.scenePipeline.getBindGroupLayout(0),
            entries: sceneEntries(this.cameraUniformBufR),
        });

        // Display bind group
        this.displayBindGroup = device.createBindGroup({
            layout: this.displayPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.noiseBuf) },
                { binding: 1, resource: this.colorTexView },
                { binding: 2, resource: this.motionTexView },
                { binding: 3, resource: buf(this.displayUniformBuf) },
                { binding: 4, resource: this.starTexView },
                { binding: 5, resource: buf(this.starDensityBuf) },
                { binding: 6, resource: this.starTexRView },
            ],
        });

        // Star warp bind groups
        const splatGroup = (uniformBuf, texView) => device.createBindGroup({
            layout: this.starSplatPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(uniformBuf) },
                { binding: 1, resource: texView },
                { binding: 2, resource: buf(this.starDensityBuf) },
            ],
        });
        this.starSplatBindGroup = splatGroup(this.starUniformBuf, this.motionTexView);
        this.starSplatBindGroupR = splatGroup(this.starUniformBufR, this.motionTexRView);
        // Stereo merge reuses the SAME splat kernel to transport the other
        // eye's uniform mass along the cross flow into this eye's density.
        this.mergeSplatBindGroupL = splatGroup(this.starUniformBuf, this.crossTexRView);
        this.mergeSplatBindGroupR = splatGroup(this.starUniformBuf, this.crossTexLView);
        this.starScanRowsBindGroup = device.createBindGroup({
            layout: this.starScanRowsPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.starUniformBuf) },
                { binding: 1, resource: buf(this.starDensityBuf) },
                { binding: 2, resource: buf(this.starRowPrefixBuf) },
            ],
        });
        this.starScanCdfBindGroup = device.createBindGroup({
            layout: this.starScanCdfPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.starUniformBuf) },
                { binding: 1, resource: buf(this.starRowPrefixBuf) },
                { binding: 2, resource: buf(this.starRowCdfBuf) },
            ],
        });
        this.starUpdateBindGroup = device.createBindGroup({
            layout: this.starUpdatePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.starUniformBuf) },
                { binding: 1, resource: this.motionTexView },
                { binding: 2, resource: buf(this.starDensityBuf) },
                { binding: 3, resource: buf(this.starRowPrefixBuf) },
                { binding: 4, resource: buf(this.starRowCdfBuf) },
                { binding: 5, resource: buf(this.starBuf) },
                { binding: 6, resource: buf(this.starMetaBuf) },
                { binding: 7, resource: buf(this.starCountersBuf) },
            ],
        });
        this.starUpdateBindGroupR = device.createBindGroup({
            layout: this.starUpdatePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.starUniformBufR) },
                { binding: 1, resource: this.motionTexRView },
                { binding: 2, resource: buf(this.starDensityBuf) },
                { binding: 3, resource: buf(this.starRowPrefixBuf) },
                { binding: 4, resource: buf(this.starRowCdfBuf) },
                { binding: 5, resource: buf(this.starBufR) },
                { binding: 6, resource: buf(this.starMetaBufR) },
                { binding: 7, resource: buf(this.starCountersBuf) },
            ],
        });
        // Stereo merge select: own stream + other stream reprojected by the
        // cross flow, thinned against 1 + transported density (see shader).
        const mergeGroup = (crossView, ownPos, ownMeta, otherPos, otherMeta, mergedPos, mergedMeta, indirect) =>
            device.createBindGroup({
                layout: this.mergeSelectPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: buf(this.starUniformBuf) },
                    { binding: 1, resource: crossView },
                    { binding: 2, resource: buf(this.starDensityBuf) },
                    { binding: 3, resource: buf(ownPos) },
                    { binding: 4, resource: buf(ownMeta) },
                    { binding: 5, resource: buf(otherPos) },
                    { binding: 6, resource: buf(otherMeta) },
                    { binding: 7, resource: buf(mergedPos) },
                    { binding: 8, resource: buf(mergedMeta) },
                    { binding: 9, resource: buf(indirect) },
                ],
            });
        this.mergeSelectBindGroupL = mergeGroup(this.crossTexRView,
            this.starBuf, this.starMetaBuf, this.starBufR, this.starMetaBufR,
            this.mergedPosL, this.mergedMetaL, this.mergeIndirectL);
        this.mergeSelectBindGroupR = mergeGroup(this.crossTexLView,
            this.starBufR, this.starMetaBufR, this.starBuf, this.starMetaBuf,
            this.mergedPosR, this.mergedMetaR, this.mergeIndirectR);
        // Same entries for both render pipelines ('auto' layouts are per-pipeline).
        const starRenderEntries = () => [
            { binding: 0, resource: buf(this.starRenderUniformBuf) },
            { binding: 1, resource: buf(this.starBuf) },
            { binding: 2, resource: buf(this.starMetaBuf) },
            { binding: 3, resource: this.atlasTexView },
            { binding: 4, resource: this.atlasSampler },
        ];
        this.starRenderBindGroup = device.createBindGroup({
            layout: this.starRenderPipeline.getBindGroupLayout(0),
            entries: starRenderEntries(),
        });
        this.starRenderBindGroupEmoji = device.createBindGroup({
            layout: this.starRenderPipelineEmoji.getBindGroupLayout(0),
            entries: starRenderEntries(),
        });
        const mergedRenderEntries = (posBuf, metaBuf) => [
            { binding: 0, resource: buf(this.starRenderUniformBuf) },
            { binding: 1, resource: buf(posBuf) },
            { binding: 2, resource: buf(metaBuf) },
            { binding: 3, resource: this.atlasTexView },
            { binding: 4, resource: this.atlasSampler },
        ];
        this.mergedRenderBG = {};   // [eye][emoji 0|1] -> bind group
        for (const [eye, posBuf, metaBuf] of [['L', this.mergedPosL, this.mergedMetaL],
                                              ['R', this.mergedPosR, this.mergedMetaR]]) {
            this.mergedRenderBG[eye] = [
                device.createBindGroup({
                    layout: this.starRenderPipeline.getBindGroupLayout(0),
                    entries: mergedRenderEntries(posBuf, metaBuf),
                }),
                device.createBindGroup({
                    layout: this.starRenderPipelineEmoji.getBindGroupLayout(0),
                    entries: mergedRenderEntries(posBuf, metaBuf),
                }),
            ];
        }

        // Build deformation
        this.buildDeformBindGroup = device.createBindGroup({
            layout: this.buildDeformPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.computeUniformBuf) },
                { binding: 1, resource: this.motionTexView },
                { binding: 2, resource: buf(this.deformationBuf) },
            ],
        });

        // Backward map
        this.backwardMapBindGroup = device.createBindGroup({
            layout: this.backwardMapPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.computeUniformBuf) },
                { binding: 1, resource: buf(this.deformationBuf) },
                { binding: 2, resource: buf(this.ticketCountBuf) },
                { binding: 3, resource: buf(this.masterFieldBuf) },
                { binding: 4, resource: buf(this.areaFieldBuf) },
            ],
        });

        // Brownian bridge
        this.brownianBindGroup = device.createBindGroup({
            layout: this.brownianPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.computeUniformBuf) },
                { binding: 1, resource: buf(this.ticketCountBuf) },
                { binding: 2, resource: buf(this.masterFieldBuf) },
                { binding: 3, resource: buf(this.areaFieldBuf) },
                { binding: 4, resource: buf(this.noiseBuf) },
                { binding: 5, resource: buf(this.bufferBuf) },
                { binding: 6, resource: buf(this.totalRequestBuf) },
            ],
        });

        // Normalize
        this.normalizeBindGroup = device.createBindGroup({
            layout: this.normalizePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.computeUniformBuf) },
                { binding: 1, resource: buf(this.bufferBuf) },
                { binding: 2, resource: buf(this.deformationBuf) },
                { binding: 3, resource: buf(this.totalRequestBuf) },
                { binding: 4, resource: buf(this.noiseBuf) },
            ],
        });

        // Blue noise blur bind groups
        const bnBlurLayout = this.bnBlurPipeline.getBindGroupLayout(0);
        this.bnBlurHBindGroup = device.createBindGroup({
            layout: bnBlurLayout,
            entries: [
                { binding: 0, resource: buf(this.bnBlurHUniformBuf) },
                { binding: 1, resource: buf(this.noiseBuf) },
                { binding: 2, resource: buf(this.bufferBuf) },
            ],
        });
        this.bnBlurVBindGroups = this.bnBlurVUniformBufs.map(vBuf =>
            device.createBindGroup({
                layout: bnBlurLayout,
                entries: [
                    { binding: 0, resource: buf(vBuf) },
                    { binding: 1, resource: buf(this.bufferBuf) },
                    { binding: 2, resource: buf(this.noiseBuf) },
                ],
            })
        );
    }

    _createVertexBuffers() {
        const { device } = this;
        const uploadVB = (data) => {
            const vb = device.createBuffer({
                size: data.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(vb, 0, data);
            return vb;
        };

        const boxData = boxVertices();
        const bevelData = beveledBoxVertices(0.75, 1.5, 0.18, 0.04, 2);
        const sphereData = sphereVertices();
        const quadData = quadVertices();
        const terrainData = terrainMeshVertices(600, 4500);

        this.boxVB = uploadVB(boxData);
        this.boxVertCount = boxData.length / 6;
        this.bevelBoxVB = uploadVB(bevelData);
        this.bevelBoxVertCount = bevelData.length / 6;
        this.sphereVB = uploadVB(sphereData);
        this.sphereVertCount = sphereData.length / 6;
        this.quadVB = uploadVB(quadData);
        this.quadVertCount = quadData.length / 4;
        this.terrainVB = uploadVB(terrainData);
        this.terrainVertCount = terrainData.length / 6;

        // Write static terrain instance: identity model matrix, sentinel color
        this._writeTerrainInstance();
    }

    /**
     * Write the terrain instance data (identity model, sentinel color) into the
     * instance buffer at TERRAIN_INSTANCE_IDX. Called once after buffer creation.
     * Not pure: writes to GPU buffer.
     */
    _writeTerrainInstance() {
        // Identity mat4 (column-major)
        const identity = new Float32Array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ]);
        // Sentinel terrain color: R=0.12, G=0.48, B=0.08, A=1.0
        // (distinct from all other materials, detected in the fragment shader)
        const terrainColor = new Float32Array([0.12, 0.48, 0.08, 1.0]);

        const instanceData = new Float32Array(FLOATS_PER_INSTANCE);
        instanceData.set(identity, 0);   // current model
        instanceData.set(identity, 16);  // prev model (same — static)
        instanceData.set(terrainColor, 32);

        const byteOffset = TERRAIN_INSTANCE_IDX * FLOATS_PER_INSTANCE * 4;
        this.device.queue.writeBuffer(this.instanceBuf, byteOffset, instanceData);
    }

    _initProfiler() {
        if (!this.hasTimestamps) return;
        this.querySet = this.device.createQuerySet({ type: 'timestamp', count: NUM_TIMESTAMPS });
        this.tsResolveBuf = this.device.createBuffer({
            size: NUM_TIMESTAMPS * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        this.tsReadBuf = this.device.createBuffer({
            size: NUM_TIMESTAMPS * 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
    }

    _initNoise() {
        const { H, W, C, device } = this;
        const randn = makeRandn(12345);
        const data = new Float32Array(H * W * C);
        for (let i = 0; i < data.length; i++) data[i] = randn();
        device.queue.writeBuffer(this.noiseBuf, 0, data);
        this._updateBlurUniforms();
    }

    /**
     * Seed ALL MAX_STARS positions uniformly over [0, W] x [0, H] and all
     * strengths uniformly over [0, 1), not just the active N: raising N mid-run
     * then exposes stale-but-uniform stars, which is spatially valid (uniform is
     * uniform, and among living stars q is uniform too) and artifact-free.
     * Not pure: writes to GPU buffers.
     */
    _initStars() {
        const { H, W, device } = this;
        const seed = (posBuf, metaTarget, rngSeed, idBase) => {
            const rng = makeRng(rngSeed);
            const data = new Float32Array(MAX_STARS * 2);
            const metaBuf = new ArrayBuffer(MAX_STARS * 2 * 4);   // {q: f32, id: u32}
            const metaF32 = new Float32Array(metaBuf);
            const metaU32 = new Uint32Array(metaBuf);
            for (let i = 0; i < MAX_STARS; i++) {
                data[i * 2]      = rng() * W;
                data[i * 2 + 1]  = rng() * H;
                metaF32[i * 2]   = rng();          // strength q
                metaU32[i * 2+1] = idBase + i;     // identity (streams get disjoint ids)
            }
            device.queue.writeBuffer(posBuf, 0, data);
            device.queue.writeBuffer(metaTarget, 0, metaBuf);
        };
        seed(this.starBuf, this.starMetaBuf, 777, 0);
        seed(this.starBufR, this.starMetaBufR, 778, MAX_STARS);
        device.queue.writeBuffer(this.starCountersBuf, 0, new Uint32Array([2 * MAX_STARS, 0]));
    }

    _updateBlurUniforms() {
        const { H, W, device } = this;
        const D0 = Math.min(H, W) / this.blueNoiseCutoffDivider;
        const sigma = H / (2 * Math.PI * D0);
        const grey = this.greyscaleEnabled ? 1 : 0;

        const hBuf = new ArrayBuffer(32);
        const hU32 = new Uint32Array(hBuf);
        const hF32 = new Float32Array(hBuf);
        hU32[0] = H; hU32[1] = W; hF32[2] = sigma; hU32[3] = 0; hF32[4] = 0; hU32[5] = grey;
        device.queue.writeBuffer(this.bnBlurHUniformBuf, 0, hBuf);

        for (let i = 0; i < BN_INV_SIGMA_TABLE.length; i++) {
            const vBuf = new ArrayBuffer(32);
            const vU32 = new Uint32Array(vBuf);
            const vF32 = new Float32Array(vBuf);
            vU32[0] = H; vU32[1] = W; vF32[2] = sigma; vU32[3] = 1;
            vF32[4] = BN_INV_SIGMA_TABLE[i]; vU32[5] = grey;
            device.queue.writeBuffer(this.bnBlurVUniformBufs[i], 0, vBuf);
        }
    }

    _encodeBlueNoise(encoder, workgroups256) {
        const greyFlag = new Uint32Array([this.greyscaleEnabled ? 1 : 0]);
        this.device.queue.writeBuffer(this.bnBlurHUniformBuf, 20, greyFlag);
        for (const vBuf of this.bnBlurVUniformBufs) {
            this.device.queue.writeBuffer(vBuf, 20, greyFlag);
        }

        for (let i = 0; i < this.blueNoiseIterations; i++) {
            const blurH = encoder.beginComputePass();
            blurH.setPipeline(this.bnBlurPipeline);
            blurH.setBindGroup(0, this.bnBlurHBindGroup);
            blurH.dispatchWorkgroups(workgroups256);
            blurH.end();

            const blurV = encoder.beginComputePass();
            blurV.setPipeline(this.bnBlurPipeline);
            blurV.setBindGroup(0, this.bnBlurVBindGroups[i]);
            blurV.dispatchWorkgroups(workgroups256);
            blurV.end();
        }
    }

    // -----------------------------------------------------------------------
    // Per-frame
    // -----------------------------------------------------------------------

    /**
     * Render one complete frame with instanced scene rendering + warp pipeline.
     * Not pure: mutates GPU state.
     *
     * @param {object} opts
     * @param {Float32Array} opts.viewProj - current viewProj matrix
     * @param {Float32Array} opts.prevViewProj - previous frame's viewProj
     * @param {Float32Array} opts.invViewProj - inverse of current viewProj (for sky ray reconstruction)
     * @param {Float32Array} opts.instanceData - active instance data
     * @param {number} opts.numBoxInstances - total box instance count (floor + dominoes + maze + chest)
     * @param {number} opts.numSphereInstances - sphere instance count
     * @param {number} opts.numDominoInstances - domino-only count (for beveled VB draw call)
     * @param {number} opts.displayMode - 0-4
     * @param {number} opts.frameSeed - incrementing seed
     * @param {number} opts.elapsedSecs - seconds since page load, drives day/night cycle
     * @param {number[]} opts.eyePos - [x, y, z] camera world position (for flashlight)
     * @param {number[]} opts.eyeDir - [x, y, z] camera forward unit vector (for flashlight)
     */
    frame({ viewProj, prevViewProj, invViewProj, instanceData, numBoxInstances, numSphereInstances, numDominoInstances, displayMode, frameSeed, elapsedSecs = 0, eyePos = [0,0,0], eyeDir = [0,0,-1], lights = [], stereo = null }) {
        const { device, W, H, N } = this;
        const workgroups256 = Math.ceil(N / 256);
        const brownianWGs = Math.ceil(N / this.brownianWG);
        const starsMode = displayMode === STARS_MODE;
        const numStars = Math.min(this.numStars, MAX_STARS);

        // Day/night cycle: base period 300 seconds, scaled by daySpeedMultiplier.
        // daySpeedMultiplier=0 freezes time; 1.0 = original 5-min cycle; 3.0 = 100s cycle.
        const DAY_CYCLE_SECS = 300;
        const effectiveSecs = elapsedSecs * this.daySpeedMultiplier;
        const angle = (2 * Math.PI * effectiveSecs) / DAY_CYCLE_SECS + Math.PI / 2;
        // sunDir: x=cos(angle) sweeps east→west, y=sin(angle) rises/sets, z=slight tilt north
        const rawX = Math.cos(angle);
        const rawY = Math.sin(angle);
        const rawZ = 0.3;
        const len = Math.sqrt(rawX * rawX + rawY * rawY + rawZ * rawZ);
        const sunDir = [rawX / len, rawY / len, rawZ / len, 0.0];

        // Orthographic light-space matrix for directional shadow map.
        // Covers a 200×200 world-unit area centred at origin, depth range 0..300.
        const lightSpaceMatrix = buildLightSpaceMatrix(sunDir, 200, 300);

        // Stereo is a Stars-mode feature: two eye cameras, two star streams,
        // merged at render time (report §13). Mono path: otherViewProj ==
        // viewProj so the cross-eye motion target is exactly zero.
        const starsModeNow = displayMode === STARS_MODE;
        const stereoActive = starsModeNow && !!stereo && stereo.mode > 0;
        this._stereoActive = stereoActive;

        // Upload camera uniforms: viewProj (64) + prevViewProj (64) + sunDir (16) + lightSpaceMatrix (64)
        //                         + eyePos (16) + eyeDir (16) + otherViewProj (64) = 304 bytes (76 floats)
        const buildCamData = (vp, prevVp, otherVp) => {
            const camData = new Float32Array(76);
            camData.set(vp, 0);
            camData.set(prevVp, 16);
            camData.set(sunDir, 32);
            camData.set(lightSpaceMatrix, 36);
            camData.set([eyePos[0], eyePos[1], eyePos[2], 0.0], 52);
            camData.set([eyeDir[0], eyeDir[1], eyeDir[2], 0.0], 56);
            camData.set(otherVp, 60);
            return camData;
        };
        if (stereoActive) {
            device.queue.writeBuffer(this.cameraUniformBuf, 0,
                buildCamData(stereo.viewProjL, stereo.prevViewProjL, stereo.viewProjR));
            device.queue.writeBuffer(this.cameraUniformBufR, 0,
                buildCamData(stereo.viewProjR, stereo.prevViewProjR, stereo.viewProjL));
        } else {
            device.queue.writeBuffer(this.cameraUniformBuf, 0,
                buildCamData(viewProj, prevViewProj, viewProj));
        }

        // Shadow uniform: lightSpaceMatrix only (used in depth-only shadow pass)
        device.queue.writeBuffer(this.shadowUniformBuf, 0, lightSpaceMatrix);

        // Upload sky uniforms: invViewProj (64) + sunDir (16) + time vec4 (16)
        const skyData = new Float32Array(24);
        skyData.set(invViewProj, 0);
        skyData.set(sunDir, 16);
        skyData[20] = elapsedSecs;  // time.x = elapsed seconds for cloud animation
        // skyData[21..23] = padding zeros
        device.queue.writeBuffer(this.skyUniformBuf, 0, skyData);

        // Upload point lights: count(u32) + 3 pad(u32) + 32 × (posAndRadius vec4f + color vec4f)
        const lightData = new Float32Array(4 + 32 * 8);  // 260 floats = 1040 bytes
        const lightU32 = new Uint32Array(lightData.buffer);
        const numLights = this.pointLightsEnabled ? Math.min(lights.length, 32) : 0;
        lightU32[0] = numLights;
        for (let i = 0; i < numLights; i++) {
            const base = 4 + i * 8;
            lightData[base]     = lights[i].pos[0];
            lightData[base + 1] = lights[i].pos[1];
            lightData[base + 2] = lights[i].pos[2];
            lightData[base + 3] = lights[i].radius;
            lightData[base + 4] = lights[i].color[0];
            lightData[base + 5] = lights[i].color[1];
            lightData[base + 6] = lights[i].color[2];
            lightData[base + 7] = lights[i].intensity;
        }
        device.queue.writeBuffer(this.lightUniformBuf, 0, lightData);

        // Upload instance data
        if (instanceData.length > 0) {
            device.queue.writeBuffer(this.instanceBuf, 0, instanceData);
        }

        // Compute uniforms
        device.queue.writeBuffer(this.computeUniformBuf, 0,
            new Uint32Array([H, W, frameSeed, this.roundMode || 0]));

        // Display uniforms
        const displayFlags = (this.greyscaleEnabled ? 1 : 0) | (this.uniformDisplayEnabled ? 2 : 0);
        const dispBuf = new ArrayBuffer(48);
        const dispU32 = new Uint32Array(dispBuf);
        const dispF32 = new Float32Array(dispBuf);
        dispU32[0] = displayMode;
        dispU32[1] = W;
        dispU32[2] = H;
        dispU32[3] = displayFlags;
        dispU32[4] = this.thresholdOn || 0;
        dispF32[5] = this.thresholdValue || 0;
        dispF32[6] = this.noiseOpacity;
        const stereoNow = displayMode === STARS_MODE && !!stereo && stereo.mode > 0;
        dispU32[7] = stereoNow ? 0 : this.starFieldView;  // field bg is mono-only
        dispU32[8] = stereoNow ? stereo.mode : 0;
        dispU32[9] = this.stereoSwapEnabled ? 1 : 0;
        device.queue.writeBuffer(this.displayUniformBuf, 0, dispBuf);

        const encoder = device.createCommandEncoder();

        // Instance layout helpers (shared by shadow pass and scene pass)
        const numDominoes  = numDominoInstances || 0;
        const mazeStartIdx = 1 + numDominoes;
        const numMazeBoxes = numBoxInstances - mazeStartIdx;

        // --- Shadow pass: render all geometry into the 2048×2048 depth map ---
        // Only run when shadows are enabled AND sun is above horizon (night has no shadow).
        if (this.shadowsEnabled && sunDir[1] > 0.0) {
            const shadowPass = encoder.beginRenderPass({
                colorAttachments: [],
                depthStencilAttachment: {
                    view: this.shadowTexView,
                    depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0,
                },
            });
            shadowPass.setPipeline(this.shadowPipeline);
            shadowPass.setBindGroup(0, this.shadowPassBindGroup);

            // Floor
            shadowPass.setVertexBuffer(0, this.boxVB);
            shadowPass.draw(this.boxVertCount, 1, 0, 0);

            // Dominoes
            if (numDominoes > 0) {
                shadowPass.setVertexBuffer(0, this.bevelBoxVB);
                shadowPass.draw(this.bevelBoxVertCount, numDominoes, 0, 1);
            }

            // Remaining boxes
            if (numMazeBoxes > 0) {
                shadowPass.setVertexBuffer(0, this.boxVB);
                shadowPass.draw(this.boxVertCount, numMazeBoxes, 0, mazeStartIdx);
            }

            // Spheres
            if (numSphereInstances > 0) {
                shadowPass.setVertexBuffer(0, this.sphereVB);
                shadowPass.draw(this.sphereVertCount, numSphereInstances, 0, numBoxInstances);
            }

            // Terrain
            if (this.terrainEnabled) {
                shadowPass.setVertexBuffer(0, this.terrainVB);
                shadowPass.draw(this.terrainVertCount, 1, 0, TERRAIN_INSTANCE_IDX);
            }

            shadowPass.end();
        } else {
            // Shadows disabled or sun below horizon: clear shadow map to 1.0 so PCF always passes.
            const clearPass = encoder.beginRenderPass({
                colorAttachments: [],
                depthStencilAttachment: {
                    view: this.shadowTexView,
                    depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0,
                },
            });
            clearPass.end();
        }

        // --- Scene render (sky + instanced MRT), once per eye in stereo ---
        const encodeScene = (camBindGroup, motionView, crossView, withTimestamps) => {
            const scenePass = encoder.beginRenderPass({
                colorAttachments: [
                    { view: this.colorTexView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
                    { view: motionView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
                    { view: crossView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
                ],
                depthStencilAttachment: {
                    view: this.depthTexView,
                    depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0,
                },
                ...(withTimestamps && this.hasTimestamps && !this._tsMapping ? {
                    timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
                } : {}),
            });

            // Sky background: fullscreen quad at far plane
            scenePass.setPipeline(this.skyPipeline);
            scenePass.setBindGroup(0, this.skyBindGroup);
            scenePass.setVertexBuffer(0, this.quadVB);
            scenePass.draw(this.quadVertCount);

            // Scene geometry on top (depth < 1.0 wins)
            scenePass.setPipeline(this.scenePipeline);
            scenePass.setBindGroup(0, camBindGroup);

            // Draw floor (flat box): instance 0
            scenePass.setVertexBuffer(0, this.boxVB);
            scenePass.draw(this.boxVertCount, 1, 0, 0);

            // Draw dominoes (beveled box): instances 1..1+numDominoInstances
            if (numDominoes > 0) {
                scenePass.setVertexBuffer(0, this.bevelBoxVB);
                scenePass.draw(this.bevelBoxVertCount, numDominoes, 0, 1);
            }

            // Draw remaining boxes (maze, tower, marble machine, etc.)
            if (numMazeBoxes > 0) {
                scenePass.setVertexBuffer(0, this.boxVB);
                scenePass.draw(this.boxVertCount, numMazeBoxes, 0, mazeStartIdx);
            }

            // Draw spheres
            if (numSphereInstances > 0) {
                scenePass.setVertexBuffer(0, this.sphereVB);
                scenePass.draw(this.sphereVertCount, numSphereInstances, 0, numBoxInstances);
            }

            // Draw terrain mesh
            if (this.terrainEnabled) {
                scenePass.setVertexBuffer(0, this.terrainVB);
                scenePass.draw(this.terrainVertCount, 1, 0, TERRAIN_INSTANCE_IDX);
            }

            scenePass.end();
        };
        // In stereo the "main" pass IS the left eye (its motion drives the L stream).
        encodeScene(this.sceneBindGroup, this.motionTexView, this.crossTexLView, true);
        if (stereoActive) {
            encodeScene(this.sceneBindGroupR, this.motionTexRView, this.crossTexRView, false);
        }

        // --- Warp pipeline ---
        // Track lock transition: on the frame lock engages, bake blue noise
        // into noiseBuf so the snapshot includes it.
        const justLocked = this.noiseLocked && !this._wasLocked;
        this._wasLocked = this.noiseLocked;

        // Stars mode replaces the noise warp entirely (mutually exclusive
        // displays) — skipping the warp frees its ~12ms GPU budget.
        if (!this.noiseLocked && !starsMode) {

        // --- Build deformation ---
        const deformPass = encoder.beginComputePass(
            this.hasTimestamps && !this._tsMapping ? {
                timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 },
            } : undefined,
        );
        deformPass.setPipeline(this.buildDeformPipeline);
        deformPass.setBindGroup(0, this.buildDeformBindGroup);
        deformPass.dispatchWorkgroups(workgroups256);
        deformPass.end();

        // Clear intermediate buffers
        encoder.clearBuffer(this.bufferBuf);
        encoder.clearBuffer(this.totalRequestBuf);
        encoder.clearBuffer(this.ticketCountBuf);

        // --- Backward map ---
        const bwdPass = encoder.beginComputePass(
            this.hasTimestamps && !this._tsMapping ? {
                timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 4, endOfPassWriteIndex: 5 },
            } : undefined,
        );
        bwdPass.setPipeline(this.backwardMapPipeline);
        bwdPass.setBindGroup(0, this.backwardMapBindGroup);
        bwdPass.dispatchWorkgroups(workgroups256);
        bwdPass.end();

        // --- Brownian bridge ---
        const brPass = encoder.beginComputePass(
            this.hasTimestamps && !this._tsMapping ? {
                timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 6, endOfPassWriteIndex: 7 },
            } : undefined,
        );
        brPass.setPipeline(this.brownianPipeline);
        brPass.setBindGroup(0, this.brownianBindGroup);
        brPass.dispatchWorkgroups(brownianWGs);
        brPass.end();

        // --- Normalize ---
        const normPass = encoder.beginComputePass(
            this.hasTimestamps && !this._tsMapping ? {
                timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 8, endOfPassWriteIndex: 9 },
            } : undefined,
        );
        normPass.setPipeline(this.normalizePipeline);
        normPass.setBindGroup(0, this.normalizeBindGroup);
        normPass.dispatchWorkgroups(workgroups256);
        normPass.end();

        // --- Blue noise ---
        if (this.blueNoiseEnabled) {
            encoder.copyBufferToBuffer(this.noiseBuf, 0, this.bnBackupBuf, 0, N * this.C * 4);
            this._encodeBlueNoise(encoder, workgroups256);
        }

        } else if (justLocked && this.blueNoiseEnabled) {
            // Lock just engaged with blue noise on: bake it into noiseBuf
            // permanently (no restore) so the snapshot includes blue noise.
            this._encodeBlueNoise(encoder, workgroups256);
        }

        // --- Star warp (Stars display mode only) ---
        if (starsMode) {
            device.queue.writeBuffer(this.starUniformBuf, 0,
                new Uint32Array([W, H, frameSeed, numStars]));
            if (stereoActive) {
                // Independent RNG stream for the right eye's births.
                const seedR = (Math.imul(frameSeed, 2654435761) ^ 0x9e3779b9) >>> 0;
                device.queue.writeBuffer(this.starUniformBufR, 0,
                    new Uint32Array([W, H, seedR, numStars]));
                if (!this.noiseLocked) {
                    // drawIndirect args reset: [vertexCount, instanceCount, 0, 0]
                    device.queue.writeBuffer(this.mergeIndirectL, 0, new Uint32Array([0, 1, 0, 0]));
                    device.queue.writeBuffer(this.mergeIndirectR, 0, new Uint32Array([0, 1, 0, 0]));
                }
            }

            // Render uniform: tent radius (integer for exact partition-of-unity
            // brightness invariance) and hard-quad half-extent both scale with
            // resolution so stars stay visible at 2048².
            const tentRadius = Math.max(1, Math.round(W / 1024));
            const hardHalf = Math.max(0.5, W / 2048);
            const srBuf = new ArrayBuffer(48);
            const srU32 = new Uint32Array(srBuf);
            const srF32 = new Float32Array(srBuf);
            srU32[0] = W; srU32[1] = H;
            srU32[2] = stereoActive ? numStars * 2 : numStars;   // vs guard bound
            srU32[3] = this.starAAEnabled ? 1 : 0;
            srF32[4] = tentRadius; srF32[5] = hardHalf;
            srU32[6] = this.starEmojiEnabled ? 1 : 0;
            srF32[7] = Math.max(6, Math.round(W / 1024 * GLYPH_HALF_BASE));
            srU32[8] = this.starColorQEnabled ? 1 : 0;
            srU32[9] = this.starSizeQEnabled ? 1 : 0;
            // q-size slider is calibrated at 1024²: scale with resolution so a
            // "20px" star looks the same size at 512² or 2048².
            srF32[10] = this.starSizeMaxPx * (W / 1024);
            device.queue.writeBuffer(this.starRenderUniformBuf, 0, srBuf);

            // Lock [L] freezes the star field too: skip the dynamics, keep rendering.
            const encodeStarStep = (splatGroup, updateGroup) => {
                encoder.clearBuffer(this.starDensityBuf);
                const splatPass = encoder.beginComputePass();
                splatPass.setPipeline(this.starSplatPipeline);
                splatPass.setBindGroup(0, splatGroup);
                splatPass.dispatchWorkgroups(workgroups256);
                splatPass.end();

                const scanRowsPass = encoder.beginComputePass();
                scanRowsPass.setPipeline(this.starScanRowsPipeline);
                scanRowsPass.setBindGroup(0, this.starScanRowsBindGroup);
                scanRowsPass.dispatchWorkgroups(Math.ceil(H / 64));
                scanRowsPass.end();

                const scanCdfPass = encoder.beginComputePass();
                scanCdfPass.setPipeline(this.starScanCdfPipeline);
                scanCdfPass.setBindGroup(0, this.starScanCdfBindGroup);
                scanCdfPass.dispatchWorkgroups(1);
                scanCdfPass.end();

                const updatePass = encoder.beginComputePass();
                updatePass.setPipeline(this.starUpdatePipeline);
                updatePass.setBindGroup(0, updateGroup);
                updatePass.dispatchWorkgroups(Math.ceil(numStars / 256));
                updatePass.end();
            };
            // Stereo merge: transport the OTHER eye's uniform mass along the
            // cross flow (same splat kernel), then deterministically select
            // survivors from own + reprojected-other copies (report §13).
            const encodeMerge = (splatGroup, selectGroup) => {
                encoder.clearBuffer(this.starDensityBuf);
                const splatPass = encoder.beginComputePass();
                splatPass.setPipeline(this.starSplatPipeline);
                splatPass.setBindGroup(0, splatGroup);
                splatPass.dispatchWorkgroups(workgroups256);
                splatPass.end();

                const selectPass = encoder.beginComputePass();
                selectPass.setPipeline(this.mergeSelectPipeline);
                selectPass.setBindGroup(0, selectGroup);
                selectPass.dispatchWorkgroups(Math.ceil(2 * numStars / 256));
                selectPass.end();
            };

            if (!this.noiseLocked) {
                encodeStarStep(this.starSplatBindGroup, this.starUpdateBindGroup);
                if (stereoActive) {
                    encodeStarStep(this.starSplatBindGroupR, this.starUpdateBindGroupR);
                    encodeMerge(this.mergeSplatBindGroupL, this.mergeSelectBindGroupL);
                    encodeMerge(this.mergeSplatBindGroupR, this.mergeSelectBindGroupR);
                }
            }

            // Accumulate LINEAR-light star coverage (additive / premultiplied).
            const emojiIdx = this.starEmojiEnabled ? 1 : 0;
            const pipeline = this.starEmojiEnabled ? this.starRenderPipelineEmoji
                                                   : this.starRenderPipeline;
            const encodeStarRender = (texView, bindGroup, indirectBuf) => {
                const starPass = encoder.beginRenderPass({
                    colorAttachments: [{
                        view: texView,
                        loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0],
                    }],
                });
                starPass.setPipeline(pipeline);
                starPass.setBindGroup(0, bindGroup);
                if (indirectBuf) starPass.drawIndirect(indirectBuf, 0);
                else starPass.draw(numStars * 6);
                starPass.end();
            };
            if (stereoActive) {
                encodeStarRender(this.starTexView,  this.mergedRenderBG.L[emojiIdx], this.mergeIndirectL);
                encodeStarRender(this.starTexRView, this.mergedRenderBG.R[emojiIdx], this.mergeIndirectR);
            } else {
                encodeStarRender(this.starTexView,
                    this.starEmojiEnabled ? this.starRenderBindGroupEmoji : this.starRenderBindGroup,
                    null);
            }
        }

        // --- Display ---
        const canvasView = this.ctx.getCurrentTexture().createView();
        const dispPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: canvasView,
                loadOp: 'clear', storeOp: 'store',
                clearValue: [0.1, 0.1, 0.1, 1],
            }],
            ...(this.hasTimestamps && !this._tsMapping ? {
                timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 10, endOfPassWriteIndex: 11 },
            } : {}),
        });
        dispPass.setPipeline(this.displayPipeline);
        dispPass.setBindGroup(0, this.displayBindGroup);
        dispPass.setVertexBuffer(0, this.quadVB);
        dispPass.draw(this.quadVertCount);
        dispPass.end();

        // Restore noise after blue noise display (skip when locked — noiseBuf is
        // the snapshot; skip in stars mode — no backup was taken this frame, so
        // restoring would copy a STALE backup over the noise).
        if (!this.noiseLocked && this.blueNoiseEnabled && !starsMode) {
            encoder.copyBufferToBuffer(this.bnBackupBuf, 0, this.noiseBuf, 0, N * this.C * 4);
        }

        // Timestamps
        if (this.hasTimestamps && !this._tsMapping) {
            encoder.resolveQuerySet(this.querySet, 0, NUM_TIMESTAMPS, this.tsResolveBuf, 0);
            encoder.copyBufferToBuffer(this.tsResolveBuf, 0, this.tsReadBuf, 0, NUM_TIMESTAMPS * 8);
        }

        // Stats readback
        if (this.frameCount % 60 === 0 && !this._statsMapping) {
            encoder.copyBufferToBuffer(this.noiseBuf, 0, this._statsStagingBuf, 0, N * this.C * 4);
            this._statsNeedRead = true;
        }

        // Star stats readback (headless validation: uniformity + in-bounds)
        const starSample = Math.min(numStars, STAR_STATS_SAMPLE);
        if (starsMode && this.frameCount % 60 === 0 && !this._starStatsMapping) {
            encoder.copyBufferToBuffer(this.starBuf, 0, this._starStagingBuf, 0, starSample * 2 * 4);
            encoder.copyBufferToBuffer(this.starMetaBuf, 0,
                this._starStagingBuf, STAR_STATS_SAMPLE * 8, starSample * 2 * 4);
            encoder.copyBufferToBuffer(this.starCountersBuf, 0,
                this._starStagingBuf, STAR_STATS_SAMPLE * 16, 8);
            encoder.copyBufferToBuffer(this.mergeIndirectL, 0,
                this._starStagingBuf, STAR_STATS_SAMPLE * 16 + 8, 4);
            encoder.copyBufferToBuffer(this.mergeIndirectR, 0,
                this._starStagingBuf, STAR_STATS_SAMPLE * 16 + 12, 4);
            this._starStatsNeedRead = true;
        }

        device.queue.submit([encoder.finish()]);

        // Async timestamp readback
        if (this.hasTimestamps && !this._tsMapping) {
            this._tsMapping = true;
            this.tsReadBuf.mapAsync(GPUMapMode.READ).then(() => {
                const raw = new BigInt64Array(this.tsReadBuf.getMappedRange().slice(0));
                this.tsReadBuf.unmap();
                this._processTimestamps(raw);
                this._tsMapping = false;
            }).catch(() => { this._tsMapping = false; });
        }

        // Async stats readback
        if (this._statsNeedRead) {
            this._statsMapping = true;
            this._statsNeedRead = false;
            this._statsStagingBuf.mapAsync(GPUMapMode.READ).then(() => {
                const data = new Float32Array(this._statsStagingBuf.getMappedRange().slice(0));
                this._statsStagingBuf.unmap();
                this._computeNoiseStats(data);
                this._statsMapping = false;
            }).catch(() => { this._statsMapping = false; });
        }

        // Async star stats readback
        if (this._starStatsNeedRead) {
            this._starStatsMapping = true;
            this._starStatsNeedRead = false;
            this._starStagingBuf.mapAsync(GPUMapMode.READ).then(() => {
                const raw = this._starStagingBuf.getMappedRange().slice(0);
                this._starStagingBuf.unmap();
                const positions = new Float32Array(raw, 0, starSample * 2);
                const metaU32 = new Uint32Array(raw, STAR_STATS_SAMPLE * 8, starSample * 2);
                const counters = new Uint32Array(raw, STAR_STATS_SAMPLE * 16, 4);
                this._computeStarStats(positions, numStars, metaU32, counters);
                this._starStatsMapping = false;
            }).catch(() => { this._starStatsMapping = false; });
        }

        this.frameCount++;
    }

    /**
     * Uniformity + validity stats over a sample of star positions.
     * Coarse 4x4 grid occupancy: min/mean and max/mean should be ~1 for a
     * uniform field (the whole point of the star warp algorithm).
     * Not pure: publishes window.__starStats.
     *
     * Args:
     *   data (Float32Array): [sample*2] interleaved (x, y) pixel coords
     *   numStars (number): active star count (reported, not sampled)
     */
    _computeStarStats(data, numStars, metaU32 = null, counters = null) {
        const { W, H } = this;
        const GRID = 4;
        const cells = new Array(GRID * GRID).fill(0);
        const sample = data.length / 2;
        let inBounds = 0;
        for (let i = 0; i < sample; i++) {
            const x = data[i * 2], y = data[i * 2 + 1];
            if (x >= 0 && x <= W && y >= 0 && y <= H) {
                inBounds++;
                const gx = Math.min(GRID - 1, Math.floor(x / W * GRID));
                const gy = Math.min(GRID - 1, Math.floor(y / H * GRID));
                cells[gy * GRID + gx]++;
            }
        }
        const mean = inBounds / cells.length;
        this.starStats = {
            numStars,
            sample,
            inBoundsFrac: inBounds / sample,
            minOverMean: mean > 0 ? Math.min(...cells) / mean : 0,
            maxOverMean: mean > 0 ? Math.max(...cells) / mean : 0,
            deaths: counters ? counters[1] : null,
            // merged survivor counts per eye (drawIndirect vertexCount / 6);
            // in stereo each should hover at ~numStars (merged-view uniformity)
            mergedL: counters ? Math.floor(counters[2] / 6) : null,
            mergedR: counters ? Math.floor(counters[3] / 6) : null,
            stereoActive: this._stereoActive,
        };
        if (typeof window !== 'undefined') {
            window.__starStats = this.starStats;
            // ids of the sampled stars (odd u32 of each {q, id} pair): lets
            // headless tests measure identity persistence directly.
            if (metaU32) {
                const ids = new Array(sample);
                for (let i = 0; i < sample; i++) ids[i] = metaU32[i * 2 + 1];
                window.__starIds = ids;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Profiling
    // -----------------------------------------------------------------------

    _processTimestamps(data) {
        const ms = (begin, end) => Number(data[end] - data[begin]) / 1e6;
        const t = {
            scene:       ms(0, 1),
            buildDeform: ms(2, 3),
            backwardMap: ms(4, 5),
            brownian:    ms(6, 7),
            normalize:   ms(8, 9),
            display:     ms(10, 11),
        };
        t.total = t.scene + t.buildDeform + t.backwardMap + t.brownian + t.normalize + t.display;
        this._gpuTimings = t;
        this._gpuTimingHistory.push(t);
        if (this._gpuTimingHistory.length > 200) this._gpuTimingHistory.shift();

        if (typeof window !== 'undefined') {
            window.__gpuTimingSample = t;
            if (!window.__gpuTimingHistory) window.__gpuTimingHistory = [];
            window.__gpuTimingHistory.push(t);
        }
    }

    getTimingStats() {
        const h = this._gpuTimingHistory;
        if (h.length === 0) return null;
        const phases = Object.keys(h[0]);
        const result = {};
        for (const p of phases) {
            const vals = h.map(t => t[p]);
            const n = vals.length;
            const mean = vals.reduce((a, b) => a + b, 0) / n;
            const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
            result[p] = { mean, std, n };
        }
        return result;
    }

    /**
     * Change the shadow map resolution at runtime. Recreates the shadow texture
     * and rebinds affected bind groups. Not pure: destroys/creates GPU resources.
     *
     * Args:
     *   newRes (number): New shadow map resolution (e.g., 1024, 2048, 4096, 8192)
     *
     * Examples:
     *   >>> // renderer.setShadowResolution(8192)
     */
    setShadowResolution(newRes) {
        if (newRes === this.shadowResolution) return;
        this.shadowResolution = newRes;

        this.shadowTex.destroy();
        this.shadowTex = this.device.createTexture({
            size: [newRes, newRes], format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.shadowTexView = this.shadowTex.createView({ aspect: 'depth-only' });

        const buf = (b) => ({ buffer: b });
        this.sceneBindGroup = this.device.createBindGroup({
            layout: this.scenePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.cameraUniformBuf) },
                { binding: 1, resource: buf(this.instanceBuf) },
                { binding: 2, resource: this.shadowTexView },
                { binding: 3, resource: this.shadowSampler },
                { binding: 4, resource: buf(this.lightUniformBuf) },
            ],
        });
    }

    _computeNoiseStats(data) {
        let sum = 0, sum2 = 0;
        for (let i = 0; i < data.length; i++) {
            sum += data[i];
            sum2 += data[i] * data[i];
        }
        const mean = sum / data.length;
        const std = Math.sqrt(sum2 / data.length - mean * mean);
        this.noiseStats = { mean, std };
        if (typeof window !== 'undefined') window.__noiseStats = this.noiseStats;
    }
}
