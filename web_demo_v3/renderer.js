/**
 * WebGPU renderer: instanced MRT scene render, warp compute pipeline, blue noise, display.
 * Zero CPU-GPU copies per frame for the warp — only the instance buffer is uploaded.
 */

import { boxVertices, beveledBoxVertices, sphereVertices, quadVertices, terrainMeshVertices } from './geometry.js';
import {
    sceneWGSL, skyWGSL, shadowWGSL, displayWGSL,
    buildDeformWGSL, backwardMapWGSL, brownianWGSL, normalizeWGSL,
    blueNoiseBlurWGSL,
} from './shaders.js';
import { MAX_INSTANCES, FLOATS_PER_INSTANCE, TERRAIN_INSTANCE_IDX } from './scene.js';

const NUM_TIMESTAMPS = 12;

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
        this.shadowsEnabled = true;
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
    }

    destroy() {
        this.colorTex?.destroy();
        this.motionTex?.destroy();
        this.depthTex?.destroy();
        this.shadowTex?.destroy();

        const bufs = [
            this.noiseBuf, this.bufferBuf, this.totalRequestBuf, this.ticketCountBuf,
            this.masterFieldBuf, this.areaFieldBuf, this.deformationBuf,
            this.cameraUniformBuf, this.skyUniformBuf, this.shadowUniformBuf, this.lightUniformBuf, this.instanceBuf,
            this.computeUniformBuf, this.displayUniformBuf,
            this._statsStagingBuf,
            this.bnBackupBuf,
            this.bnBlurHUniformBuf,
            ...(this.bnBlurVUniformBufs || []),
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
        this.depthTex = device.createTexture({
            size: [W, H], format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        // Shadow map: 4096×4096 depth texture, sampled in scene shader for PCF.
        this.shadowTex = device.createTexture({
            size: [4096, 4096], format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.colorTexView  = this.colorTex.createView();
        this.motionTexView = this.motionTex.createView();
        this.depthTexView  = this.depthTex.createView();
        this.shadowTexView = this.shadowTex.createView({ aspect: 'depth-only' });
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
        //                 + eyePos (16) + eyeDir (16) = 240 bytes
        this.cameraUniformBuf = device.createBuffer({
            size: 240, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
            size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._statsStagingBuf = device.createBuffer({
            size: N * C * f4,
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
        this.sceneBindGroup = device.createBindGroup({
            layout: this.scenePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.cameraUniformBuf) },
                { binding: 1, resource: buf(this.instanceBuf) },
                { binding: 2, resource: this.shadowTexView },
                { binding: 3, resource: this.shadowSampler },
                { binding: 4, resource: buf(this.lightUniformBuf) },
            ],
        });

        // Display bind group
        this.displayBindGroup = device.createBindGroup({
            layout: this.displayPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.noiseBuf) },
                { binding: 1, resource: this.colorTexView },
                { binding: 2, resource: this.motionTexView },
                { binding: 3, resource: buf(this.displayUniformBuf) },
            ],
        });

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
        const terrainData = terrainMeshVertices(2400, 4500);

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
    frame({ viewProj, prevViewProj, invViewProj, instanceData, numBoxInstances, numSphereInstances, numDominoInstances, displayMode, frameSeed, elapsedSecs = 0, eyePos = [0,0,0], eyeDir = [0,0,-1], lights = [] }) {
        const { device, W, H, N } = this;
        const workgroups256 = Math.ceil(N / 256);
        const brownianWGs = Math.ceil(N / this.brownianWG);

        // Day/night cycle: one full cycle every 5 minutes (300 seconds).
        // angle=0 → sun at horizon rising; angle=π/2 → solar noon; angle=π → setting; angle>π → night.
        const DAY_CYCLE_SECS = 300;
        // Start at solar noon (π/2 offset so sun is overhead at t=0)
        const angle = (2 * Math.PI * elapsedSecs) / DAY_CYCLE_SECS + Math.PI / 2;
        // sunDir: x=cos(angle) sweeps east→west, y=sin(angle) rises/sets, z=slight tilt north
        const rawX = Math.cos(angle);
        const rawY = Math.sin(angle);
        const rawZ = 0.3;
        const len = Math.sqrt(rawX * rawX + rawY * rawY + rawZ * rawZ);
        const sunDir = [rawX / len, rawY / len, rawZ / len, 0.0];

        // Orthographic light-space matrix for directional shadow map.
        // Covers a 200×200 world-unit area centred at origin, depth range 0..300.
        const lightSpaceMatrix = buildLightSpaceMatrix(sunDir, 200, 300);

        // Upload camera uniforms: viewProj (64) + prevViewProj (64) + sunDir (16) + lightSpaceMatrix (64)
        //                         + eyePos (16) + eyeDir (16) = 240 bytes  (60 floats)
        const camData = new Float32Array(60);
        camData.set(viewProj, 0);
        camData.set(prevViewProj, 16);
        camData.set(sunDir, 32);
        camData.set(lightSpaceMatrix, 36);
        camData.set([eyePos[0], eyePos[1], eyePos[2], 0.0], 52);  // eyePos (vec4f at float index 52)
        camData.set([eyeDir[0], eyeDir[1], eyeDir[2], 0.0], 56);  // eyeDir (vec4f at float index 56)
        device.queue.writeBuffer(this.cameraUniformBuf, 0, camData);

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
        const numLights = Math.min(lights.length, 32);
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
        const dispBuf = new ArrayBuffer(32);
        const dispU32 = new Uint32Array(dispBuf);
        const dispF32 = new Float32Array(dispBuf);
        dispU32[0] = displayMode;
        dispU32[1] = W;
        dispU32[2] = H;
        dispU32[3] = displayFlags;
        dispU32[4] = this.thresholdOn || 0;
        dispF32[5] = this.thresholdValue || 0;
        dispF32[6] = this.noiseOpacity;
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
            shadowPass.setVertexBuffer(0, this.terrainVB);
            shadowPass.draw(this.terrainVertCount, 1, 0, TERRAIN_INSTANCE_IDX);

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

        // --- Scene render (sky + instanced MRT) ---
        const scenePass = encoder.beginRenderPass({
            colorAttachments: [
                { view: this.colorTexView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
                { view: this.motionTexView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
            ],
            depthStencilAttachment: {
                view: this.depthTexView,
                depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0,
            },
            ...(this.hasTimestamps && !this._tsMapping ? {
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
        scenePass.setBindGroup(0, this.sceneBindGroup);

        // Draw floor (flat box): instance 0
        scenePass.setVertexBuffer(0, this.boxVB);
        scenePass.draw(this.boxVertCount, 1, 0, 0);

        // Draw dominoes (beveled box): instances 1..1+numDominoInstances
        if (numDominoes > 0) {
            scenePass.setVertexBuffer(0, this.bevelBoxVB);
            scenePass.draw(this.bevelBoxVertCount, numDominoes, 0, 1);
        }

        // Draw remaining boxes (maze, tower, marble machine, etc.): instances after dominoes up to numBoxInstances
        if (numMazeBoxes > 0) {
            scenePass.setVertexBuffer(0, this.boxVB);
            scenePass.draw(this.boxVertCount, numMazeBoxes, 0, mazeStartIdx);
        }

        // Draw spheres: instances numBoxInstances..numBoxInstances+numSphereInstances-1
        if (numSphereInstances > 0) {
            scenePass.setVertexBuffer(0, this.sphereVB);
            scenePass.draw(this.sphereVertCount, numSphereInstances, 0, numBoxInstances);
        }

        // Draw terrain mesh: single non-instanced draw at TERRAIN_INSTANCE_IDX
        scenePass.setVertexBuffer(0, this.terrainVB);
        scenePass.draw(this.terrainVertCount, 1, 0, TERRAIN_INSTANCE_IDX);

        scenePass.end();

        // --- Warp pipeline ---
        // Track lock transition: on the frame lock engages, bake blue noise
        // into noiseBuf so the snapshot includes it.
        const justLocked = this.noiseLocked && !this._wasLocked;
        this._wasLocked = this.noiseLocked;

        if (!this.noiseLocked) {

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

        // Restore noise after blue noise display (skip when locked — noiseBuf is the snapshot)
        if (!this.noiseLocked && this.blueNoiseEnabled) {
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

        this.frameCount++;
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
