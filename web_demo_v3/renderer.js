/**
 * WebGPU renderer: instanced MRT scene render, warp compute pipeline, blue noise, display.
 * Zero CPU-GPU copies per frame for the warp — only the instance buffer is uploaded.
 */

import { boxVertices, beveledBoxVertices, sphereVertices, quadVertices } from './geometry.js';
import {
    sceneWGSL, displayWGSL,
    buildDeformWGSL, backwardMapWGSL, brownianWGSL, normalizeWGSL,
    blueNoiseBlurWGSL,
} from './shaders.js';
import { MAX_INSTANCES, FLOATS_PER_INSTANCE } from './scene.js';

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
        this.blueNoiseIterations = 10;
        this.blueNoiseCutoffDivider = 8.0;

        this.greyscaleEnabled = false;
        this.uniformDisplayEnabled = false;
        this.noiseOpacity = 0.25;
        this.noiseLocked = false;
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

        const bufs = [
            this.noiseBuf, this.bufferBuf, this.totalRequestBuf, this.ticketCountBuf,
            this.masterFieldBuf, this.areaFieldBuf, this.deformationBuf,
            this.cameraUniformBuf, this.instanceBuf,
            this.computeUniformBuf, this.displayUniformBuf,
            this._statsStagingBuf,
            this.bnBackupBuf,
            this.bnBlurHUniformBuf,
            ...(this.bnBlurVUniformBufs || []),
            this.boxVB, this.sphereVB, this.quadVB,
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
        this.colorTexView  = this.colorTex.createView();
        this.motionTexView = this.motionTex.createView();
        this.depthTexView  = this.depthTex.createView();
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

        // Camera uniform: viewProj (64) + prevViewProj (64) = 128 bytes
        this.cameraUniformBuf = device.createBuffer({
            size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
        const sceneModule     = mod(sceneWGSL);
        const displayModule   = mod(displayWGSL);
        const buildDeformMod  = mod(buildDeformWGSL);
        const backwardMapMod  = mod(backwardMapWGSL);
        const brownianMod     = mod(brownianWGSL);
        const normalizeMod    = mod(normalizeWGSL);

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

        // Scene bind group: camera uniform + instance storage
        this.sceneBindGroup = device.createBindGroup({
            layout: this.scenePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.cameraUniformBuf) },
                { binding: 1, resource: buf(this.instanceBuf) },
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
        const bevelData = beveledBoxVertices(0.04, [0.75, 1.5, 0.18]);
        const sphereData = sphereVertices();
        const quadData = quadVertices();

        this.boxVB = uploadVB(boxData);
        this.boxVertCount = boxData.length / 6;
        this.bevelBoxVB = uploadVB(bevelData);
        this.bevelBoxVertCount = bevelData.length / 6;
        this.sphereVB = uploadVB(sphereData);
        this.sphereVertCount = sphereData.length / 6;
        this.quadVB = uploadVB(quadData);
        this.quadVertCount = quadData.length / 4;
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
     * @param {Float32Array} opts.instanceData - active instance data
     * @param {number} opts.numBoxInstances - box instance count
     * @param {number} opts.numSphereInstances - sphere instance count
     * @param {number} opts.displayMode - 0-4
     * @param {number} opts.frameSeed - incrementing seed
     */
    frame({ viewProj, prevViewProj, instanceData, numBoxInstances, numSphereInstances, displayMode, frameSeed }) {
        const { device, W, H, N } = this;
        const workgroups256 = Math.ceil(N / 256);
        const brownianWGs = Math.ceil(N / this.brownianWG);

        // Upload camera uniforms
        const camData = new Float32Array(32);
        camData.set(viewProj, 0);
        camData.set(prevViewProj, 16);
        device.queue.writeBuffer(this.cameraUniformBuf, 0, camData);

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

        // --- Scene render (instanced MRT) ---
        const scenePass = encoder.beginRenderPass({
            colorAttachments: [
                { view: this.colorTexView, loadOp: 'clear', storeOp: 'store', clearValue: [0.05, 0.05, 0.08, 1] },
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
        scenePass.setPipeline(this.scenePipeline);
        scenePass.setBindGroup(0, this.sceneBindGroup);

        // Draw floor (flat box): instance 0
        scenePass.setVertexBuffer(0, this.boxVB);
        scenePass.draw(this.boxVertCount, 1, 0, 0);

        // Draw dominoes (beveled box): instances 1..numBoxInstances-1
        if (numBoxInstances > 1) {
            scenePass.setVertexBuffer(0, this.bevelBoxVB);
            scenePass.draw(this.bevelBoxVertCount, numBoxInstances - 1, 0, 1);
        }

        // Draw spheres: instances numBoxInstances..numBoxInstances+numSphereInstances-1
        if (numSphereInstances > 0) {
            scenePass.setVertexBuffer(0, this.sphereVB);
            scenePass.draw(this.sphereVertCount, numSphereInstances, 0, numBoxInstances);
        }

        scenePass.end();

        // --- Warp pipeline (skipped when noise is locked) ---
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

        } // end noiseLocked check

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

        // Restore noise after blue noise display
        if (this.blueNoiseEnabled) {
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
