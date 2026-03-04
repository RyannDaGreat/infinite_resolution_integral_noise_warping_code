/**
 * WebGPU renderer: device, pipelines, storage buffers, compute dispatch, profiling.
 * Zero CPU-GPU copies per frame — entire particle warp runs in GPU compute shaders.
 */

import { cubeVertices, floorVertices, quadVertices } from './geometry.js';
import {
    sceneWGSL, displayWGSL,
    buildDeformWGSL, backwardMapWGSL, brownianWGSL, normalizeWGSL,
} from './shaders.js';

const NUM_TIMESTAMPS = 12; // 2 per pass × 6 passes

// ---------------------------------------------------------------------------
// Mulberry32 + Box-Muller PRNG (for CPU-side init only)
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
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {number} W - width
     * @param {number} H - height
     */
    constructor(canvas, W, H) {
        this.canvas = canvas;
        this.W = W;
        this.H = H;
        this.N = W * H;
        this.C = 4;
        this.frameCount = 0;

        // Profiling state
        this.hasTimestamps = false;
        this._tsMapping = false;
        this._gpuTimings = null;   // latest per-phase GPU timings (ms)
        this._gpuTimingHistory = []; // rolling window for stats

        // Stats readback state
        this._statsMapping = false;
        this.noiseStats = { mean: 0, std: 1 };
    }

    async init() {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('WebGPU: no adapter found');

        this.hasTimestamps = adapter.features.has('timestamp-query');
        const features = this.hasTimestamps ? ['timestamp-query'] : [];
        this.device = await adapter.requestDevice({ requiredFeatures: features });
        this.device.lost.then(info => { throw new Error('WebGPU device lost: ' + info.message); });

        // Canvas context
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
        this.noiseDispTex = device.createTexture({
            size: [W, H], format: 'rgba32float',
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
        });

        this.colorTexView  = this.colorTex.createView();
        this.motionTexView = this.motionTex.createView();
        this.depthTexView  = this.depthTex.createView();
        this.noiseDispTexView = this.noiseDispTex.createView();
    }

    _createBuffers() {
        const { device, N, C } = this;
        const MAX_TICKETS = 24;
        const f4 = 4; // sizeof(f32)

        const storage = (size, extra = 0) => device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | extra,
        });

        // Core noise state
        this.noiseBuf = storage(N * C * f4, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

        // Intermediate compute buffers (cleared each frame)
        this.bufferBuf     = storage(N * C * f4, GPUBufferUsage.COPY_DST);
        this.pixelAreaBuf  = storage(N * f4, GPUBufferUsage.COPY_DST);
        this.ticketCountBuf = storage(N * f4, GPUBufferUsage.COPY_DST);

        // Ticket data (not cleared — only valid entries read)
        this.masterFieldBuf = storage(N * MAX_TICKETS * f4);
        this.areaFieldBuf   = storage(N * MAX_TICKETS * f4);

        // Deformation map
        this.deformationBuf = storage(N * 2 * f4);

        // Uniform buffers (separate for floor vs cube — can't update mid-pass)
        this.floorUniformBuf = device.createBuffer({
            size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.cubeUniformBuf = device.createBuffer({
            size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.computeUniformBuf = device.createBuffer({
            size: 16, // { H, W, frameSeed, _pad }
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.displayUniformBuf = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Stats readback staging buffer
        this._statsStagingBuf = device.createBuffer({
            size: N * C * f4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
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

        // Scene render pipeline (MRT: color + motion)
        this.scenePipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: sceneModule, entryPoint: 'vs',
                buffers: [{
                    arrayStride: 9 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0,  format: 'float32x3' },
                        { shaderLocation: 1, offset: 12, format: 'float32x3' },
                        { shaderLocation: 2, offset: 24, format: 'float32x3' },
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

        // Display render pipeline
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

        // Compute pipelines
        const computePipeline = (module) => device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' },
        });

        this.buildDeformPipeline = computePipeline(buildDeformMod);
        this.backwardMapPipeline = computePipeline(backwardMapMod);
        this.brownianPipeline    = computePipeline(brownianMod);
        this.normalizePipeline   = computePipeline(normalizeMod);
    }

    _createBindGroups() {
        const { device } = this;
        const buf = (b) => ({ buffer: b });

        // Scene bind groups (separate for floor and cube)
        this.floorBindGroup = device.createBindGroup({
            layout: this.scenePipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: buf(this.floorUniformBuf) }],
        });
        this.cubeBindGroup = device.createBindGroup({
            layout: this.scenePipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: buf(this.cubeUniformBuf) }],
        });

        // Display bind group
        this.displayBindGroup = device.createBindGroup({
            layout: this.displayPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.noiseDispTexView },
                { binding: 1, resource: this.colorTexView },
                { binding: 2, resource: this.motionTexView },
                { binding: 3, resource: buf(this.displayUniformBuf) },
            ],
        });

        // Build deformation bind group
        this.buildDeformBindGroup = device.createBindGroup({
            layout: this.buildDeformPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.computeUniformBuf) },
                { binding: 1, resource: this.motionTexView },
                { binding: 2, resource: buf(this.deformationBuf) },
            ],
        });

        // Backward map bind group
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

        // Brownian bridge bind group
        this.brownianBindGroup = device.createBindGroup({
            layout: this.brownianPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.computeUniformBuf) },
                { binding: 1, resource: buf(this.ticketCountBuf) },
                { binding: 2, resource: buf(this.masterFieldBuf) },
                { binding: 3, resource: buf(this.areaFieldBuf) },
                { binding: 4, resource: buf(this.noiseBuf) },
                { binding: 5, resource: buf(this.bufferBuf) },
                { binding: 6, resource: buf(this.pixelAreaBuf) },
            ],
        });

        // Normalize bind group
        this.normalizeBindGroup = device.createBindGroup({
            layout: this.normalizePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: buf(this.computeUniformBuf) },
                { binding: 1, resource: buf(this.bufferBuf) },
                { binding: 2, resource: buf(this.pixelAreaBuf) },
                { binding: 3, resource: buf(this.noiseBuf) },
            ],
        });
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

        const cubeData = cubeVertices();
        const floorData = floorVertices();
        const quadData = quadVertices();

        this.cubeVB = uploadVB(cubeData);
        this.cubeVertCount = cubeData.length / 9;
        this.floorVB = uploadVB(floorData);
        this.floorVertCount = floorData.length / 9;
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

        // Also copy to display texture for first frame
        device.queue.writeTexture(
            { texture: this.noiseDispTex },
            data,
            { bytesPerRow: W * C * 4, rowsPerImage: H },
            { width: W, height: H },
        );
    }

    // -----------------------------------------------------------------------
    // Per-frame
    // -----------------------------------------------------------------------

    /**
     * Render one complete frame. All GPU work in a single command buffer.
     * Not pure: mutates GPU state and profiling accumulators.
     *
     * @param {object} opts
     * @param {Float32Array} opts.model - 4x4 model matrix
     * @param {Float32Array} opts.viewProj - 4x4 view-projection matrix
     * @param {Float32Array} opts.prevModel - previous frame's model
     * @param {Float32Array} opts.prevViewProj - previous frame's viewProj
     * @param {number} opts.displayMode - 0-4
     * @param {number} opts.frameSeed - incrementing seed
     */
    frame({ model, viewProj, prevModel, prevViewProj, displayMode, frameSeed }) {
        const { device, W, H, N } = this;
        const workgroups = Math.ceil(N / 256);

        // Update uniforms — floor uses identity model
        const IDENTITY = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
        const floorData = new Float32Array(64);
        floorData.set(IDENTITY, 0);     // model = identity
        floorData.set(viewProj, 16);
        floorData.set(IDENTITY, 32);    // prevModel = identity
        floorData.set(prevViewProj, 48);
        device.queue.writeBuffer(this.floorUniformBuf, 0, floorData);

        const cubeData = new Float32Array(64);
        cubeData.set(model, 0);
        cubeData.set(viewProj, 16);
        cubeData.set(prevModel, 32);
        cubeData.set(prevViewProj, 48);
        device.queue.writeBuffer(this.cubeUniformBuf, 0, cubeData);

        device.queue.writeBuffer(this.computeUniformBuf, 0,
            new Uint32Array([H, W, frameSeed, 0]));
        device.queue.writeBuffer(this.displayUniformBuf, 0,
            new Uint32Array([displayMode]));

        const encoder = device.createCommandEncoder();

        // --- Pass 0: Scene render (MRT) ---
        const scenePass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: this.colorTexView,
                    loadOp: 'clear', storeOp: 'store',
                    clearValue: [0, 0, 0, 1],
                },
                {
                    view: this.motionTexView,
                    loadOp: 'clear', storeOp: 'store',
                    clearValue: [0, 0, 0, 0],
                },
            ],
            depthStencilAttachment: {
                view: this.depthTexView,
                depthLoadOp: 'clear', depthStoreOp: 'store',
                depthClearValue: 1.0,
            },
            ...(this.hasTimestamps && !this._tsMapping ? {
                timestampWrites: {
                    querySet: this.querySet,
                    beginningOfPassWriteIndex: 0,
                    endOfPassWriteIndex: 1,
                },
            } : {}),
        });
        scenePass.setPipeline(this.scenePipeline);

        // Floor (identity model → zero object motion, only camera motion)
        scenePass.setBindGroup(0, this.floorBindGroup);
        scenePass.setVertexBuffer(0, this.floorVB);
        scenePass.draw(this.floorVertCount);

        // Cube (rotating model matrix)
        scenePass.setBindGroup(0, this.cubeBindGroup);
        scenePass.setVertexBuffer(0, this.cubeVB);
        scenePass.draw(this.cubeVertCount);

        scenePass.end();

        // --- Pass 1: Build deformation (compute) ---
        const deformPass = encoder.beginComputePass(
            this.hasTimestamps && !this._tsMapping ? {
                timestampWrites: {
                    querySet: this.querySet,
                    beginningOfPassWriteIndex: 2,
                    endOfPassWriteIndex: 3,
                },
            } : undefined,
        );
        deformPass.setPipeline(this.buildDeformPipeline);
        deformPass.setBindGroup(0, this.buildDeformBindGroup);
        deformPass.dispatchWorkgroups(workgroups);
        deformPass.end();

        // --- Clear intermediate buffers ---
        encoder.clearBuffer(this.bufferBuf);
        encoder.clearBuffer(this.pixelAreaBuf);
        encoder.clearBuffer(this.ticketCountBuf);

        // --- Pass 2: Backward map (compute) ---
        const bwdPass = encoder.beginComputePass(
            this.hasTimestamps && !this._tsMapping ? {
                timestampWrites: {
                    querySet: this.querySet,
                    beginningOfPassWriteIndex: 4,
                    endOfPassWriteIndex: 5,
                },
            } : undefined,
        );
        bwdPass.setPipeline(this.backwardMapPipeline);
        bwdPass.setBindGroup(0, this.backwardMapBindGroup);
        bwdPass.dispatchWorkgroups(workgroups);
        bwdPass.end();

        // --- Pass 3: Brownian bridge (compute) ---
        const brPass = encoder.beginComputePass(
            this.hasTimestamps && !this._tsMapping ? {
                timestampWrites: {
                    querySet: this.querySet,
                    beginningOfPassWriteIndex: 6,
                    endOfPassWriteIndex: 7,
                },
            } : undefined,
        );
        brPass.setPipeline(this.brownianPipeline);
        brPass.setBindGroup(0, this.brownianBindGroup);
        brPass.dispatchWorkgroups(workgroups);
        brPass.end();

        // --- Pass 4: Normalize (compute) ---
        const normPass = encoder.beginComputePass(
            this.hasTimestamps && !this._tsMapping ? {
                timestampWrites: {
                    querySet: this.querySet,
                    beginningOfPassWriteIndex: 8,
                    endOfPassWriteIndex: 9,
                },
            } : undefined,
        );
        normPass.setPipeline(this.normalizePipeline);
        normPass.setBindGroup(0, this.normalizeBindGroup);
        normPass.dispatchWorkgroups(workgroups);
        normPass.end();

        // --- Copy noise buffer → display texture ---
        encoder.copyBufferToTexture(
            { buffer: this.noiseBuf, bytesPerRow: W * this.C * 4, rowsPerImage: H },
            { texture: this.noiseDispTex },
            { width: W, height: H },
        );

        // --- Pass 5: Display render ---
        const canvasView = this.ctx.getCurrentTexture().createView();
        const dispPass = encoder.beginRenderPass({
            colorAttachments: [{
                view: canvasView,
                loadOp: 'clear', storeOp: 'store',
                clearValue: [0.1, 0.1, 0.1, 1],
            }],
            ...(this.hasTimestamps && !this._tsMapping ? {
                timestampWrites: {
                    querySet: this.querySet,
                    beginningOfPassWriteIndex: 10,
                    endOfPassWriteIndex: 11,
                },
            } : {}),
        });
        dispPass.setPipeline(this.displayPipeline);
        dispPass.setBindGroup(0, this.displayBindGroup);
        dispPass.setVertexBuffer(0, this.quadVB);
        dispPass.draw(this.quadVertCount);
        dispPass.end();

        // --- Resolve timestamps ---
        if (this.hasTimestamps && !this._tsMapping) {
            encoder.resolveQuerySet(this.querySet, 0, NUM_TIMESTAMPS, this.tsResolveBuf, 0);
            encoder.copyBufferToBuffer(this.tsResolveBuf, 0, this.tsReadBuf, 0, NUM_TIMESTAMPS * 8);
        }

        // --- Stats readback (every 60 frames) ---
        if (this.frameCount % 60 === 0 && !this._statsMapping) {
            encoder.copyBufferToBuffer(this.noiseBuf, 0, this._statsStagingBuf, 0, N * this.C * 4);
            this._statsNeedRead = true;
        }

        // Submit
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
    // Profiling internals
    // -----------------------------------------------------------------------

    _processTimestamps(data) {
        // data is BigInt64Array[12] — pairs of begin/end nanoseconds
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

        // Expose for test suite
        if (typeof window !== 'undefined') {
            window.__gpuTimingSample = t;
            if (!window.__gpuTimingHistory) window.__gpuTimingHistory = [];
            window.__gpuTimingHistory.push(t);
        }
    }

    /**
     * Get rolling statistics for GPU timings over the last N samples.
     * @returns {object|null} { phase: { mean, std, n } } or null if no data
     */
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

        if (typeof window !== 'undefined') {
            window.__noiseStats = this.noiseStats;
        }
    }

}
