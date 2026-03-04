/**
 * JS port of the 4-phase particle warp kernel from inf_int_noise_warp.py.
 *
 * Algorithm 3 (particle-based GWTF) from:
 *   Deng et al., "Infinite-Resolution Integral Noise Warping", ICLR 2025
 *
 * Warps Gaussian noise through a deformation map while preserving spatial
 * uncorrelation via Brownian bridge stochastic sampling.
 */

const EPS = 1e-6;
const MAX_TICKETS = 24;

// ---------------------------------------------------------------------------
// Mulberry32 + Box-Muller PRNG (deterministic, seedable)
// ---------------------------------------------------------------------------

/**
 * Pure. Mulberry32 PRNG — returns next uint32 and updated state.
 * @param {number} state - uint32 state
 * @returns {[number, number]} [random uint32, next state]
 */
function mulberry32(state) {
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) | 0;
    return [(t ^ (t >>> 14)) >>> 0, state];
}

/**
 * Creates a seeded PRNG that returns uniform floats in (0, 1).
 * Not pure: returns a stateful closure.
 * @param {number} seed
 * @returns {function(): number}
 */
function makeRng(seed) {
    let state = seed | 0;
    return function() {
        let val;
        [val, state] = mulberry32(state);
        return (val >>> 0) / 4294967296;
    };
}

/**
 * Creates a seeded PRNG that returns standard normal samples (Box-Muller).
 * Not pure: returns a stateful closure.
 * @param {number} seed
 * @returns {function(): number}
 */
function makeRandn(seed) {
    const rng = makeRng(seed);
    let spare = null;
    return function() {
        if (spare !== null) {
            const s = spare;
            spare = null;
            return s;
        }
        let u1 = rng();
        while (u1 < 1e-10) u1 = rng();
        const u2 = rng();
        const r = Math.sqrt(-2 * Math.log(u1));
        const theta = 6.283185307179586 * u2;
        spare = r * Math.sin(theta);
        return r * Math.cos(theta);
    };
}

// ---------------------------------------------------------------------------
// ParticleWarper
// ---------------------------------------------------------------------------

export class ParticleWarper {
    /**
     * @param {number} H - image height
     * @param {number} W - image width
     * @param {number} C - number of noise channels
     */
    constructor(H, W, C) {
        this.H = H;
        this.W = W;
        this.C = C;

        // Noise state: [H * W * C] flat, row-major, channel-last
        this.noise = new Float32Array(H * W * C);

        // Pre-allocated working buffers
        this.buffer       = new Float32Array(H * W * C);  // scatter accumulator
        this.pixelArea    = new Float32Array(H * W);       // total area per dest pixel
        this.ticketCount  = new Int32Array(H * W);         // ticket counter per source pixel
        this.masterField  = new Int32Array(H * W * MAX_TICKETS);  // raveled dest index per ticket
        this.areaField    = new Float32Array(H * W * MAX_TICKETS); // bilinear weight per ticket

        // Identity deformation map (cell centers): [H * W * 2] (row, col)
        this.identityCC = new Float32Array(H * W * 2);
        for (let i = 0; i < H; i++) {
            for (let j = 0; j < W; j++) {
                const idx = (i * W + j) * 2;
                this.identityCC[idx]     = i + 0.5;  // row
                this.identityCC[idx + 1] = j + 0.5;  // col
            }
        }

        // Deformation map: [H * W * 2] (row, col) — written each frame
        this.deformation = new Float32Array(H * W * 2);

        // PRNG seed, incremented each frame
        this._frameSeed = 42;

        // Initialize noise with randn
        const randn = makeRandn(12345);
        for (let i = 0; i < this.noise.length; i++) {
            this.noise[i] = randn();
        }
    }

    /**
     * Get initial noise flipped for OpenGL upload (row 0 = bottom).
     * @returns {Float32Array} [H * W * C] with rows flipped
     */
    getInitNoiseForGPU() {
        return this._flipRows(this.noise);
    }

    /**
     * Run one warp step using GPU motion vectors.
     *
     * Not pure: mutates internal noise state.
     *
     * @param {Float32Array} motionGPU - [H * W * 2] motion vectors from GPU
     *   in (mv_x, mv_y) UV space, row 0 = bottom (OpenGL order).
     * @returns {Float32Array} [H * W * C] warped noise, flipped for GPU upload
     */
    step(motionGPU) {
        const { H, W } = this;

        // Build deformation map: identity_cc - flow_rc
        // GPU motion: (mv_x, mv_y) UV, row 0 = bottom, Y-up
        // We need: (row, col) pixel space, row 0 = top
        for (let i = 0; i < H; i++) {
            const gpuRow = H - 1 - i;  // flip Y
            for (let j = 0; j < W; j++) {
                const mvIdx = (gpuRow * W + j) * 2;
                const mvX = motionGPU[mvIdx];      // horizontal displacement
                const mvY = motionGPU[mvIdx + 1];  // vertical displacement (up = positive)
                const dIdx = (i * W + j) * 2;
                // deform_row = identity_row + mv_y * H  (Y-up → row-down cancels)
                this.deformation[dIdx]     = this.identityCC[dIdx]     + mvY * H;
                // deform_col = identity_col - mv_x * W
                this.deformation[dIdx + 1] = this.identityCC[dIdx + 1] - mvX * W;
            }
        }

        this._runKernel();

        this._frameSeed += 1;
        return this._flipRows(this.noise);
    }

    /**
     * 4-phase particle warp kernel. Direct port of _particle_warp_kernel.
     * Not pure: mutates buffer, pixelArea, ticketCount, masterField, areaField, noise.
     */
    _runKernel() {
        const { H, W, C, deformation, noise, buffer, pixelArea,
                ticketCount, masterField, areaField } = this;
        const N = H * W;

        // Phase 1: Clear
        buffer.fill(0);
        pixelArea.fill(0);
        ticketCount.fill(0);
        // areaField doesn't need clearing — we only read up to ticketCount entries

        // Phase 2: Backward map — distribute weighted requests via ticket counters
        for (let idx = 0; idx < N; idx++) {
            const dIdx = idx * 2;
            // warped_pos = deformation - 0.5  (map_field[i,j] - 0.5 in taichi)
            const wpRow = deformation[dIdx]     - 0.5;
            const wpCol = deformation[dIdx + 1] - 0.5;
            const lRow = Math.floor(wpRow);
            const lCol = Math.floor(wpCol);
            const fRow = wpRow - lRow;
            const fCol = wpCol - lCol;
            const uRow = lRow + 1;
            const uCol = lCol + 1;

            // 4 bilinear corners
            this._addTicket(lRow, lCol, idx, (1 - fRow) * (1 - fCol));
            this._addTicket(uRow, uCol, idx, fRow * fCol);
            this._addTicket(lRow, uCol, idx, (1 - fRow) * fCol);
            this._addTicket(uRow, lCol, idx, fRow * (1 - fCol));
        }

        // Phase 3: Brownian bridge sample + scatter back to dest pixels
        const randn = makeRandn(this._frameSeed * 104729);

        for (let srcIdx = 0; srcIdx < N; srcIdx++) {
            const nTickets = ticketCount[srcIdx];
            if (nTickets === 0) continue;

            // Sum total request weight
            let totalRequest = 0;
            const ticketBase = srcIdx * MAX_TICKETS;
            for (let k = 0; k < nTickets; k++) {
                totalRequest += areaField[ticketBase + k];
            }
            if (totalRequest <= 0) continue;

            // Brownian bridge sampling
            let pastRange = 0;
            // pastValue[c] for each channel
            const pastValue = new Float32Array(C);  // starts at 0

            for (let k = 0; k < nTickets; k++) {
                const w = areaField[ticketBase + k];
                const destIdx = masterField[ticketBase + k];
                const normalizedW = w / totalRequest;
                const nextRange = pastRange + normalizedW;

                // Sample Brownian bridge: B(nextRange) | B(pastRange) = pastValue
                // x = noise at this source pixel
                const srcNoiseBase = srcIdx * C;
                const destNoiseBase = destIdx * C;

                for (let c = 0; c < C; c++) {
                    const x = noise[srcNoiseBase + c];
                    let nextValue;

                    if (nextRange >= 1.0 || (1.0 - pastRange) <= EPS) {
                        nextValue = x;
                    } else {
                        const denom = 1.0 - pastRange;
                        const mu = (1.0 - nextRange) / denom * pastValue[c]
                                 + (nextRange - pastRange) / denom * x;
                        const variance = (nextRange - pastRange) * (1.0 - nextRange) / denom;
                        nextValue = randn() * Math.sqrt(Math.max(0, variance)) + mu;
                    }

                    const currValue = nextValue - pastValue[c];
                    buffer[destNoiseBase + c] += currValue;
                    pastValue[c] = nextValue;
                }

                pixelArea[destIdx] += normalizedW;
                pastRange = nextRange;
            }
        }

        // Phase 4: Normalize — preserve variance; unassigned pixels get fresh noise
        const freshRandn = makeRandn(this._frameSeed * 7919 + 31337);
        for (let idx = 0; idx < N; idx++) {
            const area = pixelArea[idx];
            const base = idx * C;
            if (area > 0) {
                const scale = 1.0 / Math.sqrt(area);
                for (let c = 0; c < C; c++) {
                    noise[base + c] = scale * buffer[base + c];
                }
            } else {
                for (let c = 0; c < C; c++) {
                    noise[base + c] = freshRandn();
                }
            }
        }
    }

    /**
     * Add a bilinear ticket for source pixel (row, col) requesting dest pixel destIdx.
     * @param {number} row - source row
     * @param {number} col - source col
     * @param {number} destIdx - raveled dest pixel index
     * @param {number} weight - bilinear weight
     */
    _addTicket(row, col, destIdx, weight) {
        if (weight <= 0) return;
        if (row < 0 || row >= this.H || col < 0 || col >= this.W) return;
        const srcIdx = row * this.W + col;
        const t = this.ticketCount[srcIdx];
        if (t >= MAX_TICKETS) return;
        this.ticketCount[srcIdx] = t + 1;
        const ticketBase = srcIdx * MAX_TICKETS;
        this.masterField[ticketBase + t] = destIdx;
        this.areaField[ticketBase + t] = weight;
    }

    /**
     * Flip rows of a flat [H * W * C] array (row 0 <-> row H-1).
     * Pure function.
     * @param {Float32Array} data
     * @returns {Float32Array} new array with rows flipped
     */
    _flipRows(data) {
        const { H, W, C } = this;
        const stride = W * C;
        const out = new Float32Array(data.length);
        for (let i = 0; i < H; i++) {
            const srcOff = i * stride;
            const dstOff = (H - 1 - i) * stride;
            out.set(data.subarray(srcOff, srcOff + stride), dstOff);
        }
        return out;
    }

    /**
     * Compute mean and std of noise array for validation.
     * @returns {{mean: number, std: number}}
     */
    stats() {
        const n = this.noise;
        let sum = 0, sum2 = 0;
        for (let i = 0; i < n.length; i++) {
            sum += n[i];
            sum2 += n[i] * n[i];
        }
        const mean = sum / n.length;
        const std = Math.sqrt(sum2 / n.length - mean * mean);
        return { mean, std };
    }
}
