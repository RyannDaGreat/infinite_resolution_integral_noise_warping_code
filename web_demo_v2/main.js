/**
 * Entry point: game loop, FPS camera, spinning cube, profiling overlay.
 * V2: WebGPU compute — zero CPU-GPU copies per frame.
 */

import { WebGPURenderer } from './renderer.js';

const { mat4, vec3, glMatrix } = window.glMatrix;

const WIDTH = 2048;
const HEIGHT = 2048;

const MODE_NAMES = ['noise', 'color', 'motion', 'side-by-side', 'raw'];

// ---------------------------------------------------------------------------
// FPS Camera (pointer lock + WASD/EQ)
// ---------------------------------------------------------------------------

class FPSCamera {
    constructor() {
        this.position = vec3.fromValues(0, 1, 4);
        this.yaw = -90;
        this.pitch = -10;
        this.speed = 3;
        this.sensitivity = 0.15;
        this.keys = {};
    }

    processMouse(dx, dy) {
        this.yaw += dx * this.sensitivity;
        this.pitch -= dy * this.sensitivity;
        this.pitch = Math.max(-89, Math.min(89, this.pitch));
    }

    processKeys(dt) {
        const fwd = this._forward();
        const right = vec3.create();
        vec3.cross(right, fwd, [0, 1, 0]);
        vec3.normalize(right, right);
        const v = this.speed * dt;
        if (this.keys['KeyW'] || this.keys['ArrowUp'])    vec3.scaleAndAdd(this.position, this.position, fwd, v);
        if (this.keys['KeyS'] || this.keys['ArrowDown'])  vec3.scaleAndAdd(this.position, this.position, fwd, -v);
        if (this.keys['KeyA'] || this.keys['ArrowLeft'])   vec3.scaleAndAdd(this.position, this.position, right, -v);
        if (this.keys['KeyD'] || this.keys['ArrowRight']) vec3.scaleAndAdd(this.position, this.position, right, v);
        if (this.keys['KeyE'] || this.keys['Space'])      this.position[1] += v;
        if (this.keys['KeyQ'] || this.keys['ShiftLeft'])  this.position[1] -= v;
    }

    _forward() {
        const ry = glMatrix.toRadian(this.yaw);
        const rp = glMatrix.toRadian(this.pitch);
        const f = vec3.fromValues(
            Math.cos(rp) * Math.cos(ry), Math.sin(rp), Math.cos(rp) * Math.sin(ry),
        );
        vec3.normalize(f, f);
        return f;
    }

    viewMatrix() {
        const f = this._forward();
        const target = vec3.create();
        vec3.add(target, this.position, f);
        const out = mat4.create();
        mat4.lookAt(out, this.position, target, [0, 1, 0]);
        return out;
    }
}

// ---------------------------------------------------------------------------
// Spinning Cube
// ---------------------------------------------------------------------------

class SpinningCube {
    constructor() {
        this.angle = 0;
        this.axis = vec3.fromValues(0.7071, 0.7071, 0);
        this.speed = 1.5;
        this.targetAxis = this._randomAxis();
        this.targetSpeed = 0.75 + Math.random() * 2.25;
        this.changeTimer = 0;
        this.changeInterval = 3;
        this.prevModel = mat4.create();
    }

    _randomAxis() {
        const randn = () => {
            const u1 = Math.max(Math.random(), 1e-10), u2 = Math.random();
            return Math.sqrt(-2 * Math.log(u1)) * Math.cos(6.283185307 * u2);
        };
        const v = vec3.fromValues(randn(), randn(), randn());
        vec3.normalize(v, v);
        return v;
    }

    update(dt) {
        mat4.copy(this.prevModel, this.modelMatrix());
        const t = Math.min(dt * 2, 1);
        vec3.lerp(this.axis, this.axis, this.targetAxis, t);
        vec3.normalize(this.axis, this.axis);
        this.speed += (this.targetSpeed - this.speed) * t;
        this.angle += this.speed * dt;
        this.changeTimer += dt;
        if (this.changeTimer > this.changeInterval) {
            this.changeTimer = 0;
            this.targetAxis = this._randomAxis();
            this.targetSpeed = 0.75 + Math.random() * 2.25;
            this.changeInterval = 2 + Math.random() * 3;
        }
    }

    modelMatrix() {
        const out = mat4.create();
        mat4.rotate(out, out, this.angle, this.axis);
        return out;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
    // URL params for benchmark tuning (e.g. ?brownian_wg=128)
    const params = new URLSearchParams(window.location.search);

    const canvas = document.getElementById('canvas');
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    // Retina: render at full resolution, display at half CSS size
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = (WIDTH / dpr) + 'px';
    canvas.style.height = (HEIGHT / dpr) + 'px';

    const statsEl = document.getElementById('stats');
    const controlsEl = document.getElementById('controls');
    const hintEl = document.getElementById('hint');
    const errorEl = document.getElementById('error');

    // Check WebGPU support
    if (!navigator.gpu) {
        errorEl.textContent = 'WebGPU not available. Try Chrome 113+ or enable chrome://flags/#enable-webgpu';
        return;
    }

    let renderer;
    try {
        renderer = new WebGPURenderer(canvas, WIDTH, HEIGHT);
        const brownianWG = parseInt(params.get('brownian_wg')) || 256;
        renderer.brownianWGOverride = brownianWG;
        await renderer.init();
    } catch (e) {
        errorEl.textContent = e.message;
        throw e;
    }

    statsEl.textContent = 'WebGPU ready. Click to start.';

    const camera = new FPSCamera();
    const cube = new SpinningCube();

    const proj = mat4.create();
    mat4.perspective(proj, glMatrix.toRadian(60), WIDTH / HEIGHT, 0.1, 100);

    let prevViewProj = mat4.create();
    mat4.mul(prevViewProj, proj, camera.viewMatrix());

    let displayMode = 0;
    let mouseCaptured = false;
    let lastTime = performance.now();
    let frameSeed = 42;

    // FPS tracking
    let frameCounter = 0;
    let fpsAccum = 0;
    let displayFPS = 0;

    // CPU frame timing
    let cpuFrameMs = 0;
    const cpuFrameHistory = [];

    // --- Input handlers ---
    document.addEventListener('keydown', (e) => {
        camera.keys[e.code] = true;
        if (e.code >= 'Digit1' && e.code <= 'Digit5') {
            displayMode = parseInt(e.code.slice(5)) - 1;
        }
        if (e.code === 'Escape' && mouseCaptured) {
            document.exitPointerLock();
        }
    });
    document.addEventListener('keyup', (e) => { camera.keys[e.code] = false; });
    canvas.addEventListener('click', () => {
        if (!mouseCaptured) canvas.requestPointerLock();
    });
    document.addEventListener('pointerlockchange', () => {
        mouseCaptured = document.pointerLockElement === canvas;
        controlsEl.style.display = mouseCaptured ? 'none' : 'block';
    });
    document.addEventListener('mousemove', (e) => {
        if (mouseCaptured) camera.processMouse(e.movementX, e.movementY);
    });

    // --- Game loop ---
    function frame(now) {
        const dt = Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;

        camera.processKeys(dt);
        cube.update(dt);

        const viewProj = mat4.create();
        mat4.mul(viewProj, proj, camera.viewMatrix());
        const model = cube.modelMatrix();

        const cpuStart = performance.now();

        renderer.frame({
            model, viewProj,
            prevModel: cube.prevModel,
            prevViewProj: prevViewProj,
            displayMode,
            frameSeed,
        });

        cpuFrameMs = performance.now() - cpuStart;
        cpuFrameHistory.push(cpuFrameMs);
        if (cpuFrameHistory.length > 200) cpuFrameHistory.shift();

        prevViewProj = viewProj;
        frameSeed++;

        // FPS + stats update (1x/sec)
        frameCounter++;
        fpsAccum += dt;
        if (fpsAccum >= 1.0) {
            displayFPS = Math.round(frameCounter / fpsAccum);
            frameCounter = 0;
            fpsAccum = 0;
            updateStats();
        }

        requestAnimationFrame(frame);
    }

    function updateStats() {
        const { mean, std } = renderer.noiseStats;
        const gpuStats = renderer.getTimingStats();

        let lines = [];

        // Line 1: FPS + noise stats + mode
        lines.push(
            `FPS: ${displayFPS} | ` +
            `mean: ${mean.toFixed(3)} | std: ${std.toFixed(3)} | ` +
            `mode: ${MODE_NAMES[displayMode]}`
        );

        // Line 2: GPU per-phase breakdown (if available)
        if (gpuStats) {
            const fmt = (s) => `${s.mean.toFixed(2)}±${s.std.toFixed(2)}`;
            lines.push(
                `[GPU] scene:${fmt(gpuStats.scene)} ` +
                `deform:${fmt(gpuStats.buildDeform)} ` +
                `bwd:${fmt(gpuStats.backwardMap)} ` +
                `brown:${fmt(gpuStats.brownian)} ` +
                `norm:${fmt(gpuStats.normalize)} ` +
                `disp:${fmt(gpuStats.display)} ` +
                `total:${fmt(gpuStats.total)} ms`
            );
        }

        // Line 3: CPU encode+submit timing
        if (cpuFrameHistory.length > 10) {
            const recent = cpuFrameHistory.slice(-100);
            const cpuMean = recent.reduce((a,b) => a+b, 0) / recent.length;
            const cpuStd = Math.sqrt(recent.reduce((a,b) => a + (b - cpuMean)**2, 0) / recent.length);
            lines.push(`[CPU] encode+submit: ${cpuMean.toFixed(2)}±${cpuStd.toFixed(2)} ms`);
        }

        statsEl.textContent = lines.join('\n');

        // Expose for test suite
        window.__stats = {
            fps: displayFPS,
            noiseStats: { mean, std },
            gpuStats,
            cpuFrameMs: cpuFrameHistory.slice(-100),
        };

        // Warn if noise stats drift
        if (Math.abs(mean) > 0.1 || Math.abs(std - 1.0) > 0.2) {
            console.warn(`Noise stats drifting: mean=${mean.toFixed(4)}, std=${std.toFixed(4)}`);
        }
    }

    requestAnimationFrame(frame);
}

main();
