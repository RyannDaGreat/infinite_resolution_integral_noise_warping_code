/**
 * Entry point: game loop, FPS camera, spinning cube, profiling overlay.
 * V2: WebGPU compute — zero CPU-GPU copies per frame.
 */

import { WebGPURenderer } from './renderer.js';

const { mat4, vec3, glMatrix } = window.glMatrix;

const RESOLUTIONS = [2048, 1024, 512, 256];
const BN_ITERS_OPTIONS = [2, 5, 10];
const MODE_NAMES = ['noise', 'color', 'motion', 'side-by-side', 'raw'];
const ROUND_MODES = ['None', 'All', '>1'];  // 0=none, 1=round all, 2=round if >1px
const STORAGE_KEY = 'iinw_v2_settings';
const SETTINGS_VERSION = 5;  // bump to force-reset when defaults change

// ---------------------------------------------------------------------------
// Settings persistence
// ---------------------------------------------------------------------------

const DEFAULTS = {
    resIdx: 0,
    blueNoise: false,
    bnItersIdx: 2,  // index into BN_ITERS_OPTIONS (default: 10)
    greyscale: false,
    uniformDisplay: true,
    retina: true,
    bilinear: false,
    threshOn: false,
    threshSlider: 500,  // [0,1000] → [0.0, 1.0] threshold value (0.50 = half)
    roundMode: 0,       // 0=none, 1=round all, 2=round if >1px
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
        if (this.keys['KeyE'])                             this.position[1] += v;
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
    const params = new URLSearchParams(window.location.search);

    const canvas = document.getElementById('canvas');
    const statsEl = document.getElementById('stats');
    const controlsEl = document.getElementById('controls');
    const errorEl = document.getElementById('error');

    // Check WebGPU support
    if (!navigator.gpu) {
        errorEl.textContent = 'WebGPU not available. Try Chrome 113+ or enable chrome://flags/#enable-webgpu';
        return;
    }

    // --- Load saved settings ---
    let settings = loadSettings();

    let resIdx = settings.resIdx;
    let W = RESOLUTIONS[resIdx];
    let H = W;
    let retinaOn = settings.retina;
    let bilinearOn = settings.bilinear;
    let greyscaleOn = settings.greyscale;
    let uniformDisplayOn = settings.uniformDisplay;
    let blueNoiseOn = settings.blueNoise;
    let bnItersIdx = settings.bnItersIdx;
    let bnIters = BN_ITERS_OPTIONS[bnItersIdx];
    let roundMode = settings.roundMode;
    let renderer = null;
    let displayMode = 0;
    let frameSeed = 42;

    // --- UI elements ---
    const resBtn = document.getElementById('resBtn');
    const blueNoiseBtn = document.getElementById('blueNoiseBtn');
    const bnItersBtn = document.getElementById('bnItersBtn');
    const greyBtn = document.getElementById('greyBtn');
    const uniformBtn = document.getElementById('uniformBtn');
    const retinaBtn = document.getElementById('retinaBtn');
    const interpBtn = document.getElementById('interpBtn');
    const roundBtn = document.getElementById('roundBtn');
    const resetBtn = document.getElementById('resetBtn');
    const threshBtn = document.getElementById('threshBtn');
    const threshSlider = document.getElementById('threshSlider');
    const threshLabel = document.getElementById('threshLabel');

    // --- Threshold state ---
    let threshOn = settings.threshOn;
    let threshSliderVal = settings.threshSlider;
    threshSlider.value = threshSliderVal;
    threshSlider.disabled = !threshOn;

    function updateThresholdUI() {
        threshSlider.disabled = !threshOn;
        threshBtn.textContent = `[H] Thresh: ${threshOn ? 'ON' : 'OFF'}`;
        threshBtn.classList.toggle('on', threshOn);
        // Slider [0,1000] → [0.0, 1.0] threshold value
        threshLabel.textContent = threshOn ? (threshSliderVal / 1000).toFixed(2) : '';
    }

    function applyThreshold() {
        if (renderer) {
            renderer.thresholdOn = threshOn ? 1 : 0;
            renderer.thresholdValue = threshSliderVal / 1000;
        }
        updateThresholdUI();
    }

    // --- Sync UI to loaded settings ---
    function syncUI() {
        resBtn.textContent = `[P] ${W}`;
        blueNoiseBtn.textContent = `[B] Blue: ${blueNoiseOn ? 'ON' : 'OFF'}`;
        blueNoiseBtn.classList.toggle('on', blueNoiseOn);
        bnItersBtn.textContent = `[N] BN\u00d7${bnIters}`;
        greyBtn.textContent = `[G] Grey: ${greyscaleOn ? 'ON' : 'OFF'}`;
        greyBtn.classList.toggle('on', greyscaleOn);
        uniformBtn.textContent = `[U] Uniform: ${uniformDisplayOn ? 'ON' : 'OFF'}`;
        uniformBtn.classList.toggle('on', uniformDisplayOn);
        retinaBtn.textContent = `[T] Retina: ${retinaOn ? 'ON' : 'OFF'}`;
        retinaBtn.classList.toggle('on', retinaOn);
        interpBtn.textContent = `[I] Interp: ${bilinearOn ? 'Bilinear' : 'Nearest'}`;
        interpBtn.classList.toggle('on', !bilinearOn);
        roundBtn.textContent = `[O] Round: ${ROUND_MODES[roundMode]}`;
        roundBtn.classList.toggle('on', roundMode > 0);
        threshSlider.value = threshSliderVal;
        updateThresholdUI();
    }

    function persistSettings() {
        saveSettings({
            resIdx, blueNoise: blueNoiseOn, bnItersIdx,
            greyscale: greyscaleOn, uniformDisplay: uniformDisplayOn,
            retina: retinaOn, bilinear: bilinearOn,
            roundMode, threshOn, threshSlider: threshSliderVal,
        });
    }

    // --- Canvas sizing ---
    function updateCanvasSize() {
        canvas.width = W;
        canvas.height = H;
        const dpr = retinaOn ? (window.devicePixelRatio || 1) : 1;
        canvas.style.width = (W / dpr) + 'px';
        canvas.style.height = (H / dpr) + 'px';
    }

    function updateInterpolation() {
        canvas.style.imageRendering = bilinearOn ? 'auto' : 'pixelated';
    }

    // --- Renderer lifecycle ---
    async function createRenderer() {
        if (renderer) renderer.destroy();
        renderer = new WebGPURenderer(canvas, W, H);
        renderer.brownianWGOverride = parseInt(params.get('brownian_wg')) || 256;
        renderer.blueNoiseEnabled = blueNoiseOn;
        renderer.blueNoiseIterations = bnIters;
        renderer.greyscaleEnabled = greyscaleOn;
        renderer.uniformDisplayEnabled = uniformDisplayOn;
        renderer.roundMode = roundMode;
        await renderer.init();
    }

    // --- Initial setup ---
    updateCanvasSize();
    updateInterpolation();
    syncUI();

    try {
        await createRenderer();
        applyThreshold();
    } catch (e) {
        errorEl.textContent = e.message;
        throw e;
    }

    statsEl.textContent = 'WebGPU ready. Click to start.';

    // --- Button handlers ---

    function cycleResolution() {
        resIdx = (resIdx + 1) % RESOLUTIONS.length;
        W = RESOLUTIONS[resIdx];
        H = W;
        resBtn.textContent = `[P] ${W}`;
        updateCanvasSize();
        persistSettings();
        createRenderer();
    }
    resBtn.addEventListener('click', cycleResolution);

    function toggleBlueNoise() {
        blueNoiseOn = !blueNoiseOn;
        renderer.blueNoiseEnabled = blueNoiseOn;
        blueNoiseBtn.textContent = `[B] Blue: ${blueNoiseOn ? 'ON' : 'OFF'}`;
        blueNoiseBtn.classList.toggle('on', blueNoiseOn);
        persistSettings();
    }
    blueNoiseBtn.addEventListener('click', toggleBlueNoise);

    function cycleBnIters() {
        bnItersIdx = (bnItersIdx + 1) % BN_ITERS_OPTIONS.length;
        bnIters = BN_ITERS_OPTIONS[bnItersIdx];
        renderer.blueNoiseIterations = bnIters;
        bnItersBtn.textContent = `[N] BN\u00d7${bnIters}`;
        persistSettings();
    }
    bnItersBtn.addEventListener('click', cycleBnIters);

    function toggleGreyscale() {
        greyscaleOn = !greyscaleOn;
        renderer.greyscaleEnabled = greyscaleOn;
        greyBtn.textContent = `[G] Grey: ${greyscaleOn ? 'ON' : 'OFF'}`;
        greyBtn.classList.toggle('on', greyscaleOn);
        persistSettings();
    }
    greyBtn.addEventListener('click', toggleGreyscale);

    function toggleUniformDisplay() {
        uniformDisplayOn = !uniformDisplayOn;
        renderer.uniformDisplayEnabled = uniformDisplayOn;
        uniformBtn.textContent = `[U] Uniform: ${uniformDisplayOn ? 'ON' : 'OFF'}`;
        uniformBtn.classList.toggle('on', uniformDisplayOn);
        persistSettings();
    }
    uniformBtn.addEventListener('click', toggleUniformDisplay);

    function toggleRetina() {
        retinaOn = !retinaOn;
        retinaBtn.textContent = `[T] Retina: ${retinaOn ? 'ON' : 'OFF'}`;
        retinaBtn.classList.toggle('on', retinaOn);
        updateCanvasSize();
        persistSettings();
    }
    retinaBtn.addEventListener('click', toggleRetina);

    function toggleInterp() {
        bilinearOn = !bilinearOn;
        interpBtn.textContent = `[I] Interp: ${bilinearOn ? 'Bilinear' : 'Nearest'}`;
        interpBtn.classList.toggle('on', !bilinearOn);
        updateInterpolation();
        persistSettings();
    }
    interpBtn.addEventListener('click', toggleInterp);

    function cycleRoundMode() {
        roundMode = (roundMode + 1) % ROUND_MODES.length;
        renderer.roundMode = roundMode;
        roundBtn.textContent = `[O] Round: ${ROUND_MODES[roundMode]}`;
        roundBtn.classList.toggle('on', roundMode > 0);
        persistSettings();
    }
    roundBtn.addEventListener('click', cycleRoundMode);

    function toggleThreshold() {
        threshOn = !threshOn;
        applyThreshold();
        persistSettings();
    }
    threshBtn.addEventListener('click', toggleThreshold);

    threshSlider.addEventListener('input', () => {
        threshSliderVal = parseInt(threshSlider.value);
        applyThreshold();
        persistSettings();
    });

    function resetAll() {
        localStorage.removeItem(STORAGE_KEY);
        const d = DEFAULTS;
        resIdx = d.resIdx; W = RESOLUTIONS[resIdx]; H = W;
        blueNoiseOn = d.blueNoise;
        bnItersIdx = d.bnItersIdx; bnIters = BN_ITERS_OPTIONS[bnItersIdx];
        greyscaleOn = d.greyscale;
        uniformDisplayOn = d.uniformDisplay;
        retinaOn = d.retina;
        bilinearOn = d.bilinear;
        roundMode = d.roundMode;
        threshOn = d.threshOn;
        threshSliderVal = d.threshSlider;
        syncUI();
        updateCanvasSize();
        updateInterpolation();
        createRenderer().then(() => applyThreshold());
    }
    resetBtn.addEventListener('click', resetAll);

    // --- Camera & cube ---
    const camera = new FPSCamera();
    const cube = new SpinningCube();

    let proj = mat4.create();
    mat4.perspective(proj, glMatrix.toRadian(60), 1.0, 0.1, 100);

    let prevViewProj = mat4.create();
    mat4.mul(prevViewProj, proj, camera.viewMatrix());

    let mouseCaptured = false;
    let paused = false;
    let lastTime = performance.now();

    // FPS tracking
    let frameCounter = 0;
    let fpsAccum = 0;
    let displayFPS = 0;

    // CPU frame timing
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
        // Skip hotkeys when Cmd/Ctrl held — prevents Cmd+R (refresh) from cycling resolution, etc.
        if (!e.metaKey && !e.ctrlKey) {
            if (e.code === 'KeyB') toggleBlueNoise();
            if (e.code === 'KeyN') cycleBnIters();
            if (e.code === 'KeyG') toggleGreyscale();
            if (e.code === 'KeyU') toggleUniformDisplay();
            if (e.code === 'KeyP' && !mouseCaptured) cycleResolution();
            if (e.code === 'KeyT') toggleRetina();
            if (e.code === 'KeyI') toggleInterp();
            if (e.code === 'KeyH') toggleThreshold();
            if (e.code === 'KeyO') cycleRoundMode();
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
        if (!renderer) { requestAnimationFrame(frame); return; }

        paused = camera.keys['Space'] === true;
        if (paused) { lastTime = now; requestAnimationFrame(frame); return; }

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

        const cpuMs = performance.now() - cpuStart;
        cpuFrameHistory.push(cpuMs);
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

        lines.push(
            `FPS: ${displayFPS} | ` +
            `${W}x${H} | ` +
            `mean: ${mean.toFixed(3)} | std: ${std.toFixed(3)} | ` +
            `mode: ${MODE_NAMES[displayMode]}`
        );

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

        if (cpuFrameHistory.length > 10) {
            const recent = cpuFrameHistory.slice(-100);
            const cpuMean = recent.reduce((a,b) => a+b, 0) / recent.length;
            const cpuStd = Math.sqrt(recent.reduce((a,b) => a + (b - cpuMean)**2, 0) / recent.length);
            lines.push(`[CPU] encode+submit: ${cpuMean.toFixed(2)}±${cpuStd.toFixed(2)} ms`);
        }

        statsEl.textContent = lines.join('\n');

        window.__stats = {
            fps: displayFPS,
            noiseStats: { mean, std },
            gpuStats,
            cpuFrameMs: cpuFrameHistory.slice(-100),
        };

        if (Math.abs(mean) > 0.1 || Math.abs(std - 1.0) > 0.2) {
            console.warn(`Noise stats drifting: mean=${mean.toFixed(4)}, std=${std.toFixed(4)}`);
        }
    }

    requestAnimationFrame(frame);
}

main();
