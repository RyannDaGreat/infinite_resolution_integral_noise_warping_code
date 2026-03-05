/**
 * UI state management: settings persistence, button handlers, hotkeys, stats overlay.
 * Knows nothing about GPU internals — communicates via callbacks.
 */

const RESOLUTIONS = [2048, 1024, 512, 256];
const BN_ITERS_OPTIONS = [2, 5, 10];
const MODE_NAMES = ['noise', 'scene', 'scene+noise', 'dither', 'motion', 'raw'];
const ROUND_MODES = ['None', 'All', '>1'];
const SHADOW_RES_OPTIONS = [512, 1024, 2048, 4096, 8192, 16384];
const STORAGE_KEY = 'iinw_v3_settings';
const SETTINGS_VERSION = 3;

const DEFAULTS = {
    resIdx: 1,          // 1024 default for V3 (physics is heavier)
    blueNoise: false,
    bnItersIdx: 0,
    greyscale: true,
    uniformDisplay: true,
    retina: false,
    bilinear: false,
    threshOn: false,
    threshSlider: 500,
    roundMode: 2,       // '>1' rounding
    noiseOpacity: 25,   // 0-100 → 0.0-1.0
    shadows: true,
    pointLights: true,
    terrain: true,
    shadowResIdx: 3,    // 4096 default
    daySpeed: 20,       // 0-300 → 0.0-3.0 multiplier (20 = 0.2x = 5x slower)
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

export class UIManager {
    /**
     * @param {object} callbacks - { onResolutionChange, onRendererUpdate, onResetScene }
     */
    constructor(callbacks) {
        this.callbacks = callbacks;
        this.settings = loadSettings();
        this.displayMode = 3;  // dither mode

        // Bind DOM elements
        this.resBtn = document.getElementById('resBtn');
        this.blueNoiseBtn = document.getElementById('blueNoiseBtn');
        this.bnItersBtn = document.getElementById('bnItersBtn');
        this.greyBtn = document.getElementById('greyBtn');
        this.uniformBtn = document.getElementById('uniformBtn');
        this.retinaBtn = document.getElementById('retinaBtn');
        this.interpBtn = document.getElementById('interpBtn');
        this.roundBtn = document.getElementById('roundBtn');
        this.resetBtn = document.getElementById('resetBtn');
        this.threshBtn = document.getElementById('threshBtn');
        this.threshSlider = document.getElementById('threshSlider');
        this.threshLabel = document.getElementById('threshLabel');
        this.lockNoiseBtn = document.getElementById('lockNoiseBtn');
        this.noiseOpacitySlider = document.getElementById('noiseOpacitySlider');
        this.noiseOpacityLabel = document.getElementById('noiseOpacityLabel');
        this.modeSettingsEl = document.getElementById('modeSettings');
        this.resetSceneBtn = document.getElementById('resetSceneBtn');
        this.slowMoBtn = document.getElementById('slowMoBtn');
        this.timeMiddayBtn = document.getElementById('timeMidday');
        this.timeSunsetBtn = document.getElementById('timeSunset');
        this.timeNightBtn = document.getElementById('timeNight');
        this.timeDawnBtn = document.getElementById('timeDawn');
        this.modeBar = document.getElementById('modeBar');
        this.modeBtns = this.modeBar ? [...this.modeBar.querySelectorAll('.modeBtn')] : [];
        this.statsEl = document.getElementById('stats');

        this.shadowsBtn = document.getElementById('shadowsBtn');
        this.pointLightsBtn = document.getElementById('pointLightsBtn');
        this.terrainBtn = document.getElementById('terrainBtn');
        this.shadowResBtn = document.getElementById('shadowResBtn');
        this.daySpeedSlider = document.getElementById('daySpeedSlider');
        this.daySpeedLabel = document.getElementById('daySpeedLabel');

        this.slowMo = false;
        this.noiseLocked = false;
        this._setupHandlers();
        this.syncUI();
    }

    get W() { return RESOLUTIONS[this.settings.resIdx]; }
    get H() { return this.W; }

    _persist() { saveSettings(this.settings); }

    syncUI() {
        const s = this.settings;
        this.resBtn.textContent = `[P] ${this.W}`;
        this.blueNoiseBtn.textContent = `[B] Blue: ${s.blueNoise ? 'ON' : 'OFF'}`;
        this.blueNoiseBtn.classList.toggle('on', s.blueNoise);
        this.bnItersBtn.textContent = `[N] BN\u00d7${BN_ITERS_OPTIONS[s.bnItersIdx]}`;
        this.greyBtn.textContent = `[G] Grey: ${s.greyscale ? 'ON' : 'OFF'}`;
        this.greyBtn.classList.toggle('on', s.greyscale);
        this.uniformBtn.textContent = `[U] Uniform: ${s.uniformDisplay ? 'ON' : 'OFF'}`;
        this.uniformBtn.classList.toggle('on', s.uniformDisplay);
        this.retinaBtn.textContent = `[T] Retina: ${s.retina ? 'ON' : 'OFF'}`;
        this.retinaBtn.classList.toggle('on', s.retina);
        this.interpBtn.textContent = `[I] Interp: ${s.bilinear ? 'Bilinear' : 'Nearest'}`;
        this.interpBtn.classList.toggle('on', !s.bilinear);
        this.roundBtn.textContent = `[O] Round: ${ROUND_MODES[s.roundMode]}`;
        this.roundBtn.classList.toggle('on', s.roundMode > 0);
        this.threshSlider.value = s.threshSlider;
        this.threshSlider.disabled = !s.threshOn;
        this.threshBtn.textContent = `[H] Thresh: ${s.threshOn ? 'ON' : 'OFF'}`;
        this.threshBtn.classList.toggle('on', s.threshOn);
        this.threshLabel.textContent = s.threshOn ? (s.threshSlider / 1000).toFixed(2) : '';
        this.slowMoBtn.textContent = `[M] Slow: ${this.slowMo ? 'ON' : 'OFF'}`;
        this.slowMoBtn.classList.toggle('on', this.slowMo);
        this.lockNoiseBtn.textContent = `[L] Lock: ${this.noiseLocked ? 'ON' : 'OFF'}`;
        this.lockNoiseBtn.classList.toggle('on', this.noiseLocked);
        this.shadowsBtn.textContent = `[V] Shadows: ${s.shadows ? 'ON' : 'OFF'}`;
        this.shadowsBtn.classList.toggle('on', s.shadows);
        this.pointLightsBtn.textContent = `Lights: ${s.pointLights ? 'ON' : 'OFF'}`;
        this.pointLightsBtn.classList.toggle('on', s.pointLights);
        this.terrainBtn.textContent = `Terrain: ${s.terrain ? 'ON' : 'OFF'}`;
        this.terrainBtn.classList.toggle('on', s.terrain);
        this.shadowResBtn.textContent = `Shadow: ${SHADOW_RES_OPTIONS[s.shadowResIdx]}`;
        this.shadowResBtn.classList.toggle('on', s.shadowResIdx > 2);
        this.daySpeedSlider.value = s.daySpeed;
        this.daySpeedLabel.textContent = `${(s.daySpeed / 100).toFixed(1)}x`;
        // Noise opacity slider (visible only in S+N mode)
        this.modeSettingsEl.style.display = (this.displayMode === 2) ? 'inline' : 'none';
        this.noiseOpacitySlider.value = s.noiseOpacity;
        this.noiseOpacityLabel.textContent = (s.noiseOpacity / 100).toFixed(2);
        // Sync mode bar
        for (const btn of this.modeBtns) {
            btn.classList.toggle('on', parseInt(btn.dataset.mode) === this.displayMode);
        }
    }

    /**
     * Apply current settings to a renderer instance.
     * Not pure: mutates renderer state.
     */
    applyToRenderer(renderer) {
        const s = this.settings;
        renderer.blueNoiseEnabled = s.blueNoise;
        renderer.blueNoiseIterations = BN_ITERS_OPTIONS[s.bnItersIdx];
        renderer.greyscaleEnabled = s.greyscale;
        renderer.uniformDisplayEnabled = s.uniformDisplay;
        renderer.roundMode = s.roundMode;
        renderer.thresholdOn = s.threshOn ? 1 : 0;
        renderer.thresholdValue = s.threshSlider / 1000;
        renderer.noiseOpacity = s.noiseOpacity / 100;
        renderer.noiseLocked = this.noiseLocked;
        renderer.shadowsEnabled = s.shadows;
        renderer.pointLightsEnabled = s.pointLights;
        renderer.terrainEnabled = s.terrain;
        renderer.daySpeedMultiplier = s.daySpeed / 100;
    }

    /**
     * Update the canvas element's CSS size and interpolation.
     * Not pure: mutates DOM.
     */
    updateCanvas(canvas) {
        const s = this.settings;
        const dpr = s.retina ? (window.devicePixelRatio || 1) : 1;
        canvas.width = this.W;
        canvas.height = this.H;
        canvas.style.width = (this.W / dpr) + 'px';
        canvas.style.height = (this.H / dpr) + 'px';
        canvas.style.imageRendering = s.bilinear ? 'auto' : 'pixelated';
    }

    _setupHandlers() {
        const s = this.settings;
        const update = () => { this._persist(); this.syncUI(); this.callbacks.onRendererUpdate?.(); };

        this.resBtn.addEventListener('click', () => {
            s.resIdx = (s.resIdx + 1) % RESOLUTIONS.length;
            this._persist(); this.syncUI();
            this.callbacks.onResolutionChange?.();
        });

        this.blueNoiseBtn.addEventListener('click', () => { s.blueNoise = !s.blueNoise; update(); });
        this.bnItersBtn.addEventListener('click', () => {
            s.bnItersIdx = (s.bnItersIdx + 1) % BN_ITERS_OPTIONS.length; update();
        });
        this.greyBtn.addEventListener('click', () => { s.greyscale = !s.greyscale; update(); });
        this.uniformBtn.addEventListener('click', () => { s.uniformDisplay = !s.uniformDisplay; update(); });
        this.retinaBtn.addEventListener('click', () => {
            s.retina = !s.retina; this._persist(); this.syncUI();
            this.callbacks.onResolutionChange?.();
        });
        this.interpBtn.addEventListener('click', () => {
            s.bilinear = !s.bilinear; this._persist(); this.syncUI();
            this.callbacks.onRendererUpdate?.();
        });
        this.roundBtn.addEventListener('click', () => {
            s.roundMode = (s.roundMode + 1) % ROUND_MODES.length; update();
        });
        this.threshBtn.addEventListener('click', () => { s.threshOn = !s.threshOn; update(); });
        this.threshSlider.addEventListener('input', () => {
            s.threshSlider = parseInt(this.threshSlider.value); update();
        });
        this.resetSceneBtn.addEventListener('click', () => { this.callbacks.onResetScene?.(); });
        this.slowMoBtn.addEventListener('click', () => {
            this.slowMo = !this.slowMo; this.syncUI();
            this.callbacks.onSlowMo?.(this.slowMo);
        });
        this.lockNoiseBtn.addEventListener('click', () => {
            this.noiseLocked = !this.noiseLocked; this.syncUI();
            this.callbacks.onRendererUpdate?.();
        });
        this.noiseOpacitySlider.addEventListener('input', () => {
            s.noiseOpacity = parseInt(this.noiseOpacitySlider.value); update();
        });
        this.shadowsBtn.addEventListener('click', () => { s.shadows = !s.shadows; update(); });
        this.pointLightsBtn.addEventListener('click', () => { s.pointLights = !s.pointLights; update(); });
        this.terrainBtn.addEventListener('click', () => { s.terrain = !s.terrain; update(); });
        this.shadowResBtn.addEventListener('click', () => {
            s.shadowResIdx = (s.shadowResIdx + 1) % SHADOW_RES_OPTIONS.length;
            this._persist(); this.syncUI();
            this.callbacks.onShadowResChange?.(SHADOW_RES_OPTIONS[s.shadowResIdx]);
        });
        this.daySpeedSlider.addEventListener('input', () => {
            s.daySpeed = parseInt(this.daySpeedSlider.value); update();
        });
        // Mode bar buttons
        for (const btn of this.modeBtns) {
            btn.addEventListener('click', () => {
                this.displayMode = parseInt(btn.dataset.mode);
                this.syncUI();
            });
        }

        // Time-of-day presets (elapsedSecs that place the sun at the desired angle)
        const TIME_PRESETS = { midday: 0, sunset: 75, night: 150, dawn: 225 };
        this.timeMiddayBtn.addEventListener('click', () => { this.callbacks.onSetTime?.(TIME_PRESETS.midday); });
        this.timeSunsetBtn.addEventListener('click', () => { this.callbacks.onSetTime?.(TIME_PRESETS.sunset); });
        this.timeNightBtn.addEventListener('click',  () => { this.callbacks.onSetTime?.(TIME_PRESETS.night); });
        this.timeDawnBtn.addEventListener('click',   () => { this.callbacks.onSetTime?.(TIME_PRESETS.dawn); });

        this.resetBtn.addEventListener('click', () => {
            localStorage.removeItem(STORAGE_KEY);
            this.settings = { ...DEFAULTS };
            this.slowMo = false;
            this.noiseLocked = false;
            this._persist(); this.syncUI();
            this.callbacks.onResolutionChange?.();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.metaKey || e.ctrlKey) return;
            if (e.code >= 'Digit1' && e.code <= 'Digit6') {
                this.displayMode = parseInt(e.code.slice(5)) - 1;
                this.syncUI();
            }
            if (e.code === 'KeyB') this.blueNoiseBtn.click();
            if (e.code === 'KeyN') this.bnItersBtn.click();
            if (e.code === 'KeyG') this.greyBtn.click();
            if (e.code === 'KeyU') this.uniformBtn.click();
            if (e.code === 'KeyT') this.retinaBtn.click();
            if (e.code === 'KeyI') this.interpBtn.click();
            if (e.code === 'KeyH') this.threshBtn.click();
            if (e.code === 'KeyO') this.roundBtn.click();
            if (e.code === 'KeyP') this.resBtn.click();
            if (e.code === 'KeyR') this.resetSceneBtn.click();
            if (e.code === 'KeyL') this.lockNoiseBtn.click();
            if (e.code === 'KeyM') this.slowMoBtn.click();
            if (e.code === 'KeyV') this.shadowsBtn.click();
        });
    }

    updateStats(fps, noiseStats, gpuStats, cpuFrameHistory, physicsMs) {
        const { mean, std } = noiseStats;
        let lines = [];
        lines.push(
            `FPS: ${fps} | ${this.W}x${this.H} | ` +
            `mean: ${mean.toFixed(3)} | std: ${std.toFixed(3)} | ` +
            `mode: ${MODE_NAMES[this.displayMode]}`
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
            lines.push(`[CPU] encode: ${cpuMean.toFixed(2)}ms | physics: ${physicsMs.toFixed(2)}ms`);
        }
        this.statsEl.textContent = lines.join('\n');
    }
}

export { RESOLUTIONS, BN_ITERS_OPTIONS, MODE_NAMES };
