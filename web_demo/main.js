/**
 * Entry point: game loop, FPS camera, spinning cube.
 * Port of game_demo/main.py (taichi mode).
 */

import { Renderer } from './renderer.js';
import { ParticleWarper } from './particle_warp.js';

const { mat4, vec3, glMatrix } = window.glMatrix;

const WIDTH = 512;
const HEIGHT = 512;
const CHANNELS = 4;

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
        if (this.keys['KeyS'] || this.keys['ArrowDown'])   vec3.scaleAndAdd(this.position, this.position, fwd, -v);
        if (this.keys['KeyA'] || this.keys['ArrowLeft'])   vec3.scaleAndAdd(this.position, this.position, right, -v);
        if (this.keys['KeyD'] || this.keys['ArrowRight'])  vec3.scaleAndAdd(this.position, this.position, right, v);
        if (this.keys['KeyE'] || this.keys['Space'])       this.position[1] += v;
        if (this.keys['KeyQ'] || this.keys['ShiftLeft'])   this.position[1] -= v;
    }

    _forward() {
        const ry = glMatrix.toRadian(this.yaw);
        const rp = glMatrix.toRadian(this.pitch);
        const f = vec3.fromValues(
            Math.cos(rp) * Math.cos(ry),
            Math.sin(rp),
            Math.cos(rp) * Math.sin(ry),
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
// Spinning Cube (smooth random axis/speed changes)
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
        const v = vec3.fromValues(
            this._randn(), this._randn(), this._randn(),
        );
        vec3.normalize(v, v);
        return v;
    }

    _randn() {
        // Box-Muller
        const u1 = Math.max(Math.random(), 1e-10);
        const u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(6.283185307 * u2);
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
// Noise upload helper: expand C-channel noise to RGBA for texture
// ---------------------------------------------------------------------------

/**
 * Convert [H*W*C] flat noise to [H*W*4] RGBA Float32Array for GPU upload.
 * If C < 4, pads with 0. If C >= 4, takes first 4.
 * Pure function.
 * @param {Float32Array} data - [H*W*C] noise
 * @param {number} H
 * @param {number} W
 * @param {number} C
 * @returns {Float32Array} [H*W*4]
 */
function noiseToRGBA(data, H, W, C) {
    if (C === 4) return data;
    const out = new Float32Array(H * W * 4);
    for (let i = 0; i < H * W; i++) {
        for (let c = 0; c < Math.min(C, 4); c++) {
            out[i * 4 + c] = data[i * C + c];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
    const canvas = document.getElementById('canvas');
    canvas.width = WIDTH;
    canvas.height = HEIGHT;

    const statsEl = document.getElementById('stats');
    const controlsEl = document.getElementById('controls');
    const errorEl = document.getElementById('error');

    let renderer;
    try {
        renderer = new Renderer(canvas, WIDTH, HEIGHT);
    } catch (e) {
        errorEl.textContent = e.message;
        throw e;
    }
    const warper = new ParticleWarper(HEIGHT, WIDTH, CHANNELS);
    const camera = new FPSCamera();
    const cube = new SpinningCube();

    // Upload initial noise
    const initNoise = warper.getInitNoiseForGPU();
    renderer.uploadNoise(noiseToRGBA(initNoise, HEIGHT, WIDTH, CHANNELS));

    const proj = mat4.create();
    mat4.perspective(proj, glMatrix.toRadian(60), WIDTH / HEIGHT, 0.1, 100);

    let prevViewProj = mat4.create();
    mat4.mul(prevViewProj, proj, camera.viewMatrix());

    let displayMode = 0;
    let mouseCaptured = false;
    let lastTime = performance.now();
    let frameCounter = 0;
    let fpsAccum = 0;
    let displayFPS = 0;

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
    document.addEventListener('keyup', (e) => {
        camera.keys[e.code] = false;
    });
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

        // 1) Scene pass → MRT FBO
        renderer.renderScene(model, viewProj, cube.prevModel, prevViewProj);

        // 2) Read motion vectors → JS particle warp → upload noise
        const motion = renderer.readMotion();
        const warpedNoise = warper.step(motion);
        renderer.uploadNoise(noiseToRGBA(warpedNoise, HEIGHT, WIDTH, CHANNELS));

        // 3) Display pass → canvas
        renderer.display(displayMode);

        prevViewProj = viewProj;

        // FPS + validation stats
        frameCounter++;
        fpsAccum += dt;
        if (fpsAccum >= 1.0) {
            displayFPS = Math.round(frameCounter / fpsAccum);
            frameCounter = 0;
            fpsAccum = 0;

            const { mean, std } = warper.stats();
            statsEl.textContent = `FPS: ${displayFPS} | mean: ${mean.toFixed(3)} | std: ${std.toFixed(3)} | mode: ${displayMode}`;

            // Validation: log warning if stats drift too far from N(0,1)
            if (Math.abs(mean) > 0.1 || Math.abs(std - 1.0) > 0.2) {
                console.warn(`Noise stats drifting: mean=${mean.toFixed(4)}, std=${std.toFixed(4)}`);
            }
        }

        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

main();
