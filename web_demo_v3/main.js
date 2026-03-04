/**
 * Entry point: game loop, module wiring, input routing.
 * V3: Physics sandbox with domino chain + sphere shooting + noise warp.
 */

import { WebGPURenderer } from './renderer.js';
import { PhysicsWorld } from './physics.js';
import { FPSCamera, SceneManager } from './scene.js';
import { UIManager } from './ui.js';

const { mat4, glMatrix } = window.glMatrix;

async function main() {
    const canvas = document.getElementById('canvas');
    const controlsEl = document.getElementById('controls');
    const errorEl = document.getElementById('error');

    if (!navigator.gpu) {
        errorEl.textContent = 'WebGPU not available. Try Chrome 113+ or enable chrome://flags/#enable-webgpu';
        return;
    }

    // --- Physics ---
    const physics = new PhysicsWorld();
    try {
        await physics.init();
    } catch (e) {
        errorEl.textContent = 'Failed to load Rapier physics: ' + e.message;
        throw e;
    }

    // --- Camera & Scene ---
    const camera = new FPSCamera();
    const scene = new SceneManager();

    // --- Renderer ---
    let renderer = null;
    let frameSeed = 42;

    // --- UI ---
    const ui = new UIManager({
        onResolutionChange: async () => {
            ui.updateCanvas(canvas);
            await createRenderer();
        },
        onRendererUpdate: () => {
            if (renderer) {
                ui.applyToRenderer(renderer);
                ui.updateCanvas(canvas);
            }
        },
        onResetScene: () => { physics.reset(); scene.prevTransforms.clear(); },
        onSlowMo: (on) => { physics.slowMo = on; },
        onSetTime: (targetElapsed) => { startTime = performance.now() - targetElapsed * 1000; },
    });

    async function createRenderer() {
        if (renderer) renderer.destroy();
        renderer = new WebGPURenderer(canvas, ui.W, ui.H);
        await renderer.init();
        ui.applyToRenderer(renderer);
    }

    // Initial setup
    ui.updateCanvas(canvas);
    try {
        await createRenderer();
    } catch (e) {
        errorEl.textContent = e.message;
        throw e;
    }

    document.getElementById('stats').textContent = 'Ready. Click to start.';

    // --- Projection matrix ---
    let proj = mat4.create();
    mat4.perspective(proj, glMatrix.toRadian(70), 1.0, 0.1, 100000);

    // --- Prev viewProj for motion vectors ---
    const eyePos = physics.getPlayerEyePos();
    let prevViewProj = mat4.create();
    mat4.mul(prevViewProj, proj, camera.viewMatrix(eyePos));

    // --- Input ---
    let mouseCaptured = false;

    document.addEventListener('keydown', (e) => {
        camera.keys[e.code] = true;
        if (e.code === 'Escape' && mouseCaptured) {
            document.exitPointerLock();
        }
        // Jump
        if (e.code === 'Space' && mouseCaptured) {
            e.preventDefault();
            physics.jump();
        }
    });
    document.addEventListener('keyup', (e) => { camera.keys[e.code] = false; });

    canvas.addEventListener('click', () => {
        if (!mouseCaptured) {
            canvas.requestPointerLock();
        }
    });

    // Shoot on mousedown (only when captured)
    // Use document-level listener: pointer lock delivers events to document in some browsers
    document.addEventListener('mousedown', (e) => {
        if (mouseCaptured && e.button === 0) {
            const eye = physics.getPlayerEyePos();
            const [dx, dy, dz] = camera.forward();
            physics.shoot(eye, dx, dy, dz);
        }
    });

    document.addEventListener('pointerlockchange', () => {
        mouseCaptured = document.pointerLockElement === canvas;
        controlsEl.style.display = mouseCaptured ? 'none' : 'block';
    });

    document.addEventListener('mousemove', (e) => {
        if (mouseCaptured) camera.processMouse(e.movementX, e.movementY);
    });

    // --- FPS tracking ---
    let frameCounter = 0;
    let fpsAccum = 0;
    let displayFPS = 0;
    let startTime = performance.now();
    let lastTime = performance.now();
    const cpuFrameHistory = [];
    let physicsMs = 0;

    // --- Game loop ---
    function frame(now) {
        if (!renderer) { requestAnimationFrame(frame); return; }

        const dt = Math.min((now - lastTime) / 1000, 0.1);
        lastTime = now;
        const elapsedSecs = (now - startTime) / 1000;

        // Physics step
        const physStart = performance.now();
        physics.step();
        physicsMs = performance.now() - physStart;

        // Player movement
        const [fx, fz] = camera.forwardXZ();
        const [rx, rz] = camera.rightXZ();
        physics.movePlayer(fx, fz, rx, rz, camera.keys);

        // Camera follows player
        const eyePos = physics.getPlayerEyePos();
        const viewProj = mat4.create();
        mat4.mul(viewProj, proj, camera.viewMatrix(eyePos));

        // Inverse viewProj for sky ray reconstruction
        const invViewProj = mat4.create();
        mat4.invert(invViewProj, viewProj);

        // Build instance buffer
        const sceneData = physics.getSceneData();
        const { numBoxInstances, numSphereInstances, numDominoInstances } = scene.buildInstances(sceneData);
        const instanceData = scene.getActiveData();

        // Collect point lights from glowing objects (spheres + mushroom caps)
        const lights = [];
        for (const s of sceneData.spheres) {
            if (lights.length >= 32) break;
            lights.push({ pos: [s.pos.x, s.pos.y, s.pos.z], color: [0.3, 0.5, 1.0], intensity: 3.0, radius: 15.0 });
        }
        if (sceneData.mushrooms) {
            for (let i = 0; i < sceneData.mushrooms.length; i++) {
                if (lights.length >= 32) break;
                const m = sceneData.mushrooms[i];
                // Only cap pieces glow (every other mushroom part is cap vs stem)
                if (i % 2 === 1) {
                    lights.push({ pos: [m.pos.x, m.pos.y, m.pos.z], color: [0.2, 0.8, 0.4], intensity: 2.0, radius: 10.0 });
                }
            }
        }

        // Render
        const cpuStart = performance.now();
        const fwd = camera.forward();
        renderer.frame({
            viewProj,
            prevViewProj,
            invViewProj,
            instanceData,
            numBoxInstances,
            numSphereInstances,
            numDominoInstances,
            displayMode: ui.displayMode,
            frameSeed,
            elapsedSecs,
            eyePos: [eyePos.x, eyePos.y, eyePos.z],
            eyeDir: [fwd[0], fwd[1], fwd[2]],
            lights,
        });
        const cpuMs = performance.now() - cpuStart;
        cpuFrameHistory.push(cpuMs);
        if (cpuFrameHistory.length > 200) cpuFrameHistory.shift();

        prevViewProj = viewProj;
        frameSeed++;

        // FPS update (1x/sec)
        frameCounter++;
        fpsAccum += dt;
        if (fpsAccum >= 1.0) {
            displayFPS = Math.round(frameCounter / fpsAccum);
            frameCounter = 0;
            fpsAccum = 0;
            ui.updateStats(displayFPS, renderer.noiseStats, renderer.getTimingStats(), cpuFrameHistory, physicsMs);
        }

        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

main();
