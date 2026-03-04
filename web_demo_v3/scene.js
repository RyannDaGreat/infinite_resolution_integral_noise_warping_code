/**
 * Scene management: FPS camera, instance buffer construction, transform sync.
 * Bridges physics → rendering by building per-instance GPU data each frame.
 */

const { mat4, vec3, quat, glMatrix } = window.glMatrix;

// Instance layout: model (16f) + prevModel (16f) + color (4f) = 36 floats = 144 bytes
const FLOATS_PER_INSTANCE = 36;
const MAX_INSTANCES = 32768;  // increased from 8192 to support ~22k visual terrain columns at 1.0m spacing

// ---------------------------------------------------------------------------
// FPS Camera
// ---------------------------------------------------------------------------

export class FPSCamera {
    constructor() {
        this.yaw = -90;
        this.pitch = -10;
        this.sensitivity = 0.15;
        this.keys = {};
    }

    processMouse(dx, dy) {
        this.yaw += dx * this.sensitivity;
        this.pitch -= dy * this.sensitivity;
        this.pitch = Math.max(-89, Math.min(89, this.pitch));
    }

    /**
     * Pure. Returns the forward direction vector from yaw/pitch.
     *
     * >>> // Returns [x, y, z] normalized
     */
    forward() {
        const ry = glMatrix.toRadian(this.yaw);
        const rp = glMatrix.toRadian(this.pitch);
        return [
            Math.cos(rp) * Math.cos(ry),
            Math.sin(rp),
            Math.cos(rp) * Math.sin(ry),
        ];
    }

    /**
     * Pure. Returns horizontal-only forward (for movement, ignoring pitch).
     */
    forwardXZ() {
        const ry = glMatrix.toRadian(this.yaw);
        const len = Math.hypot(Math.cos(ry), Math.sin(ry));
        return [Math.cos(ry) / len, Math.sin(ry) / len];
    }

    /**
     * Pure. Returns right direction vector (horizontal).
     */
    rightXZ() {
        const [fx, fz] = this.forwardXZ();
        return [-fz, fx];  // cross(forward, up) projected to XZ
    }

    /**
     * Pure. Build view matrix from eye position + look direction.
     */
    viewMatrix(eyePos) {
        const f = this.forward();
        const target = [eyePos.x + f[0], eyePos.y + f[1], eyePos.z + f[2]];
        const out = mat4.create();
        mat4.lookAt(out, [eyePos.x, eyePos.y, eyePos.z], target, [0, 1, 0]);
        return out;
    }
}

// ---------------------------------------------------------------------------
// Scene Manager
// ---------------------------------------------------------------------------

export class SceneManager {
    constructor() {
        this.instanceData = new Float32Array(MAX_INSTANCES * FLOATS_PER_INSTANCE);
        this.prevTransforms = new Map();  // bodyId → mat4 (previous frame)
        this.numBoxInstances = 0;
        this.numSphereInstances = 0;
        this.numDominoInstances = 0;  // dominoes only (for separate draw call with beveled VB)
    }

    /**
     * Build instance buffer from physics scene data.
     * Not pure: mutates instanceData and prevTransforms.
     *
     * @param {object} sceneData - from PhysicsWorld.getSceneData()
     * @returns {object} { numBoxInstances, numSphereInstances }
     */
    buildInstances(sceneData) {
        const {
            floor, dominoes, dominoHalf, spheres, sphereRadius,
            mazeWalls, mazeChest, mazeIvy,
            towerPlatforms, towerRamps, towerFlag,
            mmStructure, mmWheels, mmChains,
            terrainBlocks, trees, shrubs, mushrooms,
            signposts, fence,
        } = sceneData;
        let boxIdx = 0;

        // Floor instance (box 0)
        const floorModel = this._makeModel(floor.pos, floor.rot, [floor.half.x, floor.half.y, floor.half.z]);
        this._writeInstance(boxIdx++, floorModel, 'floor', [0.35, 0.35, 0.4, 1.0]);

        // Domino instances
        for (let i = 0; i < dominoes.length; i++) {
            const d = dominoes[i];
            const model = this._makeModel(d.pos, d.rot, [dominoHalf.x, dominoHalf.y, dominoHalf.z]);
            this._writeInstance(boxIdx++, model, `domino_${i}`, [0.92, 0.90, 0.85, 1.0]);
        }

        this.numDominoInstances = dominoes.length;

        // Maze wall instances: gray-green stone color (detected in shader via color)
        if (mazeWalls) {
            for (let i = 0; i < mazeWalls.length; i++) {
                const w = mazeWalls[i];
                const model = this._makeModel(w.pos, w.rot, w.half);
                // Color signals "maze wall" to the shader: R=0.45, G=0.44, B=0.42, A=1
                this._writeInstance(boxIdx++, model, `maze_${i}`, [0.45, 0.44, 0.42, 1.0]);
            }
        }

        // Treasure chest instance: gold/brown color
        if (mazeChest) {
            const c = mazeChest;
            const model = this._makeModel(c.pos, c.rot, [c.half.x, c.half.y, c.half.z]);
            // Color signals "chest" to shader: warm gold R=0.85, G=0.65, B=0.2
            this._writeInstance(boxIdx++, model, 'maze_chest', [0.85, 0.65, 0.20, 1.0]);
        }

        // Maze ivy: render-only leaf clusters and vine tendrils (no physics body)
        if (mazeIvy) {
            for (let i = 0; i < mazeIvy.length; i++) {
                const iv = mazeIvy[i];
                const model = this._makeModel(iv.pos, iv.rot, iv.half);
                this._writeInstance(boxIdx++, model, `ivy_${i}`, iv.color);
            }
        }

        // Tower platform instances
        if (towerPlatforms) {
            for (let i = 0; i < towerPlatforms.length; i++) {
                const p = towerPlatforms[i];
                const model = this._makeModel(p.pos, p.rot, p.half);
                this._writeInstance(boxIdx++, model, `tower_plat_${i}`, p.color);
            }
        }

        // Tower ramp instances
        if (towerRamps) {
            for (let i = 0; i < towerRamps.length; i++) {
                const r = towerRamps[i];
                const model = this._makeModel(r.pos, r.rot, r.half);
                this._writeInstance(boxIdx++, model, `tower_ramp_${i}`, r.color);
            }
        }

        // Tower flag instances (pole + flag)
        if (towerFlag) {
            for (let i = 0; i < towerFlag.length; i++) {
                const f = towerFlag[i];
                const model = this._makeModel(f.pos, f.rot, f.half);
                this._writeInstance(boxIdx++, model, `tower_flag_${i}`, f.color);
            }
        }

        // Marble machine structure (fixed boxes: platforms, stairs, ramps, gutters)
        if (mmStructure) {
            for (let i = 0; i < mmStructure.length; i++) {
                const s = mmStructure[i];
                const model = this._makeModel(s.pos, s.rot, s.half);
                this._writeInstance(boxIdx++, model, `mm_struct_${i}`, s.color);
            }
        }

        // Marble machine spinning wheels (dynamic)
        if (mmWheels) {
            for (let i = 0; i < mmWheels.length; i++) {
                const w = mmWheels[i];
                const model = this._makeModel(w.pos, w.rot, w.half);
                this._writeInstance(boxIdx++, model, `mm_wheel_${i}`, w.color);
            }
        }

        // Marble machine dangling chains (dynamic)
        if (mmChains) {
            for (let i = 0; i < mmChains.length; i++) {
                const c = mmChains[i];
                const model = this._makeModel(c.pos, c.rot, c.half);
                this._writeInstance(boxIdx++, model, `mm_chain_${i}`, c.color);
            }
        }

        // Forest terrain columns
        if (terrainBlocks) {
            for (let i = 0; i < terrainBlocks.length; i++) {
                const b = terrainBlocks[i];
                const model = this._makeModel(b.pos, b.rot, b.half);
                this._writeInstance(boxIdx++, model, `terrain_${i}`, b.color);
            }
        }

        // Forest trees (trunks + canopies)
        if (trees) {
            for (let i = 0; i < trees.length; i++) {
                const t = trees[i];
                const model = this._makeModel(t.pos, t.rot, t.half);
                this._writeInstance(boxIdx++, model, `tree_${i}`, t.color);
            }
        }

        // Forest shrubs
        if (shrubs) {
            for (let i = 0; i < shrubs.length; i++) {
                const s = shrubs[i];
                const model = this._makeModel(s.pos, s.rot, s.half);
                this._writeInstance(boxIdx++, model, `shrub_${i}`, s.color);
            }
        }

        // Forest mushrooms
        if (mushrooms) {
            for (let i = 0; i < mushrooms.length; i++) {
                const m = mushrooms[i];
                const model = this._makeModel(m.pos, m.rot, m.half);
                this._writeInstance(boxIdx++, model, `mushroom_${i}`, m.color);
            }
        }

        // Area signposts (poles + boards)
        if (signposts) {
            for (let i = 0; i < signposts.length; i++) {
                const s = signposts[i];
                const model = this._makeModel(s.pos, s.rot, s.half);
                this._writeInstance(boxIdx++, model, `sign_${i}`, s.color);
            }
        }

        // Perimeter fence (render-only: posts + rails)
        if (fence) {
            for (let i = 0; i < fence.length; i++) {
                const f = fence[i];
                const model = this._makeModel(f.pos, f.rot, f.half);
                this._writeInstance(boxIdx++, model, `fence_${i}`, f.color);
            }
        }

        this.numBoxInstances = boxIdx;

        // Sphere instances (after all boxes)
        let sphereIdx = boxIdx;
        for (let i = 0; i < spheres.length; i++) {
            const s = spheres[i];
            const model = this._makeModel(s.pos, s.rot, [sphereRadius, sphereRadius, sphereRadius]);
            this._writeInstance(sphereIdx++, model, `sphere_${i}`, [0.2, 0.4, 0.95, 1.0]);
        }

        this.numSphereInstances = sphereIdx - boxIdx;

        return {
            numBoxInstances: this.numBoxInstances,
            numSphereInstances: this.numSphereInstances,
            numDominoInstances: this.numDominoInstances,
        };
    }

    /**
     * Build a mat4 from position + quaternion + half-extents scale.
     * Pure. Multiplies halfExtents by 2 because geometry primitives have unit
     * half-extent 0.5 (box at ±0.5, sphere radius 0.5).
     */
    _makeModel(pos, rot, halfExtents) {
        const m = mat4.create();
        const q = quat.fromValues(rot.x, rot.y, rot.z, rot.w);
        mat4.fromRotationTranslationScale(m, q, [pos.x, pos.y, pos.z],
            [halfExtents[0] * 2, halfExtents[1] * 2, halfExtents[2] * 2]);
        return m;
    }

    /**
     * Write one instance (model + prevModel + color) into the instance data array.
     * Not pure: mutates instanceData and prevTransforms.
     */
    _writeInstance(idx, model, id, color) {
        const offset = idx * FLOATS_PER_INSTANCE;

        // Current model
        this.instanceData.set(model, offset);

        // Previous model (or current if first frame)
        const prev = this.prevTransforms.get(id);
        if (prev) {
            this.instanceData.set(prev, offset + 16);
        } else {
            this.instanceData.set(model, offset + 16);
        }

        // Color
        this.instanceData[offset + 32] = color[0];
        this.instanceData[offset + 33] = color[1];
        this.instanceData[offset + 34] = color[2];
        this.instanceData[offset + 35] = color[3];

        // Store for next frame
        this.prevTransforms.set(id, new Float32Array(model));
    }

    /**
     * Get the subarray of instance data actually in use.
     * Pure.
     */
    getActiveData() {
        const total = this.numBoxInstances + this.numSphereInstances;
        return this.instanceData.subarray(0, total * FLOATS_PER_INSTANCE);
    }
}

// Reserved instance slot for the terrain mesh (static, written once by renderer).
// Kept far from the dynamic instances so it never overlaps.
const TERRAIN_INSTANCE_IDX = MAX_INSTANCES - 1;

export { MAX_INSTANCES, FLOATS_PER_INSTANCE, TERRAIN_INSTANCE_IDX };
