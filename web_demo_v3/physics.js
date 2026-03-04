/**
 * Rapier physics: world, floor, dominoes, player body, sphere shooting.
 * All physics state lives here. Rendering knows nothing about Rapier.
 */

const GRAVITY = { x: 0, y: -25, z: 0 };
const DOMINO_HALF = { x: 0.75, y: 1.5, z: 0.18 };  // wide × tall × thin (3x scale)
const DOMINO_SPACING = 2.5;  // along Z (wide gaps for 3x-scale dominoes)
const DOMINO_FRICTION = 0.4;
const DOMINO_RESTITUTION = 0.1;
const FLOOR_HALF = { x: 200, y: 0.05, z: 200 };
const PLAYER_HALF_HEIGHT = 0.5;
const PLAYER_RADIUS = 0.3;
const PLAYER_EYE_OFFSET = 0.7;
const JUMP_IMPULSE = 6;
const MOVE_SPEED = 5;
const SPRINT_SPEED = 12;
const SPHERE_RADIUS = 0.3;
const SPHERE_DENSITY = 30;
const SPHERE_SPEED = 15;
const MAX_SPHERES = 20;

let RAPIER = null;

export class PhysicsWorld {
    constructor() {
        this.world = null;
        this.floor = null;
        this.dominoes = [];
        this.spheres = [];       // { body, collider }
        this.playerBody = null;
        this.playerCollider = null;
        this.slowMo = false;
    }

    /**
     * Load Rapier WASM and create the physics world.
     * Not pure: loads WASM, creates global physics state.
     */
    async init() {
        RAPIER = await import('https://cdn.jsdelivr.net/npm/@dimforge/rapier3d-compat/+esm');
        await RAPIER.init();

        this.world = new RAPIER.World(GRAVITY);
        this._createFloor();
        this._createDominoes(60);
        this._createPlayer();
    }

    _createFloor() {
        const rbDesc = RAPIER.RigidBodyDesc.fixed()
            .setTranslation(0, -FLOOR_HALF.y, 0);
        this.floor = this.world.createRigidBody(rbDesc);
        const colDesc = RAPIER.ColliderDesc.cuboid(FLOOR_HALF.x, FLOOR_HALF.y, FLOOR_HALF.z)
            .setFriction(0.5)
            .setRestitution(0.1);
        this.world.createCollider(colDesc, this.floor);
    }

    _createDominoes(count) {
        for (let i = 0; i < count; i++) {
            const z = -i * DOMINO_SPACING;
            const rbDesc = RAPIER.RigidBodyDesc.dynamic()
                .setTranslation(0, DOMINO_HALF.y, z);
            const rb = this.world.createRigidBody(rbDesc);
            const colDesc = RAPIER.ColliderDesc.cuboid(DOMINO_HALF.x, DOMINO_HALF.y, DOMINO_HALF.z)
                .setFriction(DOMINO_FRICTION)
                .setRestitution(DOMINO_RESTITUTION)
                .setDensity(1.0);
            this.world.createCollider(colDesc, rb);
            this.dominoes.push(rb);
        }
    }

    _createPlayer() {
        const rbDesc = RAPIER.RigidBodyDesc.dynamic()
            .setTranslation(0, PLAYER_HALF_HEIGHT + PLAYER_RADIUS + 0.1, 6)
            .lockRotations()
            .setLinearDamping(0.5);
        this.playerBody = this.world.createRigidBody(rbDesc);
        const colDesc = RAPIER.ColliderDesc.capsule(PLAYER_HALF_HEIGHT, PLAYER_RADIUS)
            .setFriction(0.0)
            .setRestitution(0.0);
        this.playerCollider = this.world.createCollider(colDesc, this.playerBody);
    }

    /**
     * Get player eye position (body center + eye offset).
     * Pure.
     *
     * >>> // Returns {x, y, z}
     */
    getPlayerEyePos() {
        const t = this.playerBody.translation();
        return { x: t.x, y: t.y + PLAYER_EYE_OFFSET, z: t.z };
    }

    /**
     * Apply movement to player body based on input direction.
     * Not pure: mutates player body velocity.
     *
     * @param {number} forwardX - forward direction X component
     * @param {number} forwardZ - forward direction Z component
     * @param {number} rightX - right direction X component
     * @param {number} rightZ - right direction Z component
     * @param {object} keys - pressed key map
     */
    movePlayer(forwardX, forwardZ, rightX, rightZ, keys) {
        const vel = this.playerBody.linvel();
        const speed = (keys['ShiftLeft'] || keys['ShiftRight']) ? SPRINT_SPEED : MOVE_SPEED;
        let vx = 0, vz = 0;

        if (keys['KeyW'] || keys['ArrowUp'])    { vx += forwardX * speed; vz += forwardZ * speed; }
        if (keys['KeyS'] || keys['ArrowDown'])  { vx -= forwardX * speed; vz -= forwardZ * speed; }
        if (keys['KeyA'] || keys['ArrowLeft'])   { vx -= rightX * speed; vz -= rightZ * speed; }
        if (keys['KeyD'] || keys['ArrowRight']) { vx += rightX * speed; vz += rightZ * speed; }

        this.playerBody.setLinvel({ x: vx, y: vel.y, z: vz }, true);
    }

    /**
     * Apply jump impulse if grounded.
     * Not pure: mutates player body, casts ray into world.
     */
    jump() {
        const t = this.playerBody.translation();
        const ray = new RAPIER.Ray(
            { x: t.x, y: t.y, z: t.z },
            { x: 0, y: -1, z: 0 }
        );
        const hit = this.world.castRay(ray, PLAYER_HALF_HEIGHT + PLAYER_RADIUS + 0.15, true);
        if (hit) {
            this.playerBody.applyImpulse({ x: 0, y: JUMP_IMPULSE, z: 0 }, true);
        }
    }

    /**
     * Spawn a projectile sphere at pos flying in dir.
     * Not pure: creates rigid body in world, may destroy oldest sphere.
     */
    shoot(pos, dirX, dirY, dirZ) {
        // Recycle oldest if at max
        if (this.spheres.length >= MAX_SPHERES) {
            const old = this.spheres.shift();
            this.world.removeRigidBody(old.body);
        }

        // Spawn 1.5 units ahead of eye to clear player capsule
        const rbDesc = RAPIER.RigidBodyDesc.dynamic()
            .setTranslation(pos.x + dirX * 1.5, pos.y + dirY * 1.5, pos.z + dirZ * 1.5)
            .setLinearDamping(0.3)
            .setAngularDamping(0.3);
        const rb = this.world.createRigidBody(rbDesc);
        rb.setLinvel({ x: dirX * SPHERE_SPEED, y: dirY * SPHERE_SPEED, z: dirZ * SPHERE_SPEED }, true);
        const colDesc = RAPIER.ColliderDesc.ball(SPHERE_RADIUS)
            .setDensity(SPHERE_DENSITY)
            .setFriction(0.3)
            .setRestitution(0.5);
        this.world.createCollider(colDesc, rb);
        this.spheres.push({ body: rb });
    }

    /**
     * Step the physics simulation.
     * Not pure: advances world state.
     */
    step() {
        const dt = this.slowMo ? 1/240 : 1/60;
        this.world.timestep = dt;
        this.world.step();
    }

    /**
     * Reset scene: remove all dominoes and spheres, recreate them.
     * Not pure: destroys and recreates rigid bodies.
     */
    reset() {
        // Remove spheres
        for (const s of this.spheres) {
            this.world.removeRigidBody(s.body);
        }
        this.spheres = [];

        // Remove dominoes
        for (const d of this.dominoes) {
            this.world.removeRigidBody(d);
        }
        this.dominoes = [];

        // Recreate
        this._createDominoes(60);

        // Reset player position
        this.playerBody.setTranslation({ x: 0, y: PLAYER_HALF_HEIGHT + PLAYER_RADIUS + 0.1, z: 6 }, true);
        this.playerBody.setLinvel({ x: 0, y: 0, z: 0 }, true);
    }

    /**
     * Get all renderable bodies as arrays of {position, rotation, halfExtents/radius, type}.
     * Pure (reads physics state but doesn't mutate).
     *
     * Returns:
     *     object: { floor, dominoes, spheres } each with position/rotation data
     */
    getSceneData() {
        const floorPos = this.floor.translation();
        const floorRot = this.floor.rotation();

        const dominoData = this.dominoes.map(d => ({
            pos: d.translation(),
            rot: d.rotation(),
        }));

        const sphereData = this.spheres.map(s => ({
            pos: s.body.translation(),
            rot: s.body.rotation(),
        }));

        return {
            floor: { pos: floorPos, rot: floorRot, half: FLOOR_HALF },
            dominoes: dominoData,
            dominoHalf: DOMINO_HALF,
            spheres: sphereData,
            sphereRadius: SPHERE_RADIUS,
        };
    }
}

export { PLAYER_EYE_OFFSET };
