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
const MAX_SPHERES = 1000;

// Forest / Wilds constants
const FOREST_CENTER_X = 0;
const FOREST_CENTER_Z = 200;      // far north
const FOREST_GRID_SPACING = 1.0;  // visual terrain grid step (world units) — 1.0m for smoother look
const FOREST_HALF_EXTENT = 75;    // half-size of terrain grid (150×150 total)
const FOREST_AMPLITUDE = 40;      // max terrain height
const FOREST_FREQ = 0.025;        // base noise frequency
const FOREST_SEED = 31337;        // deterministic seed

// Signpost pole/sign dimensions
const SIGN_POLE_HALF  = [0.075, 2.5, 0.075];   // tall thin pole
const SIGN_BOARD_HALF = [2.0,   1.0, 0.08];     // large flat board
const SIGN_POLE_COLOR = [0.55, 0.35, 0.15, 1.0]; // wood brown

// Per-area sign colors
const SIGN_COLORS = {
    domino:      [0.92, 0.90, 0.85, 1.0],  // ivory/cream
    labyrinth:   [0.50, 0.55, 0.45, 1.0],  // stone gray-green
    contraption: [0.30, 0.40, 0.70, 1.0],  // metallic blue
    skySteps:    [0.76, 0.65, 0.50, 1.0],  // warm sandstone
    wilds:       [0.20, 0.50, 0.20, 1.0],  // forest green
};

// Maze constants
const MAZE_ROWS = 15;
const MAZE_COLS = 15;
const MAZE_CELL_SIZE = 4.0;       // world units per cell
const MAZE_WALL_THICKNESS = 0.5;  // wall thickness
const MAZE_WALL_HEIGHT = 3.0;     // wall height
const MAZE_CENTER_X = -100;       // world X center of maze
const MAZE_CENTER_Z = 0;          // world Z center of maze

// Maze wall half-extents (used for both colliders and rendering)
const MAZE_WALL_HALF_Y = MAZE_WALL_HEIGHT / 2;
const MAZE_WALL_HALF_THICK = MAZE_WALL_THICKNESS / 2;

// Treasure chest half-extents
const CHEST_HALF = { x: 0.6, y: 0.5, z: 0.4 };

// --- Marble machine constants ---
const MM_CX = 100;
const MM_CZ = 0;
const MM_PLATFORM_HEIGHTS = [3, 6, 9];
const MM_PLATFORM_HX = 10;
const MM_PLATFORM_HY = 0.15;
const MM_PLATFORM_HZ = 8;
const MM_STEP_HH = 0.15;       // step half-height (rise = 0.3)
const MM_STEP_HD = 0.75;       // step half-depth (run = 1.5)
const MM_STEP_HW = 1.5;        // step half-width
const MM_RAILING_HH = 0.6;
const MM_RAILING_THICK = 0.08;
const MM_RAMP_HW = 1.25;
const MM_RAMP_THICK = 0.12;
const MM_GUTTER_HH = 0.15;
const MM_GUTTER_THICK = 0.06;
const MM_WHEEL_R = 1.2;
const MM_WHEEL_THICK = 0.15;
const MM_CHAIN_SEG = { x: 0.08, y: 0.25, z: 0.08 };
const MM_CHAIN_LEN = 6;

// Platform tower constants
const TOWER_CENTER_X = 0;
const TOWER_CENTER_Z = -200;
const TOWER_TARGET_HEIGHT = 40;
const TOWER_PLATFORM_FRICTION = 0.8;
const TOWER_RAMP_FRICTION = 0.7;

// ---------------------------------------------------------------------------
// Terrain noise helpers (seeded, deterministic, pure)
// ---------------------------------------------------------------------------

/**
 * Pure. Seeded Wang hash for integer grid coordinates → [0, 1].
 *
 * Args:
 *     a (number): Integer X index
 *     b (number): Integer Z index
 *     seed (number): Integer seed
 *
 * Returns:
 *     number: Uniform value in [0, 1]
 *
 * Examples:
 *     >>> hashCell(3, 7, 42) >= 0 && hashCell(3, 7, 42) <= 1
 *     true
 */
function hashCell(a, b, seed) {
    let s = (seed ^ (a * 1619 + b * 31337)) | 0;
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b) | 0;
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b) | 0;
    s = s ^ (s >>> 16);
    return (s >>> 0) / 0xFFFFFFFF;
}

/**
 * Pure. 2D bilinear value noise at (x, z) with given seed.
 * Uses smoothstep for C1-continuous interpolation.
 *
 * Args:
 *     x (number): Noise coordinate X
 *     z (number): Noise coordinate Z
 *     seed (number): Integer seed for this octave
 *
 * Returns:
 *     number: Smoothly interpolated value in [0, 1]
 *
 * Examples:
 *     >>> valueNoise2D(0.5, 1.5, 42) >= 0 && valueNoise2D(0.5, 1.5, 42) <= 1
 *     true
 */
function valueNoise2D(x, z, seed) {
    const ix = Math.floor(x), iz = Math.floor(z);
    const fx = x - ix, fz = z - iz;
    const ux = fx * fx * (3 - 2 * fx);
    const uz = fz * fz * (3 - 2 * fz);
    const v00 = hashCell(ix,     iz,     seed);
    const v10 = hashCell(ix + 1, iz,     seed);
    const v01 = hashCell(ix,     iz + 1, seed);
    const v11 = hashCell(ix + 1, iz + 1, seed);
    return v00 + ux * (v10 - v00) + uz * (v01 - v00) + ux * uz * (v00 - v10 - v01 + v11);
}

/**
 * Pure. Fractal Brownian Motion: layered value noise at increasing frequencies.
 * Returns a value in [0, 1] (normalized by total amplitude).
 *
 * Args:
 *     x (number): Input X
 *     z (number): Input Z
 *     seed (number): Base seed (each octave offset by i*7919)
 *     octaves (number): Number of noise octaves (default 6)
 *     lacunarity (number): Frequency multiplier per octave (default 2.0)
 *     gain (number): Amplitude multiplier per octave (default 0.5)
 *
 * Returns:
 *     number: FBM value in [0, 1]
 *
 * Examples:
 *     >>> fbm2D(1.0, 2.0, FOREST_SEED) >= 0 && fbm2D(1.0, 2.0, FOREST_SEED) <= 1
 *     true
 */
function fbm2D(x, z, seed, octaves = 6, lacunarity = 2.0, gain = 0.5) {
    let value = 0, amplitude = 1.0, totalAmp = 0, freq = 1.0;
    for (let i = 0; i < octaves; i++) {
        value += amplitude * valueNoise2D(x * freq, z * freq, seed + i * 7919);
        totalAmp += amplitude;
        amplitude *= gain;
        freq *= lacunarity;
    }
    return value / totalAmp;
}

/**
 * Pure. Terrain height at world (wx, wz). Fades to zero at forest boundaries.
 *
 * Args:
 *     wx (number): World X coordinate
 *     wz (number): World Z coordinate
 *
 * Returns:
 *     number: Height >= 0 in world units
 *
 * Examples:
 *     >>> terrainHeight(FOREST_CENTER_X, FOREST_CENTER_Z) >= 0
 *     true
 */
function terrainHeight(wx, wz) {
    const lx = wx - FOREST_CENTER_X;
    const lz = wz - FOREST_CENTER_Z;
    const raw = fbm2D(lx * FOREST_FREQ, lz * FOREST_FREQ, FOREST_SEED);
    const shaped = Math.pow(raw, 1.4);  // flatten lowlands, steepen peaks
    const h = shaped * FOREST_AMPLITUDE;
    // Boundary fades: XZ edge fade + south entrance fade
    const edgeFadeX = Math.max(0, 1 - Math.abs(lx) / (FOREST_HALF_EXTENT * 0.85));
    const edgeFadeZ = Math.max(0, 1 - Math.max(0, -lz) / 40);
    const fade = Math.pow(Math.min(edgeFadeX, edgeFadeZ), 1.5);
    return h * fade;
}

/**
 * Pure. Terrain slope at (wx, wz): max height diff across cardinal neighbors.
 *
 * Args:
 *     wx (number): World X
 *     wz (number): World Z
 *     step (number): Sample step (default: FOREST_GRID_SPACING)
 *
 * Returns:
 *     number: Max slope magnitude in height units per step
 *
 * Examples:
 *     >>> terrainSlope(FOREST_CENTER_X, FOREST_CENTER_Z) >= 0
 *     true
 */
function terrainSlope(wx, wz, step = FOREST_GRID_SPACING) {
    const h  = terrainHeight(wx, wz);
    const hx = terrainHeight(wx + step, wz);
    const hz = terrainHeight(wx, wz + step);
    return Math.max(Math.abs(hx - h), Math.abs(hz - h));
}

/**
 * Pure. Terrain biome color based on height and slope.
 *
 * Biomes (in priority order):
 *   - steep (slope > 4): rock gray
 *   - high (> 25): snow white
 *   - mid (10–25): forest floor dark green
 *   - low flat: grass green
 *
 * Args:
 *     height (number): Terrain height
 *     slope (number): Terrain slope
 *
 * Returns:
 *     number[4]: [r, g, b, 1.0]
 *
 * Examples:
 *     >>> biomeColor(5, 0.5)
 *     [0.3, 0.5, 0.2, 1.0]
 */
function biomeColor(height, slope) {
    if (slope > 4.0) return [0.5,  0.5,  0.48, 1.0];
    if (height > 25) return [0.9,  0.9,  0.95, 1.0];
    if (height > 10) return [0.2,  0.4,  0.15, 1.0];
    return                  [0.3,  0.5,  0.2,  1.0];
}

/**
 * Pure. Generate a maze grid using recursive backtracker (DFS).
 * Returns wall presence arrays for horizontal and vertical walls.
 *
 * The grid has `rows` x `cols` cells. Walls exist between adjacent cells.
 * - horizontal[r][c]: wall on the top edge of row r (between rows r-1 and r)
 *   Array size: (rows+1) x cols
 * - vertical[r][c]: wall on the left edge of col c (between cols c-1 and c)
 *   Array size: rows x (cols+1)
 *
 * Initially all walls are present. The DFS carves passages by removing walls
 * between the current cell and an unvisited neighbor.
 *
 * Args:
 *     rows (number): Number of cell rows
 *     cols (number): Number of cell columns
 *
 * Returns:
 *     object: { horizontal: boolean[][], vertical: boolean[][] }
 *
 * >>> const m = generateMaze(3, 3); m.horizontal.length
 * 4
 * >>> const m = generateMaze(3, 3); m.vertical[0].length
 * 4
 */
function generateMaze(rows, cols) {
    // Initialize all walls as present
    const horizontal = Array.from({ length: rows + 1 }, () => Array(cols).fill(true));
    const vertical = Array.from({ length: rows }, () => Array(cols + 1).fill(true));
    const visited = Array.from({ length: rows }, () => Array(cols).fill(false));

    // Seeded PRNG for deterministic maze
    let seed = 42;
    const rng = () => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
    };

    // Direction offsets: [dRow, dCol]
    const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];

    // Iterative DFS with explicit stack (avoids call stack overflow on large grids)
    const stack = [];
    const startRow = Math.floor(rows / 2);
    const startCol = Math.floor(cols / 2);
    visited[startRow][startCol] = true;
    stack.push([startRow, startCol]);

    while (stack.length > 0) {
        const [cr, cc] = stack[stack.length - 1];

        // Find unvisited neighbors
        const neighbors = [];
        for (const [dr, dc] of dirs) {
            const nr = cr + dr;
            const nc = cc + dc;
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !visited[nr][nc]) {
                neighbors.push([nr, nc, dr, dc]);
            }
        }

        if (neighbors.length === 0) {
            stack.pop();
            continue;
        }

        // Pick random unvisited neighbor
        const [nr, nc, dr, dc] = neighbors[Math.floor(rng() * neighbors.length)];
        visited[nr][nc] = true;

        // Remove wall between current and neighbor
        if (dr === -1) horizontal[cr][cc] = false;      // neighbor is above: remove top wall of current
        else if (dr === 1) horizontal[cr + 1][cc] = false;  // neighbor is below: remove bottom wall of current
        else if (dc === -1) vertical[cr][cc] = false;    // neighbor is left: remove left wall of current
        else if (dc === 1) vertical[cr][cc + 1] = false; // neighbor is right: remove right wall of current

        stack.push([nr, nc]);
    }

    return { horizontal, vertical };
}

/**
 * Pure. Seeded PRNG returning values in [0, 1).
 * Uses the same LCG as the maze generator.
 *
 * Args:
 *     seed (number): Initial seed value
 *
 * Returns:
 *     object: { next: () => number, range: (lo, hi) => number }
 *
 * >>> const rng = makeTowerRng(42); rng.next() >= 0 && rng.next() < 1
 * true
 */
function makeTowerRng(seed) {
    let s = seed;
    return {
        next() {
            s = (s * 1103515245 + 12345) & 0x7fffffff;
            return s / 0x7fffffff;
        },
        /** Pure. Random float in [lo, hi). */
        range(lo, hi) { return lo + this.next() * (hi - lo); },
    };
}

/**
 * Pure. Generate the tower path: a provably solvable sequence of waypoints
 * from ground level to TOWER_TARGET_HEIGHT, alternating between ramp sections
 * and jumping sections.
 *
 * Each consecutive pair of waypoints is reachable: jumping sections have
 * vertical gaps within jump height (~3 units conservative), ramp sections
 * are continuous surfaces.
 *
 * Returns an array of elements, each either:
 *   { type: 'platform', x, y, z, hw, hh, hd, color }
 *   { type: 'ramp', x, y, z, hw, hh, hd, angle, yaw, color }
 *   { type: 'flag', x, y, z }
 *
 * Args:
 *     seed (number): PRNG seed for deterministic generation
 *
 * Returns:
 *     Array<object>: tower elements
 *
 * >>> generateTowerPath(42).length > 0
 * true
 */
function generateTowerPath(seed) {
    const rng = makeTowerRng(seed);
    const elements = [];

    let cx = TOWER_CENTER_X;
    let cy = 0;
    let cz = TOWER_CENTER_Z;

    const maxDrift = 8;  // max XZ distance from tower center

    // Starting platform (wide, on the ground)
    elements.push({
        type: 'platform', x: cx, y: 0.15, z: cz,
        hw: 3.0, hh: 0.15, hd: 3.0,
        color: 'stone',
    });

    while (cy < TOWER_TARGET_HEIGHT) {
        const sectionType = rng.next();

        if (sectionType < 0.4) {
            // --- Ramp section: continuous surface gaining 3-6 units of height ---
            const rampHeightGain = rng.range(3, 6);
            const rampWidth = rng.range(1.8, 2.5);
            const rampAngle = Math.atan2(rampHeightGain, 8);  // slope from ~8 unit run

            // Pick ramp direction (pull toward center if drifting too far)
            const driftX = cx - TOWER_CENTER_X;
            const driftZ = cz - TOWER_CENTER_Z;
            let rampDirX, rampDirZ;
            if (Math.abs(driftX) + Math.abs(driftZ) > maxDrift * 0.6) {
                const len = Math.hypot(driftX, driftZ) || 1;
                rampDirX = -driftX / len;
                rampDirZ = -driftZ / len;
            } else {
                const angle = rng.range(0, Math.PI * 2);
                rampDirX = Math.cos(angle);
                rampDirZ = Math.sin(angle);
            }

            const runLen = rampHeightGain / Math.tan(rampAngle);
            const endX = cx + rampDirX * runLen;
            const endZ = cz + rampDirZ * runLen;
            const endY = cy + rampHeightGain;
            const midX = (cx + endX) / 2;
            const midZ = (cz + endZ) / 2;
            const midY = (cy + endY) / 2;
            const rampHalfLen = Math.hypot(runLen, rampHeightGain) / 2;
            const rampYaw = Math.atan2(rampDirZ, rampDirX);

            elements.push({
                type: 'ramp',
                x: midX, y: midY, z: midZ,
                hw: rampWidth / 2, hh: 0.15, hd: rampHalfLen,
                angle: rampAngle,
                yaw: rampYaw,
                color: 'ramp',
            });

            // Landing platform at top of ramp
            cx = endX;
            cy = endY;
            cz = endZ;
            elements.push({
                type: 'platform', x: cx, y: cy, z: cz,
                hw: rng.range(1.5, 2.5), hh: 0.15, hd: rng.range(1.5, 2.5),
                color: 'sandstone',
            });
        } else {
            // --- Jumping section: 3-5 platforms within jump reach ---
            const numPlatforms = Math.floor(rng.range(3, 6));
            for (let i = 0; i < numPlatforms; i++) {
                const dy = rng.range(0.5, 2.5);
                const dAngle = rng.range(0, Math.PI * 2);
                const dHoriz = rng.range(1.0, 4.0);
                let nx = cx + Math.cos(dAngle) * dHoriz;
                let nz = cz + Math.sin(dAngle) * dHoriz;

                // Clamp drift from tower center
                const dx = nx - TOWER_CENTER_X;
                const dz = nz - TOWER_CENTER_Z;
                const dist = Math.hypot(dx, dz);
                if (dist > maxDrift) {
                    nx = TOWER_CENTER_X + (dx / dist) * maxDrift;
                    nz = TOWER_CENTER_Z + (dz / dist) * maxDrift;
                }

                cy += dy;
                cx = nx;
                cz = nz;

                const hw = rng.range(0.5, 1.5);
                const hd = rng.range(0.5, 1.5);
                const colorChoice = rng.next();
                const color = colorChoice < 0.3 ? 'teal' :
                              colorChoice < 0.6 ? 'stone' : 'sandstone';

                elements.push({
                    type: 'platform', x: cx, y: cy, z: cz,
                    hw, hh: 0.15, hd,
                    color,
                });
            }
        }
    }

    // Decoy branches off the main path (dead ends for interest)
    const mainPlatforms = elements.filter(e => e.type === 'platform');
    const numDecoys = Math.min(8, Math.floor(mainPlatforms.length / 4));
    for (let d = 0; d < numDecoys; d++) {
        const branchIdx = Math.floor(rng.range(1, mainPlatforms.length - 1));
        const base = mainPlatforms[branchIdx];
        let bx = base.x, by = base.y, bz = base.z;
        const branchLen = Math.floor(rng.range(2, 4));
        for (let b = 0; b < branchLen; b++) {
            const dAngle = rng.range(0, Math.PI * 2);
            const dHoriz = rng.range(2.0, 5.0);
            bx += Math.cos(dAngle) * dHoriz;
            bz += Math.sin(dAngle) * dHoriz;
            by += rng.range(-0.3, 0.5);  // mostly sideways, slight height gain

            elements.push({
                type: 'platform', x: bx, y: by, z: bz,
                hw: rng.range(0.5, 1.2), hh: 0.15, hd: rng.range(0.5, 1.2),
                color: 'stone',
            });
        }
    }

    // Flag at the final main-path position
    elements.push({ type: 'flag', x: cx, y: cy, z: cz });

    return elements;
}

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
        this.mazeWalls = [];     // { body, halfExtents: [hx, hy, hz] }
        this.mazeChest = null;   // { body }
        this.mazeIvy   = [];     // render-only: { pos, rot, half, color } — no physics body

        // Marble machine arrays
        this.mmStructure = [];  // fixed: { body, half: [hx,hy,hz], color: [r,g,b,a] }
        this.mmWheels = [];     // dynamic: { body, half: [hx,hy,hz], color: [r,g,b,a] }
        this.mmChains = [];     // dynamic: { body, half: [hx,hy,hz], color: [r,g,b,a] }

        // Platform tower arrays
        this.towerPlatforms = [];  // { body, half: [hx,hy,hz], color: [r,g,b,a] }
        this.towerRamps = [];      // { body, half: [hx,hy,hz], color: [r,g,b,a] }
        this.towerFlag = [];       // { body, half: [hx,hy,hz], color: [r,g,b,a] }

        // Forest / Wilds arrays
        this.terrainBlocks = [];   // fixed: { body, half: [hx,hy,hz], color: [r,g,b,a] }
        this.trees         = [];   // fixed: trunks + canopies
        this.shrubs        = [];   // fixed: { body, half, color }
        this.mushrooms     = [];   // fixed: stems + caps

        // Signposts for all area entrances
        this.signposts = [];  // fixed: { body, half: [hx,hy,hz], color: [r,g,b,a] }

        // Perimeter fence: render-only posts+rails + 4 invisible wall colliders
        this.fence = [];       // render-only: { pos, rot, half, color }
        this.fenceWalls = [];  // physics-only: fixed bodies
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
        this._createMaze();
        this._createMarbleMachine();
        this._createPlatformTower();
        this._createForest();
        this._createSignposts();
        this._createFence();
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

    /**
     * Create white picket fence around the floor perimeter.
     * 4 invisible wall colliders for physics + render-only posts and rails.
     * Not pure: creates physics bodies and populates this.fence/this.fenceWalls.
     */
    _createFence() {
        const HALF = FLOOR_HALF.x;  // 200 — floor is square
        const FENCE_H = 0.75;       // fence half-height (1.5 units tall)
        const POST_SPACING = 4;
        const POST_HALF = [0.08, FENCE_H, 0.08];
        const RAIL_THICK = 0.04;
        const RAIL_HALF_H = 0.04;
        const COLOR = [0.95, 0.93, 0.88, 1.0];
        const IDENT_ROT = { x: 0, y: 0, z: 0, w: 1 };

        // 4 invisible physics wall colliders (thin cuboids, 1.5m tall)
        const walls = [
            { pos: [0, FENCE_H, -HALF], half: [HALF, FENCE_H, 0.15] },  // north
            { pos: [0, FENCE_H,  HALF], half: [HALF, FENCE_H, 0.15] },  // south
            { pos: [ HALF, FENCE_H, 0], half: [0.15, FENCE_H, HALF] },  // east
            { pos: [-HALF, FENCE_H, 0], half: [0.15, FENCE_H, HALF] },  // west
        ];
        for (const w of walls) {
            const rb = this.world.createRigidBody(
                RAPIER.RigidBodyDesc.fixed().setTranslation(...w.pos));
            this.world.createCollider(
                RAPIER.ColliderDesc.cuboid(...w.half).setFriction(0.3), rb);
            this.fenceWalls.push(rb);
        }

        // Visual fence: posts every POST_SPACING along each edge + connecting rails
        const addPost = (x, z) => {
            this.fence.push({ pos: { x, y: FENCE_H, z }, rot: IDENT_ROT, half: POST_HALF, color: COLOR });
        };

        const addRail = (cx, cy, cz, hx, hy, hz) => {
            this.fence.push({ pos: { x: cx, y: cy, z: cz }, rot: IDENT_ROT, half: [hx, hy, hz], color: COLOR });
        };

        // Rail Y positions: bottom rail at y=0.35, top rail at y=1.15
        const railYBot = 0.35;
        const railYTop = 1.15;
        const railHalfSpan = POST_SPACING / 2;  // half the distance between posts

        // North edge (z = -HALF): posts along x, rails along x
        for (let x = -HALF; x <= HALF; x += POST_SPACING) {
            addPost(x, -HALF);
            if (x + POST_SPACING <= HALF) {
                const cx = x + railHalfSpan;
                addRail(cx, railYBot, -HALF, railHalfSpan, RAIL_HALF_H, RAIL_THICK);
                addRail(cx, railYTop, -HALF, railHalfSpan, RAIL_HALF_H, RAIL_THICK);
            }
        }
        // South edge (z = +HALF)
        for (let x = -HALF; x <= HALF; x += POST_SPACING) {
            addPost(x, HALF);
            if (x + POST_SPACING <= HALF) {
                const cx = x + railHalfSpan;
                addRail(cx, railYBot, HALF, railHalfSpan, RAIL_HALF_H, RAIL_THICK);
                addRail(cx, railYTop, HALF, railHalfSpan, RAIL_HALF_H, RAIL_THICK);
            }
        }
        // East edge (x = +HALF): posts along z, rails along z (skip corners — already placed)
        for (let z = -HALF + POST_SPACING; z < HALF; z += POST_SPACING) {
            addPost(HALF, z);
            if (z + POST_SPACING < HALF) {
                const cz = z + railHalfSpan;
                addRail(HALF, railYBot, cz, RAIL_THICK, RAIL_HALF_H, railHalfSpan);
                addRail(HALF, railYTop, cz, RAIL_THICK, RAIL_HALF_H, railHalfSpan);
            }
        }
        // West edge (x = -HALF)
        for (let z = -HALF + POST_SPACING; z < HALF; z += POST_SPACING) {
            addPost(-HALF, z);
            if (z + POST_SPACING < HALF) {
                const cz = z + railHalfSpan;
                addRail(-HALF, railYBot, cz, RAIL_THICK, RAIL_HALF_H, railHalfSpan);
                addRail(-HALF, railYTop, cz, RAIL_THICK, RAIL_HALF_H, railHalfSpan);
            }
        }

        // Add final edge rails to connect the last posts to the corners on east/west
        // (the loop skips the last section since z + POST_SPACING might overshoot)
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

    /**
     * Generate and place the stone garden maze at MAZE_CENTER_X, MAZE_CENTER_Z.
     * Uses recursive backtracker (DFS) for maze generation, then creates fixed
     * cuboid colliders for each wall segment. Entrance faces +X (toward main area).
     * Not pure: creates rigid bodies in the physics world.
     */
    _createMaze() {
        const grid = generateMaze(MAZE_ROWS, MAZE_COLS);

        // World-space origin of the maze grid (top-left corner of cell [0,0])
        const originX = MAZE_CENTER_X - (MAZE_COLS * MAZE_CELL_SIZE) / 2;
        const originZ = MAZE_CENTER_Z - (MAZE_ROWS * MAZE_CELL_SIZE) / 2;

        // Remove entrance wall: rightmost wall of the middle row (faces +X)
        const entranceRow = Math.floor(MAZE_ROWS / 2);
        grid.vertical[entranceRow][MAZE_COLS] = false;

        // Create wall colliders from the grid
        // Horizontal walls: grid.horizontal[r][c] means wall on top edge of cell (r,c)
        // Runs along X (width = MAZE_CELL_SIZE), at the boundary between rows r-1 and r.
        for (let r = 0; r <= MAZE_ROWS; r++) {
            for (let c = 0; c < MAZE_COLS; c++) {
                if (!grid.horizontal[r][c]) continue;
                const cx = originX + (c + 0.5) * MAZE_CELL_SIZE;
                const cz = originZ + r * MAZE_CELL_SIZE;
                const hx = MAZE_CELL_SIZE / 2;
                const hz = MAZE_WALL_HALF_THICK;
                this._createMazeWall(cx, MAZE_WALL_HALF_Y, cz, hx, MAZE_WALL_HALF_Y, hz);
            }
        }

        // Vertical walls: grid.vertical[r][c] means wall on left edge of cell (r,c)
        // Runs along Z (height = MAZE_CELL_SIZE), at the boundary between cols c-1 and c.
        for (let r = 0; r < MAZE_ROWS; r++) {
            for (let c = 0; c <= MAZE_COLS; c++) {
                if (!grid.vertical[r][c]) continue;
                const cx = originX + c * MAZE_CELL_SIZE;
                const cz = originZ + (r + 0.5) * MAZE_CELL_SIZE;
                const hx = MAZE_WALL_HALF_THICK;
                const hz = MAZE_CELL_SIZE / 2;
                this._createMazeWall(cx, MAZE_WALL_HALF_Y, cz, hx, MAZE_WALL_HALF_Y, hz);
            }
        }

        // Treasure chest at the center cell
        const centerRow = Math.floor(MAZE_ROWS / 2);
        const centerCol = Math.floor(MAZE_COLS / 2);
        const chestX = originX + (centerCol + 0.5) * MAZE_CELL_SIZE;
        const chestZ = originZ + (centerRow + 0.5) * MAZE_CELL_SIZE;
        const chestY = CHEST_HALF.y;  // sitting on the floor
        const rbDesc = RAPIER.RigidBodyDesc.fixed().setTranslation(chestX, chestY, chestZ);
        const rb = this.world.createRigidBody(rbDesc);
        const colDesc = RAPIER.ColliderDesc.cuboid(CHEST_HALF.x, CHEST_HALF.y, CHEST_HALF.z)
            .setFriction(0.5)
            .setRestitution(0.1);
        this.world.createCollider(colDesc, rb);
        this.mazeChest = { body: rb };

        // Ivy decoration: render-only leaf clusters and vine tendrils on wall tops.
        // Uses a seeded LCG for deterministic placement (same layout every run).
        this.mazeIvy = [];
        this._populateMazeIvy(grid, originX, originZ);
    }

    /**
     * Populate this.mazeIvy with render-only leaf and tendril boxes on maze wall tops.
     * Roughly 30% of wall top segments get ivy. Uses a seeded LCG for determinism.
     * Not pure: mutates this.mazeIvy.
     *
     * @param {object} grid - { horizontal, vertical } boolean arrays from generateMaze
     * @param {number} originX - World X of the maze grid top-left corner
     * @param {number} originZ - World Z of the maze grid top-left corner
     */
    _populateMazeIvy(grid, originX, originZ) {
        const identityRot = { x: 0, y: 0, z: 0, w: 1 };
        const wallTopY    = MAZE_WALL_HEIGHT;  // top surface Y of walls
        const LEAF_HALF   = [0.3, 0.075, 0.3];
        const TENDRIL_HALF = [0.05, 0.20, 0.05];
        const IVY_PROB    = 0.30;  // ~30% of wall segments get ivy

        // Simple seeded LCG so ivy layout is deterministic each run.
        // Returns next pseudo-random float in [0, 1).
        let seed = 0xdeadbeef;
        const rng = () => {
            seed = (seed * 1664525 + 1013904223) >>> 0;
            return (seed >>> 0) / 0xffffffff;
        };

        // Leaf cluster color: random green variation per instance.
        const leafColor = () => {
            const g = rng() * 0.20 + 0.40;  // 0.40 – 0.60
            const r = rng() * 0.10 + 0.10;  // 0.10 – 0.20
            const b = rng() * 0.08 + 0.06;  // 0.06 – 0.14
            return [r, g, b, 1.0];
        };

        // Vine stem color: dark green, fixed.
        const stemColor = () => [0.08, 0.22, 0.05, 1.0];

        // Horizontal walls: run along X at z = originZ + r * MAZE_CELL_SIZE
        for (let r = 0; r <= MAZE_ROWS; r++) {
            for (let c = 0; c < MAZE_COLS; c++) {
                if (!grid.horizontal[r][c]) continue;
                if (rng() > IVY_PROB) continue;

                const cx = originX + (c + 0.5) * MAZE_CELL_SIZE;
                const cz = originZ + r * MAZE_CELL_SIZE;

                // Leaf cluster on top of wall
                const leafOffsetX = (rng() - 0.5) * (MAZE_CELL_SIZE * 0.6);
                this.mazeIvy.push({
                    pos: { x: cx + leafOffsetX, y: wallTopY + LEAF_HALF[1], z: cz },
                    rot: identityRot,
                    half: LEAF_HALF,
                    color: leafColor(),
                });

                // 1 – 3 tendrils hanging down the +Z face
                const numTendrils = 1 + Math.floor(rng() * 3);
                for (let t = 0; t < numTendrils; t++) {
                    const tx = cx + (rng() - 0.5) * MAZE_CELL_SIZE * 0.7;
                    const ty = wallTopY - TENDRIL_HALF[1] - rng() * 0.4;
                    const tz = cz + MAZE_WALL_HALF_THICK + TENDRIL_HALF[2];
                    this.mazeIvy.push({
                        pos: { x: tx, y: ty, z: tz },
                        rot: identityRot,
                        half: TENDRIL_HALF,
                        color: stemColor(),
                    });
                }
            }
        }

        // Vertical walls: run along Z at x = originX + c * MAZE_CELL_SIZE
        for (let r = 0; r < MAZE_ROWS; r++) {
            for (let c = 0; c <= MAZE_COLS; c++) {
                if (!grid.vertical[r][c]) continue;
                if (rng() > IVY_PROB) continue;

                const cx = originX + c * MAZE_CELL_SIZE;
                const cz = originZ + (r + 0.5) * MAZE_CELL_SIZE;

                // Leaf cluster on top of wall
                const leafOffsetZ = (rng() - 0.5) * (MAZE_CELL_SIZE * 0.6);
                this.mazeIvy.push({
                    pos: { x: cx, y: wallTopY + LEAF_HALF[1], z: cz + leafOffsetZ },
                    rot: identityRot,
                    half: LEAF_HALF,
                    color: leafColor(),
                });

                // 1 – 3 tendrils hanging down the +X face
                const numTendrils = 1 + Math.floor(rng() * 3);
                for (let t = 0; t < numTendrils; t++) {
                    const tx = cx + MAZE_WALL_HALF_THICK + TENDRIL_HALF[0];
                    const ty = wallTopY - TENDRIL_HALF[1] - rng() * 0.4;
                    const tz = cz + (rng() - 0.5) * MAZE_CELL_SIZE * 0.7;
                    this.mazeIvy.push({
                        pos: { x: tx, y: ty, z: tz },
                        rot: identityRot,
                        half: TENDRIL_HALF,
                        color: stemColor(),
                    });
                }
            }
        }
    }

    /**
     * Create a single fixed maze wall collider at (cx, cy, cz) with half-extents.
     * Not pure: creates a rigid body in the physics world, pushes to this.mazeWalls.
     *
     * @param {number} cx - Center X position
     * @param {number} cy - Center Y position
     * @param {number} cz - Center Z position
     * @param {number} hx - Half-extent X
     * @param {number} hy - Half-extent Y
     * @param {number} hz - Half-extent Z
     */
    _createMazeWall(cx, cy, cz, hx, hy, hz) {
        const rbDesc = RAPIER.RigidBodyDesc.fixed().setTranslation(cx, cy, cz);
        const rb = this.world.createRigidBody(rbDesc);
        const colDesc = RAPIER.ColliderDesc.cuboid(hx, hy, hz)
            .setFriction(0.6)
            .setRestitution(0.0);
        this.world.createCollider(colDesc, rb);
        this.mazeWalls.push({ body: rb, halfExtents: [hx, hy, hz] });
    }

    /**
     * Create the marble machine ("The Contraption") at (MM_CX, MM_CZ).
     *
     * Flow (top-down, Z increases northward):
     *
     *   Ball exit  ←[Ramp3]←  [P1, Y=3, Z=0]  ←[Ramp2]←  [P2, Y=6, Z=25]  ←[Ramp1]←  [P3, Y=9, Z=50]
     *              [Stair1]↑                    [Stair2]↑                    [Stair3]↑
     *
     * Stairs climb northward (+Z) on the west side (X ≈ MM_CX - 7).
     * Ramps descend southward (-Z) on the east side (X ≈ MM_CX + 7).
     * Chains hang from platform east edges — anchored at platform top, hanging freely below.
     * Wheels sit at ramp midpoints, axis=X so rolling balls spin them.
     *
     * Not pure: creates rigid bodies and joints in the physics world.
     */
    _createMarbleMachine() {
        this.mmStructure = [];
        this.mmWheels    = [];
        this.mmChains    = [];

        const C  = [0.6,  0.4,  0.2,  1.0]; // warm wood (platforms, stairs)
        const M  = [0.7,  0.7,  0.75, 1.0]; // brushed metal (railings, gutters)
        const S  = [0.55, 0.52, 0.48, 1.0]; // stone (ramps)
        const I  = [0.25, 0.25, 0.30, 1.0]; // dark iron (chains)
        const WC = [                          // wheel accent colors
            [0.85, 0.15, 0.15, 1.0],  // red
            [0.15, 0.25, 0.85, 1.0],  // blue
            [0.15, 0.75, 0.25, 1.0],  // green
        ];

        // ------------------------------------------------------------------
        // Platforms: all at X=MM_CX, spaced 25 units apart in Z.
        // Each platform center Y is the surface height (top = Y + MM_PLATFORM_HY ≈ Y + 0.15).
        // ------------------------------------------------------------------
        const P1 = { x: MM_CX, y: 3, z: MM_CZ };       // lowest,  Z=0
        const P2 = { x: MM_CX, y: 6, z: MM_CZ + 25 };  // middle,  Z=25
        const P3 = { x: MM_CX, y: 9, z: MM_CZ + 50 };  // highest, Z=50

        for (const p of [P1, P2, P3]) {
            this._mmBox(p.x, p.y, p.z, MM_PLATFORM_HX, MM_PLATFORM_HY, MM_PLATFORM_HZ, C, 0.7);
        }

        // Vertical pillars beneath each platform corner for visual support.
        for (const p of [P1, P2, P3]) {
            const pHY = p.y / 2;
            const ex  = MM_PLATFORM_HX - 0.5;
            const ez  = MM_PLATFORM_HZ - 0.5;
            this._mmBox(p.x + ex, pHY, p.z - ez, 0.4, pHY, 0.4, S);
            this._mmBox(p.x - ex, pHY, p.z - ez, 0.4, pHY, 0.4, S);
            this._mmBox(p.x + ex, pHY, p.z + ez, 0.4, pHY, 0.4, S);
            this._mmBox(p.x - ex, pHY, p.z + ez, 0.4, pHY, 0.4, S);
        }

        // ------------------------------------------------------------------
        // Staircases on the west side (X = MM_CX - 7), climbing in +Z direction.
        //
        // Rise per step = MM_STEP_HH * 2 = 0.30.  Run per step = MM_STEP_HD * 2 = 1.50.
        // 10 steps cover 3 Y units and 9 * 1.5 = 13.5 Z units (from step-0 to step-9 center).
        //
        // Stair 1 (ground → P1):
        //   want last step to land at P1 south edge Z = P1.z - MM_PLATFORM_HZ = -8
        //   sz_start = -8 - 9 * 1.5 = -21.5
        // Stair 2 (P1 → P2):
        //   first step just north of P1 north edge: sz = P1.z + MM_PLATFORM_HZ + 0.5 = 8.5
        //   last step center at 8.5 + 13.5 = 22 — inside P2 (Z: 17..33) ✓
        // Stair 3 (P2 → P3):
        //   sz = P2.z + MM_PLATFORM_HZ + 0.5 = 33.5
        //   last step at 33.5 + 13.5 = 47 — inside P3 (Z: 42..58) ✓
        // ------------------------------------------------------------------
        const STAIR_X  = MM_CX - 7;
        const stairSz1 = (P1.z - MM_PLATFORM_HZ) - 9 * (MM_STEP_HD * 2);
        const stairSz2 = P1.z + MM_PLATFORM_HZ + 0.5;
        const stairSz3 = P2.z + MM_PLATFORM_HZ + 0.5;

        this._mmStairs(STAIR_X, stairSz1, 0,    P1.y, 0, 1, C, M);
        this._mmStairs(STAIR_X, stairSz2, P1.y, P2.y, 0, 1, C, M);
        this._mmStairs(STAIR_X, stairSz3, P2.y, P3.y, 0, 1, C, M);

        // ------------------------------------------------------------------
        // Ramps on the east side (X = MM_CX + 7), descending in -Z direction.
        //
        // Each ramp is 15 units long with guttered sides and low friction.
        // Ramp 1: P3 south edge → Y drops from 9 to 6.
        // Ramp 2: P2 south edge → Y drops from 6 to 3.
        // Ramp 3: P1 south edge → Y drops from 3 to 0.3 (ball exits south).
        // ------------------------------------------------------------------
        const RAMP_X   = MM_CX + 7;
        const RAMP_LEN = 15;

        this._mmRamp(RAMP_X, P3.z - MM_PLATFORM_HZ, P3.y, P2.y, RAMP_LEN, 0, -1, S, M);
        this._mmRamp(RAMP_X, P2.z - MM_PLATFORM_HZ, P2.y, P1.y, RAMP_LEN, 0, -1, S, M);
        this._mmRamp(RAMP_X, P1.z - MM_PLATFORM_HZ, P1.y, 0.3,  RAMP_LEN, 0, -1, S, M);

        // Catch wall at the foot of Ramp 3 so balls pile up instead of escaping.
        const catchZ = (P1.z - MM_PLATFORM_HZ) - RAMP_LEN - 0.5;
        this._mmBox(RAMP_X, 0.25, catchZ, MM_RAMP_HW + 0.1, 0.4, 0.15, M);

        // ------------------------------------------------------------------
        // Spinning wheels at ramp midpoints (one per ramp).
        // Wheel axis = X (ax=1, az=0): ball rolling in -Z direction spins it.
        // Positioned at ramp center X, mid-height of the ramp's Y span, ramp mid-Z.
        // ------------------------------------------------------------------
        const rampMidZ = (startZ) => startZ - RAMP_LEN / 2;  // dz=-1: midpoint is startZ - half
        const rampMidY = (yHigh, yLow) => (yHigh + yLow) / 2;

        this._mmWheel(RAMP_X, rampMidY(P3.y, P2.y), rampMidZ(P3.z - MM_PLATFORM_HZ), 1, 0, 0, WC[0]);
        this._mmWheel(RAMP_X, rampMidY(P2.y, P1.y), rampMidZ(P2.z - MM_PLATFORM_HZ), 1, 0, 0, WC[1]);
        this._mmWheel(RAMP_X, rampMidY(P1.y, 0.3),  rampMidZ(P1.z - MM_PLATFORM_HZ), 1, 0, 0, WC[2]);

        // ------------------------------------------------------------------
        // Chains hanging from platform east edges (X = MM_CX + MM_PLATFORM_HX).
        // Anchor body fixed at platform surface height; links hang freely below.
        //
        // Per link: segH = MM_CHAIN_SEG.y * 2 + 0.05 gap ≈ 0.55 units.
        // 6 links → drop ≈ 3.3 units.
        //   P3 chains (anchor Y ≈ 9.15): bottom ≈ 5.85  — above P2 surface (6.15) ✓
        //   P2 chains (anchor Y ≈ 6.15): bottom ≈ 2.85  — above P1 surface (3.15) ✓
        //   P1 chains (5 links, anchor Y ≈ 3.15): bottom ≈ 0.4 — clears ground ✓
        // ------------------------------------------------------------------
        const anchorY   = (p) => p.y + MM_PLATFORM_HY;
        const EAST_EDGE = MM_CX + MM_PLATFORM_HX;  // X=110

        // P3: three chains spaced along Z
        this._mmChain(EAST_EDGE, anchorY(P3), P3.z - 4, 6, I);
        this._mmChain(EAST_EDGE, anchorY(P3), P3.z,     6, I);
        this._mmChain(EAST_EDGE, anchorY(P3), P3.z + 4, 6, I);

        // P2: two chains
        this._mmChain(EAST_EDGE, anchorY(P2), P2.z - 3, 6, I);
        this._mmChain(EAST_EDGE, anchorY(P2), P2.z + 3, 6, I);

        // P1: two chains, 5 links so they do not graze the ground
        this._mmChain(EAST_EDGE, anchorY(P1), P1.z - 3, 5, I);
        this._mmChain(EAST_EDGE, anchorY(P1), P1.z + 3, 5, I);
    }

    /** Not pure: creates a fixed box body, pushes to this.mmStructure. */
    _mmBox(cx, cy, cz, hx, hy, hz, color, friction = 0.5) {
        const rb = this.world.createRigidBody(
            RAPIER.RigidBodyDesc.fixed().setTranslation(cx, cy, cz)
        );
        this.world.createCollider(
            RAPIER.ColliderDesc.cuboid(hx, hy, hz).setFriction(friction).setRestitution(0.05), rb
        );
        this.mmStructure.push({ body: rb, half: [hx, hy, hz], color });
    }

    /** Not pure: creates staircase steps + railings via _mmBox. */
    _mmStairs(sx, sz, yFrom, yTo, dx, dz, stepColor, railColor) {
        const rise = MM_STEP_HH * 2;
        const run = MM_STEP_HD * 2;
        const n = Math.ceil((yTo - yFrom) / rise);
        const px = -dz, pz = dx;
        for (let i = 0; i < n; i++) {
            const y = yFrom + (i + 0.5) * rise;
            const x = sx + dx * i * run;
            const z = sz + dz * i * run;
            this._mmBox(x, y, z, MM_STEP_HW, MM_STEP_HH, MM_STEP_HD, stepColor, 0.6);
            const ry = y + MM_STEP_HH + MM_RAILING_HH;
            const rHx = (Math.abs(px) > 0) ? MM_RAILING_THICK : MM_STEP_HD;
            const rHz = (Math.abs(pz) > 0) ? MM_RAILING_THICK : MM_STEP_HD;
            const off = MM_STEP_HW + MM_RAILING_THICK;
            this._mmBox(x + px * off, ry, z + pz * off, rHx, MM_RAILING_HH, rHz, railColor);
            this._mmBox(x - px * off, ry, z - pz * off, rHx, MM_RAILING_HH, rHz, railColor);
        }
    }

    /** Not pure: creates ramp segments + gutter walls via _mmBox. */
    _mmRamp(sx, sz, yFrom, yTo, len, dx, dz, rampColor, gutterColor) {
        const segs = 8;
        const segLen = len / segs;
        const yDrop = (yFrom - yTo) / segs;
        const px = -dz, pz = dx;
        for (let i = 0; i < segs; i++) {
            const x = sx + dx * (i + 0.5) * segLen;
            const z = sz + dz * (i + 0.5) * segLen;
            const y = yFrom - (i + 0.5) * yDrop;
            const sHx = (dx !== 0) ? segLen / 2 : MM_RAMP_HW;
            const sHz = (dz !== 0) ? segLen / 2 : MM_RAMP_HW;
            this._mmBox(x, y, z, sHx, MM_RAMP_THICK, sHz, rampColor, 0.3);
            const gy = y + MM_RAMP_THICK + MM_GUTTER_HH;
            const gOff = MM_RAMP_HW + MM_GUTTER_THICK;
            const gHx = (dx !== 0) ? segLen / 2 : MM_GUTTER_THICK;
            const gHz = (dz !== 0) ? segLen / 2 : MM_GUTTER_THICK;
            this._mmBox(x + px * gOff, gy, z + pz * gOff, gHx, MM_GUTTER_HH, gHz, gutterColor);
            this._mmBox(x - px * gOff, gy, z - pz * gOff, gHx, MM_GUTTER_HH, gHz, gutterColor);
        }
    }

    /** Not pure: creates a spinning wheel + revolute joint. */
    _mmWheel(x, y, z, ax, ay, az, color) {
        const anchor = this.world.createRigidBody(
            RAPIER.RigidBodyDesc.fixed().setTranslation(x, y, z)
        );
        const wheel = this.world.createRigidBody(
            RAPIER.RigidBodyDesc.dynamic().setTranslation(x, y, z).setAngularDamping(0.5)
        );
        const hx = (ax !== 0) ? MM_WHEEL_THICK : MM_WHEEL_R;
        const hy = MM_WHEEL_R;
        const hz = (az !== 0) ? MM_WHEEL_THICK : MM_WHEEL_R;
        this.world.createCollider(
            RAPIER.ColliderDesc.cuboid(hx, hy, hz)
                .setFriction(0.4).setRestitution(0.3).setDensity(2.0),
            wheel
        );
        this.world.createImpulseJoint(
            RAPIER.JointData.revolute(
                { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }, { x: ax, y: ay, z: az }
            ), anchor, wheel, true
        );
        this.mmWheels.push({ body: wheel, half: [hx, hy, hz], color });
    }

    /** Not pure: creates a dangling chain of spherical-jointed boxes. */
    _mmChain(x, y, z, n, color) {
        const segH = MM_CHAIN_SEG.y * 2;
        const gap = 0.05;
        const anchorBody = this.world.createRigidBody(
            RAPIER.RigidBodyDesc.fixed().setTranslation(x, y, z)
        );
        let prev = anchorBody;
        for (let i = 0; i < n; i++) {
            const sy = y - (i + 0.5) * (segH + gap);
            const seg = this.world.createRigidBody(
                RAPIER.RigidBodyDesc.dynamic()
                    .setTranslation(x, sy, z).setLinearDamping(0.3).setAngularDamping(0.5)
            );
            this.world.createCollider(
                RAPIER.ColliderDesc.cuboid(MM_CHAIN_SEG.x, MM_CHAIN_SEG.y, MM_CHAIN_SEG.z)
                    .setFriction(0.3).setRestitution(0.1).setDensity(5.0),
                seg
            );
            const a1y = (i === 0) ? 0 : -MM_CHAIN_SEG.y - gap / 2;
            const a2y = MM_CHAIN_SEG.y + gap / 2;
            this.world.createImpulseJoint(
                RAPIER.JointData.spherical(
                    { x: 0, y: a1y, z: 0 }, { x: 0, y: a2y, z: 0 }
                ), prev, seg, true
            );
            this.mmChains.push({
                body: seg,
                half: [MM_CHAIN_SEG.x, MM_CHAIN_SEG.y, MM_CHAIN_SEG.z],
                color,
            });
            prev = seg;
        }
    }

    /**
     * Build the platforming tower in the far south of the world.
     * Generates a provably solvable path using generateTowerPath(),
     * then creates fixed cuboid colliders for each platform, ramp, and flag piece.
     * Not pure: creates rigid bodies in the physics world.
     */
    _createPlatformTower() {
        const elements = generateTowerPath(7777);

        // Color palette
        const COLORS = {
            stone:     [0.55, 0.53, 0.52, 1.0],
            sandstone: [0.76, 0.65, 0.50, 1.0],
            teal:      [0.30, 0.60, 0.65, 1.0],
            ramp:      [0.40, 0.40, 0.38, 1.0],
            flag_red:  [1.00, 0.20, 0.10, 1.0],
            flag_pole: [0.90, 0.90, 0.90, 1.0],
        };

        for (const el of elements) {
            if (el.type === 'platform') {
                const rbDesc = RAPIER.RigidBodyDesc.fixed()
                    .setTranslation(el.x, el.y, el.z);
                const rb = this.world.createRigidBody(rbDesc);
                const colDesc = RAPIER.ColliderDesc.cuboid(el.hw, el.hh, el.hd)
                    .setFriction(TOWER_PLATFORM_FRICTION)
                    .setRestitution(0.0);
                this.world.createCollider(colDesc, rb);
                this.towerPlatforms.push({
                    body: rb,
                    half: [el.hw, el.hh, el.hd],
                    color: COLORS[el.color] || COLORS.stone,
                });

            } else if (el.type === 'ramp') {
                // Ramp: rotated cuboid. Rotation = qYaw * qPitch
                // qYaw rotates around Y by el.yaw, qPitch tilts around local X by el.angle
                const cy = Math.cos(el.yaw / 2), sy = Math.sin(el.yaw / 2);
                const cp = Math.cos(el.angle / 2), sp = Math.sin(el.angle / 2);
                // Quaternion multiplication: qYaw(w=cy,x=0,y=sy,z=0) * qPitch(w=cp,x=sp,y=0,z=0)
                const qw = cy * cp;
                const qx = cy * sp;
                const qy = sy * cp;
                const qz = -sy * sp;

                const rbDesc = RAPIER.RigidBodyDesc.fixed()
                    .setTranslation(el.x, el.y, el.z)
                    .setRotation({ x: qx, y: qy, z: qz, w: qw });
                const rb = this.world.createRigidBody(rbDesc);
                const colDesc = RAPIER.ColliderDesc.cuboid(el.hw, el.hh, el.hd)
                    .setFriction(TOWER_RAMP_FRICTION)
                    .setRestitution(0.0);
                this.world.createCollider(colDesc, rb);
                this.towerRamps.push({
                    body: rb,
                    half: [el.hw, el.hh, el.hd],
                    color: COLORS[el.color] || COLORS.ramp,
                });

            } else if (el.type === 'flag') {
                // Flagpole: tall thin box
                const poleHalf = [0.05, 3.0, 0.05];
                const poleY = el.y + poleHalf[1];
                const pRb = RAPIER.RigidBodyDesc.fixed()
                    .setTranslation(el.x, poleY, el.z);
                const pBody = this.world.createRigidBody(pRb);
                const pCol = RAPIER.ColliderDesc.cuboid(...poleHalf)
                    .setFriction(0.3).setRestitution(0.0);
                this.world.createCollider(pCol, pBody);
                this.towerFlag.push({
                    body: pBody, half: poleHalf, color: COLORS.flag_pole,
                });

                // Flag: small box at top of pole, offset to the side
                const flagHalf = [0.6, 0.4, 0.05];
                const flagY = el.y + poleHalf[1] * 2 - flagHalf[1];
                const flagX = el.x + flagHalf[0];
                const fRb = RAPIER.RigidBodyDesc.fixed()
                    .setTranslation(flagX, flagY, el.z);
                const fBody = this.world.createRigidBody(fRb);
                const fCol = RAPIER.ColliderDesc.cuboid(...flagHalf)
                    .setFriction(0.3).setRestitution(0.0);
                this.world.createCollider(fCol, fBody);
                this.towerFlag.push({
                    body: fBody, half: flagHalf, color: COLORS.flag_red,
                });
            }
        }
    }

    /**
     * Create the forest and mountain biome in the far north (The Wilds).
     * Places terrain columns on a grid with height from FBM noise, trees on flat ground,
     * shrubs on low grass, and mushrooms sparsely in clearings.
     *
     * Not pure: creates many fixed rigid bodies in the physics world, populates
     * this.terrainBlocks, this.trees, this.shrubs, this.mushrooms.
     */
    _createForest() {
        this.terrainBlocks = [];  // { body, half: [hx,hy,hz], color: [r,g,b,a] }
        this.trees         = [];  // { body, half: [hx,hy,hz], color: [r,g,b,a] }
        this.shrubs        = [];  // { body, half: [hx,hy,hz], color: [r,g,b,a] }
        this.mushrooms     = [];  // { body, half: [hx,hy,hz], color: [r,g,b,a] }

        const step = FOREST_GRID_SPACING;
        const half = FOREST_HALF_EXTENT;
        const hx = step / 2;
        const hz = step / 2;

        // --- Terrain columns with physics cuboids ---
        // At 1.0m spacing, height differences between adjacent columns are small enough
        // for the player to walk over without jumping.
        for (let ix = -half; ix <= half; ix += step) {
            for (let iz = -half; iz <= half; iz += step) {
                const wx = FOREST_CENTER_X + ix;
                const wz = FOREST_CENTER_Z + iz;
                const h = terrainHeight(wx, wz);
                if (h < 0.1) continue;

                const hy = h / 2;
                const cy = hy;
                const slope = terrainSlope(wx, wz);
                const col = biomeColor(h, slope);

                const rbDesc = RAPIER.RigidBodyDesc.fixed().setTranslation(wx, cy, wz);
                const rb = this.world.createRigidBody(rbDesc);
                const colDesc = RAPIER.ColliderDesc.cuboid(hx, hy, hz)
                    .setFriction(0.7)
                    .setRestitution(0.05);
                this.world.createCollider(colDesc, rb);
                this.terrainBlocks.push({ body: rb, half: [hx, hy, hz], color: col });
            }
        }

        // --- Trees ---
        const treeRng = makeTowerRng(FOREST_SEED + 1);
        const treeGridStep = 10;
        for (let ix = -half; ix < half; ix += treeGridStep) {
            for (let iz = -half; iz < half; iz += treeGridStep) {
                const wx = FOREST_CENTER_X + ix + treeRng.range(-treeGridStep * 0.4, treeGridStep * 0.4);
                const wz = FOREST_CENTER_Z + iz + treeRng.range(-treeGridStep * 0.4, treeGridStep * 0.4);
                const h = terrainHeight(wx, wz);
                const slope = terrainSlope(wx, wz);
                if (h < 20 && slope < 3.0) {
                    const scale = treeRng.range(0.7, 1.3);
                    this._createTree(wx, wz, h, scale);
                }
            }
        }

        // --- Shrubs (dense, low flat areas only) ---
        const shrubRng = makeTowerRng(FOREST_SEED + 2);
        const shrubGridStep = 5;
        for (let ix = -half; ix < half; ix += shrubGridStep) {
            for (let iz = -half; iz < half; iz += shrubGridStep) {
                const wx = FOREST_CENTER_X + ix + shrubRng.range(-shrubGridStep * 0.45, shrubGridStep * 0.45);
                const wz = FOREST_CENTER_Z + iz + shrubRng.range(-shrubGridStep * 0.45, shrubGridStep * 0.45);
                const h = terrainHeight(wx, wz);
                const slope = terrainSlope(wx, wz);
                if (h < 12 && slope < 2.0 && shrubRng.next() < 0.35) {
                    const shrubHalf = [0.4, 0.3, 0.4];
                    const cy = h + shrubHalf[1];
                    const rb = this.world.createRigidBody(
                        RAPIER.RigidBodyDesc.fixed().setTranslation(wx, cy, wz)
                    );
                    this.world.createCollider(
                        RAPIER.ColliderDesc.cuboid(...shrubHalf).setFriction(0.5)
                    , rb);
                    this.shrubs.push({ body: rb, half: shrubHalf, color: [0.2, 0.45, 0.15, 1.0] });
                }
            }
        }

        // --- Mushrooms (sparse) ---
        const mushRng = makeTowerRng(FOREST_SEED + 3);
        for (let ix = -half; ix < half; ix += 15) {
            for (let iz = -half; iz < half; iz += 15) {
                const wx = FOREST_CENTER_X + ix + mushRng.range(-6, 6);
                const wz = FOREST_CENTER_Z + iz + mushRng.range(-6, 6);
                const h = terrainHeight(wx, wz);
                const slope = terrainSlope(wx, wz);
                if (h < 10 && slope < 1.5 && mushRng.next() < 0.3) {
                    this._createMushroom(wx, wz, h);
                }
            }
        }

    }

    /**
     * Create a single tree (trunk + canopy) at world (wx, -, wz) on terrain height h.
     * Not pure: creates rigid bodies, pushes to this.trees.
     *
     * @param {number} wx - World X
     * @param {number} wz - World Z
     * @param {number} h  - Terrain height at this position
     * @param {number} scale - Scale factor (0.7–1.3)
     */
    _createTree(wx, wz, h, scale) {
        // Trunk
        const trunkHalf = [0.3 * scale, 2.0 * scale, 0.3 * scale];
        const trunkY = h + trunkHalf[1];
        const trunkRb = this.world.createRigidBody(
            RAPIER.RigidBodyDesc.fixed().setTranslation(wx, trunkY, wz)
        );
        this.world.createCollider(
            RAPIER.ColliderDesc.cuboid(...trunkHalf).setFriction(0.5)
        , trunkRb);
        this.trees.push({ body: trunkRb, half: trunkHalf, color: [0.45, 0.3, 0.15, 1.0] });

        // Canopy
        const canopyHalf = [1.5 * scale, 1.0 * scale, 1.5 * scale];
        const canopyY = trunkY + trunkHalf[1] + canopyHalf[1] - 0.3 * scale;  // slight overlap
        const canopyRb = this.world.createRigidBody(
            RAPIER.RigidBodyDesc.fixed().setTranslation(wx, canopyY, wz)
        );
        this.world.createCollider(
            RAPIER.ColliderDesc.cuboid(...canopyHalf).setFriction(0.4)
        , canopyRb);
        this.trees.push({ body: canopyRb, half: canopyHalf, color: [0.15, 0.4, 0.1, 1.0] });
    }

    /**
     * Create a mushroom (stem + cap) at world (wx, -, wz).
     * Not pure: creates rigid bodies, pushes to this.mushrooms.
     *
     * @param {number} wx - World X
     * @param {number} wz - World Z
     * @param {number} h  - Terrain height at this position
     */
    _createMushroom(wx, wz, h) {
        // Stem
        const stemHalf = [0.1, 0.25, 0.1];
        const stemY = h + stemHalf[1];
        const stemRb = this.world.createRigidBody(
            RAPIER.RigidBodyDesc.fixed().setTranslation(wx, stemY, wz)
        );
        this.world.createCollider(
            RAPIER.ColliderDesc.cuboid(...stemHalf).setFriction(0.3)
        , stemRb);
        this.mushrooms.push({ body: stemRb, half: stemHalf, color: [0.85, 0.82, 0.75, 1.0] });

        // Cap
        const capHalf = [0.35, 0.18, 0.35];
        const capY = stemY + stemHalf[1] + capHalf[1];
        const capRb = this.world.createRigidBody(
            RAPIER.RigidBodyDesc.fixed().setTranslation(wx, capY, wz)
        );
        this.world.createCollider(
            RAPIER.ColliderDesc.cuboid(...capHalf).setFriction(0.3)
        , capRb);
        this.mushrooms.push({ body: capRb, half: capHalf, color: [0.8, 0.15, 0.1, 1.0] });
    }

    /**
     * Create billboard signposts at every area entrance.
     * Each sign: a tall wood pole + a large colored board on top.
     * Signs are color-coded per area for identification (no text rendering).
     * Not pure: creates fixed rigid bodies, populates this.signposts.
     *
     * >>> // Populates this.signposts with pole+board pairs for all 5 areas
     */
    _createSignposts() {
        // [ wx, wz, colorKey ] — position each sign near its area entrance
        const areas = [
            [  0,   -5, 'domino'      ],  // Domino Alley — near spawn, dominoes run south
            [-40,    0, 'labyrinth'   ],  // The Labyrinth — west entrance
            [ 40,    0, 'contraption' ],  // The Contraption — east entrance
            [  0,  -80, 'skySteps'    ],  // Sky Steps — south entrance
            [  0,   50, 'wilds'       ],  // The Wilds — north entrance
        ];
        for (const [wx, wz, key] of areas) {
            this._placeSignpost(wx, wz, SIGN_COLORS[key]);
        }
    }

    /**
     * Place a single billboard signpost (pole + sign board) at world (wx, wz).
     * Not pure: creates fixed rigid bodies, pushes 2 entries to this.signposts.
     *
     * @param {number} wx - World X
     * @param {number} wz - World Z
     * @param {number[]} boardColor - RGBA color for the sign board
     *
     * >>> // Pushes { body, half, color } for pole then board into this.signposts
     */
    _placeSignpost(wx, wz, boardColor) {
        // Pole — center at half-height so the base sits on y=0
        const poleHalf = SIGN_POLE_HALF;
        const poleY = poleHalf[1];
        const poleRb = this.world.createRigidBody(
            RAPIER.RigidBodyDesc.fixed().setTranslation(wx, poleY, wz)
        );
        this.world.createCollider(
            RAPIER.ColliderDesc.cuboid(...poleHalf).setFriction(0.5), poleRb
        );
        this.signposts.push({ body: poleRb, half: poleHalf, color: SIGN_POLE_COLOR });

        // Board — sits just above the pole top
        const boardHalf = SIGN_BOARD_HALF;
        const boardY = poleY * 2 + boardHalf[1];  // = pole top + board half-height
        const boardRb = this.world.createRigidBody(
            RAPIER.RigidBodyDesc.fixed().setTranslation(wx, boardY, wz)
        );
        this.world.createCollider(
            RAPIER.ColliderDesc.cuboid(...boardHalf).setFriction(0.3), boardRb
        );
        this.signposts.push({ body: boardRb, half: boardHalf, color: boardColor });
    }

    _createPlayer() {
        const rbDesc = RAPIER.RigidBodyDesc.dynamic()
            .setTranslation(0, PLAYER_HALF_HEIGHT + PLAYER_RADIUS + 0.1, 6)
            .lockRotations()
            .setLinearDamping(0.05);
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
     * Reset scene: remove all dynamic bodies and recreate them.
     * Not pure: destroys and recreates rigid bodies and joints.
     */
    reset() {
        // Remove spheres
        for (const s of this.spheres) this.world.removeRigidBody(s.body);
        this.spheres = [];

        // Remove dominoes
        for (const d of this.dominoes) this.world.removeRigidBody(d);
        this.dominoes = [];

        // Remove marble machine dynamic bodies (wheels, chains)
        for (const w of this.mmWheels) this.world.removeRigidBody(w.body);
        this.mmWheels = [];
        for (const c of this.mmChains) this.world.removeRigidBody(c.body);
        this.mmChains = [];
        // Structure is fixed — remove and recreate along with dynamic parts
        for (const s of this.mmStructure) this.world.removeRigidBody(s.body);
        this.mmStructure = [];

        // Recreate
        this._createDominoes(60);
        this._createMarbleMachine();

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

        const mazeWallData = this.mazeWalls.map(w => ({
            pos: w.body.translation(),
            rot: w.body.rotation(),
            half: w.halfExtents,
        }));

        const chestData = this.mazeChest ? {
            pos: this.mazeChest.body.translation(),
            rot: this.mazeChest.body.rotation(),
            half: CHEST_HALF,
        } : null;

        const extractBody = (item) => ({
            pos: item.body.translation(),
            rot: item.body.rotation(),
            half: item.half,
            color: item.color,
        });

        return {
            floor: { pos: floorPos, rot: floorRot, half: FLOOR_HALF },
            dominoes: dominoData,
            dominoHalf: DOMINO_HALF,
            spheres: sphereData,
            sphereRadius: SPHERE_RADIUS,
            mazeWalls: mazeWallData,
            mazeChest: chestData,
            // mazeIvy is render-only (no body) — pos/rot/half/color stored directly
            mazeIvy: this.mazeIvy,
            towerPlatforms: this.towerPlatforms.map(extractBody),
            towerRamps: this.towerRamps.map(extractBody),
            towerFlag: this.towerFlag.map(extractBody),
            mmStructure: this.mmStructure.map(extractBody),
            mmWheels: this.mmWheels.map(extractBody),
            mmChains: this.mmChains.map(extractBody),
            terrainBlocks: this.terrainBlocks.map(extractBody),
            trees: this.trees.map(extractBody),
            shrubs: this.shrubs.map(extractBody),
            mushrooms: this.mushrooms.map(extractBody),
            signposts: this.signposts.map(extractBody),
            fence: this.fence,  // render-only: { pos, rot, half, color }
        };
    }
}

export { PLAYER_EYE_OFFSET, CHEST_HALF };
