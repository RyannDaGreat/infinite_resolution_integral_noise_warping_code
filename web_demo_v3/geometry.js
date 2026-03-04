/**
 * Vertex data for 3D primitives and fullscreen quads.
 * V3: position + normal only (6 floats per vertex). Color comes from instance data.
 */

/**
 * Pure. Unit box: half-extent 0.5 in each axis. 36 vertices x 6 floats.
 * Used for floor (no bevel needed at large scale).
 *
 * >>> boxVertices().length
 * 216
 */
export function boxVertices() {
    const faces = [
        [[0,0,1],  [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]],
        [[0,0,-1], [[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1]]],
        [[1,0,0],  [[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]]],
        [[-1,0,0], [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]]],
        [[0,1,0],  [[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]]],
        [[0,-1,0], [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]]],
    ];
    const verts = [];
    const idx = [0, 1, 2, 0, 2, 3];
    for (const [normal, corners] of faces) {
        for (const i of idx) {
            const p = corners[i];
            verts.push(p[0]*0.5, p[1]*0.5, p[2]*0.5, normal[0], normal[1], normal[2]);
        }
    }
    return new Float32Array(verts);
}

/**
 * Pure. Rounded-box vertex at a point on the surface of a box with spherical
 * bevel radius. Uses the wwwtyro rounded-box algorithm: clamp to inner box,
 * compute outward normal, offset by radius.
 *
 * Args:
 *     inner (number[3]): Inner box half-extents (halfExtents - radius)
 *     theta (number): Polar angle from the primary axis (0 = on face, pi/2 = neighbor)
 *     phi (number): Azimuthal angle from secondary axis toward tertiary axis
 *     signs (number[3]): Corner sign (+1 or -1) for each axis
 *     axes (number[3]): Axis permutation [primary, secondary, tertiary]
 *     radius (number): Bevel radius
 *
 * Returns:
 *     object: { p: [x, y, z], n: [nx, ny, nz] }
 *
 * >>> // roundedBoxVertex([0.71, 1.46, 0.14], 0, 0, [1,1,1], [0,1,2], 0.04)
 */
function roundedBoxVertex(inner, theta, phi, signs, axes, radius) {
    const ct = Math.cos(theta), st = Math.sin(theta);
    const cp = Math.cos(phi),   sp = Math.sin(phi);

    // Normal in the local (primary, secondary, tertiary) frame
    const nLocal = [ct, st * cp, st * sp];

    // Map to world axes using the permutation
    const n = [0, 0, 0];
    for (let i = 0; i < 3; i++) n[axes[i]] = signs[axes[i]] * nLocal[i];

    // Position = inner box corner + radius * normal
    const p = [0, 0, 0];
    for (let i = 0; i < 3; i++) p[i] = signs[i] * inner[i] + n[i] * radius;

    return { p, n };
}

/**
 * Pure. Emits two CCW triangles for a quad defined by four {p, n} vertices.
 * Winding is automatically corrected so the face normal points outward
 * (toward the average of the four vertex normals).
 *
 * Args:
 *     v00, v10, v11, v01: {p: number[3], n: number[3]} quad corners
 *     push: (p, n) => void — callback to append vertex data
 *
 * >>> // emitQuad({p:[0,0,0],n:[0,0,1]}, ..., push)
 */
function emitQuad(v00, v10, v11, v01, push) {
    // Geometric face normal via cross product
    const e1 = v10.p.map((x, i) => x - v00.p[i]);
    const e2 = v01.p.map((x, i) => x - v00.p[i]);
    const cross = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];

    // Outward direction = average of vertex normals
    const avgN = [0, 0, 0];
    for (const v of [v00, v10, v11, v01]) {
        avgN[0] += v.n[0]; avgN[1] += v.n[1]; avgN[2] += v.n[2];
    }
    const dot = cross[0] * avgN[0] + cross[1] * avgN[1] + cross[2] * avgN[2];

    if (dot >= 0) {
        push(v00.p, v00.n); push(v10.p, v10.n); push(v11.p, v11.n);
        push(v00.p, v00.n); push(v11.p, v11.n); push(v01.p, v01.n);
    } else {
        push(v00.p, v00.n); push(v01.p, v01.n); push(v11.p, v11.n);
        push(v00.p, v00.n); push(v11.p, v11.n); push(v10.p, v10.n);
    }
}

/**
 * Pure. Rounded box with configurable bevel segments. Uses the wwwtyro algorithm:
 * inner box = halfExtents - radius; for each surface point, clamp to inner box,
 * normal = normalize(point - clamped), position = clamped + normal * radius.
 *
 * Generates geometry at world scale, then normalizes vertex positions to unit-box
 * coordinates ([-0.5, 0.5] per axis) so the existing instance transform (which
 * scales by 2 * halfExtent) recovers the correct world-space positions. Normals
 * are left in world space. This preserves spherical corners despite non-uniform
 * instance scaling because the round-trip (divide then multiply) is exact.
 *
 * Geometry breakdown:
 *     - 6 flat face quads (inset by radius): 12 triangles
 *     - 12 edge quarter-cylinder strips (segments quads each): 24*segments tris
 *     - 8 corner eighth-sphere patches (pole fan + quad rows): 8*(2*segments^2 - segments) tris
 *     Total triangles = 12 + 24*segments + 8*(2*segments^2 - segments)
 *
 * Args:
 *     hx (number): Half-extent along X in world units
 *     hy (number): Half-extent along Y in world units
 *     hz (number): Half-extent along Z in world units
 *     radius (number): Bevel radius in world units (clamped to smallest half-extent)
 *     segments (number): Bevel subdivisions (1 = chamfer, 2+ = smooth curve)
 *     grooveAxis (string|null): Optional V-groove axis ('x', 'y', or 'z')
 *     grooveWidth (number): V-groove width in world units
 *     grooveDepth (number): V-groove depth in world units
 *
 * Returns:
 *     Float32Array: interleaved [px, py, pz, nx, ny, nz] per vertex
 *
 * >>> beveledBoxVertices(0.75, 1.5, 0.18, 0.04, 2).length % 6
 * 0
 * >>> beveledBoxVertices(1, 1, 1, 0.1, 1).length > 0
 * true
 */
export function beveledBoxVertices(hx = 1, hy = 1, hz = 1, radius = 0.04, segments = 2,
                                   grooveAxis = null, grooveWidth = 0, grooveDepth = 0) {
    const h = [hx, hy, hz];
    const r = Math.min(radius, ...h);  // clamp to smallest half-extent
    const inner = h.map(hi => hi - r); // inner box half-extents

    const verts = [];
    const push = (p, n) => {
        // Normalize position to unit-box coordinates: [-0.5, 0.5] per axis
        verts.push(p[0] / (2 * hx), p[1] / (2 * hy), p[2] / (2 * hz),
                   n[0], n[1], n[2]);
    };

    // --- 6 Flat face quads (inset by radius) ---
    // Each face perpendicular to axis `a` spans the inner extents of the other two axes.
    // Winding: cross(a1, a2) = +a for cyclic (0,1,2), so sign=+1 is CCW from outside.
    // For sign=-1, reverse winding by swapping index order.
    const faceQuadCCW = [0, 1, 2, 0, 2, 3];  // CCW for sign=+1
    const faceQuadCW  = [0, 2, 1, 0, 3, 2];  // reversed for sign=-1
    for (let a = 0; a < 3; a++) {
        const a1 = (a + 1) % 3;  // first tangent axis
        const a2 = (a + 2) % 3;  // second tangent axis
        for (const sign of [-1, 1]) {
            const n = [0, 0, 0];
            n[a] = sign;
            const faceQuadIdx = sign > 0 ? faceQuadCCW : faceQuadCW;

            // Check if this face pair needs a V-groove
            const axisName = 'xyz'[a];
            const needsGroove = grooveAxis === axisName && grooveWidth > 0 && grooveDepth > 0;

            if (needsGroove) {
                _emitGroovedFace(a, a1, a2, sign, h, inner, r,
                                 grooveWidth, grooveDepth, push);
            } else {
                // Simple inset quad: four corners at ±inner[a1], ±inner[a2]
                const corners = [
                    [-inner[a1], -inner[a2]],
                    [ inner[a1], -inner[a2]],
                    [ inner[a1],  inner[a2]],
                    [-inner[a1],  inner[a2]],
                ];
                for (const i of faceQuadIdx) {
                    const p = [0, 0, 0];
                    p[a] = sign * h[a];
                    p[a1] = corners[i][0];
                    p[a2] = corners[i][1];
                    push(p, n);
                }
            }
        }
    }

    // --- 12 Edge quarter-cylinder strips ---
    // Each edge is shared by two faces (axes a1 and a2), running along axis a3.
    // The bevel sweeps theta from 0 (face a1 normal) to pi/2 (face a2 normal).
    const halfPi = Math.PI / 2;
    for (let a1 = 0; a1 < 3; a1++) {
        for (let a2 = a1 + 1; a2 < 3; a2++) {
            const a3 = 3 - a1 - a2;  // remaining axis (the edge runs along this)
            for (const s1 of [-1, 1]) {
                for (const s2 of [-1, 1]) {
                    for (let seg = 0; seg < segments; seg++) {
                        const t0 = halfPi * seg / segments;
                        const t1 = halfPi * (seg + 1) / segments;

                        const makeVert = (theta, s3) => {
                            const ct = Math.cos(theta), st = Math.sin(theta);
                            const n = [0, 0, 0];
                            n[a1] = s1 * ct;
                            n[a2] = s2 * st;
                            const p = [0, 0, 0];
                            p[a1] = s1 * inner[a1] + n[a1] * r;
                            p[a2] = s2 * inner[a2] + n[a2] * r;
                            p[a3] = s3 * inner[a3];
                            return { p, n };
                        };

                        const v00 = makeVert(t0, -1);
                        const v10 = makeVert(t1, -1);
                        const v11 = makeVert(t1,  1);
                        const v01 = makeVert(t0,  1);
                        emitQuad(v00, v10, v11, v01, push);
                    }
                }
            }
        }
    }

    // --- 8 Corner eighth-sphere patches ---
    // Each corner is at (±inner[0], ±inner[1], ±inner[2]).
    // The eighth-sphere sweeps theta from 0 (primary axis face normal) to pi/2.
    // At theta=0, all phi values collapse to the pole (same point), so the first
    // row (i=0) emits a triangle fan instead of degenerate quads.
    for (const sx of [-1, 1]) {
        for (const sy of [-1, 1]) {
            for (const sz of [-1, 1]) {
                const signs = [sx, sy, sz];
                const axes = [0, 1, 2];
                for (let i = 0; i < segments; i++) {
                    const t0 = halfPi * i / segments;
                    const t1 = halfPi * (i + 1) / segments;

                    if (i === 0) {
                        // Pole row: triangle fan from the pole to the first ring
                        const pole = roundedBoxVertex(inner, 0, 0, signs, axes, r);
                        for (let j = 0; j < segments; j++) {
                            const p0 = halfPi * j / segments;
                            const p1 = halfPi * (j + 1) / segments;
                            const v10 = roundedBoxVertex(inner, t1, p0, signs, axes, r);
                            const v11 = roundedBoxVertex(inner, t1, p1, signs, axes, r);

                            // Determine winding from cross product
                            const e1 = v10.p.map((x, k) => x - pole.p[k]);
                            const e2 = v11.p.map((x, k) => x - pole.p[k]);
                            const cross = [
                                e1[1]*e2[2] - e1[2]*e2[1],
                                e1[2]*e2[0] - e1[0]*e2[2],
                                e1[0]*e2[1] - e1[1]*e2[0],
                            ];
                            const dot = cross[0]*pole.n[0] + cross[1]*pole.n[1] + cross[2]*pole.n[2];
                            if (dot >= 0) {
                                push(pole.p, pole.n); push(v10.p, v10.n); push(v11.p, v11.n);
                            } else {
                                push(pole.p, pole.n); push(v11.p, v11.n); push(v10.p, v10.n);
                            }
                        }
                    } else {
                        // Standard quad rows
                        for (let j = 0; j < segments; j++) {
                            const p0 = halfPi * j / segments;
                            const p1 = halfPi * (j + 1) / segments;
                            const v00 = roundedBoxVertex(inner, t0, p0, signs, axes, r);
                            const v10 = roundedBoxVertex(inner, t1, p0, signs, axes, r);
                            const v11 = roundedBoxVertex(inner, t1, p1, signs, axes, r);
                            const v01 = roundedBoxVertex(inner, t0, p1, signs, axes, r);
                            emitQuad(v00, v10, v11, v01, push);
                        }
                    }
                }
            }
        }
    }

    return new Float32Array(verts);
}

/**
 * Pure. Emit a face with a V-groove subdividing it along the tangent axis a1.
 * The groove is centered on the face and runs along axis a2.
 *
 * Cross-section (looking along a2):
 *     ___      ___
 *        \    /
 *         \  /
 *          \/
 *     <-w/2->
 *
 * Args:
 *     a (number): Face normal axis (0=x, 1=y, 2=z)
 *     a1 (number): Groove-perpendicular tangent axis (groove cuts across this)
 *     a2 (number): Groove-parallel tangent axis (groove runs along this)
 *     sign (number): +1 or -1 (which side of the box)
 *     h (number[3]): Full half-extents
 *     inner (number[3]): Inner half-extents (h - radius)
 *     r (number): Bevel radius
 *     grooveWidth (number): Width of V-groove in world units
 *     grooveDepth (number): Depth of V-groove in world units
 *     push (function): Vertex emit callback (p, n) => void
 *
 * >>> // _emitGroovedFace(2, 0, 1, 1, [0.75,1.5,0.18], [0.71,1.46,0.14], 0.04, 0.06, 0.02, push)
 */
function _emitGroovedFace(a, a1, a2, sign, h, inner, r, grooveWidth, grooveDepth, push) {
    const halfW = grooveWidth / 2;
    const facePos = sign * h[a];
    const faceN = [0, 0, 0];
    faceN[a] = sign;

    // Groove slope normal (tilted inward by the V angle)
    const slopeAngle = Math.atan2(grooveDepth, halfW);
    const cn = Math.cos(slopeAngle), sn = Math.sin(slopeAngle);

    // Same winding convention as flat faces: sign=+1 is CCW, sign=-1 reversed
    const quadIdx = sign > 0 ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2];
    const centerDepth = facePos - sign * grooveDepth;  // groove goes inward

    // Left slope normal: tilted toward -a1 and inward along a
    const leftN = [0, 0, 0];
    leftN[a] = sign * cn;
    leftN[a1] = -sn;

    // Right slope normal: tilted toward +a1 and inward along a
    const rightN = [0, 0, 0];
    rightN[a] = sign * cn;
    rightN[a1] = sn;

    // Helper: emit a quad strip with per-vertex depth along a1
    const emitStrip = (a1Lo, a1Hi, aLo, aHi, n) => {
        const corners = [
            { a1v: a1Lo, av: aLo, a2v: -inner[a2] },
            { a1v: a1Hi, av: aHi, a2v: -inner[a2] },
            { a1v: a1Hi, av: aHi, a2v:  inner[a2] },
            { a1v: a1Lo, av: aLo, a2v:  inner[a2] },
        ];
        for (const i of quadIdx) {
            const p = [0, 0, 0];
            p[a] = corners[i].av;
            p[a1] = corners[i].a1v;
            p[a2] = corners[i].a2v;
            push(p, n);
        }
    };

    // Clamp halfW to inner extent
    const clampedHalfW = Math.min(halfW, inner[a1]);

    // Left flat
    if (inner[a1] > clampedHalfW) {
        emitStrip(-inner[a1], -clampedHalfW, facePos, facePos, faceN);
    }
    // Left slope
    emitStrip(-clampedHalfW, 0, facePos, centerDepth, leftN);
    // Right slope
    emitStrip(0, clampedHalfW, centerDepth, facePos, rightN);
    // Right flat
    if (inner[a1] > clampedHalfW) {
        emitStrip(clampedHalfW, inner[a1], facePos, facePos, faceN);
    }
}

/**
 * Pure. UV sphere centered at origin with given radius.
 * Vertex format: position (vec3) + normal (vec3) = 6 floats per vertex.
 *
 * Args:
 *     radius (number): Sphere radius (default 0.5)
 *     lon (number): Longitude subdivisions (default 16)
 *     lat (number): Latitude subdivisions (default 12)
 *
 * >>> sphereVertices(0.5, 4, 3).length > 0
 * true
 */
export function sphereVertices(radius = 0.5, lon = 16, lat = 12) {
    const verts = [];
    for (let i = 0; i < lat; i++) {
        const th0 = Math.PI * i / lat;
        const th1 = Math.PI * (i + 1) / lat;
        for (let j = 0; j < lon; j++) {
            const ph0 = 2 * Math.PI * j / lon;
            const ph1 = 2 * Math.PI * (j + 1) / lon;

            const p = (th, ph) => [
                radius * Math.sin(th) * Math.cos(ph),
                radius * Math.cos(th),
                radius * Math.sin(th) * Math.sin(ph),
            ];
            const n = (th, ph) => {
                const x = Math.sin(th) * Math.cos(ph);
                const y = Math.cos(th);
                const z = Math.sin(th) * Math.sin(ph);
                return [x, y, z];
            };

            const p00 = p(th0, ph0), n00 = n(th0, ph0);
            const p10 = p(th1, ph0), n10 = n(th1, ph0);
            const p11 = p(th1, ph1), n11 = n(th1, ph1);
            const p01 = p(th0, ph1), n01 = n(th0, ph1);

            // Two triangles per quad (CCW winding from outside)
            verts.push(...p00, ...n00, ...p11, ...n11, ...p10, ...n10);
            verts.push(...p00, ...n00, ...p01, ...n01, ...p11, ...n11);
        }
    }
    return new Float32Array(verts);
}

/**
 * Pure. FBM (fractional Brownian motion) terrain height at world-space (x, z).
 * Uses 6 octaves, lacunarity 2.0, gain 0.5. Returns height in world units.
 *
 * Height is nearly zero near origin, growing with distance to produce massive
 * mountains at 2000+ units out. All heights are negative (terrain below y=0).
 *
 * Args:
 *     x (number): World X coordinate
 *     z (number): World Z coordinate
 *
 * Returns:
 *     number: World Y height (negative — terrain sits below y=0)
 *
 * >>> terrainHeight(0, 0)   // near origin: nearly flat
 * // returns a small negative number
 * >>> terrainHeight(3000, 0) // far out: large negative
 * // returns a large negative number (mountain)
 */
export function terrainHeight(x, z) {
    // FBM parameters
    const OCTAVES = 6;
    const LACUNARITY = 2.0;
    const GAIN = 0.5;
    // Base frequency: one noise "feature" every ~500 world units.
    // Phase offsets break symmetry around the origin so the terrain is not
    // degenerate along cardinal axes (gradient noise is 0 at integer lattice points).
    const BASE_FREQ = 1 / 500;
    const PHASE_X = 17.3;
    const PHASE_Z = 43.7;

    // Gradient noise hash → one of 8 diagonal-ish gradients
    const hash = (ix, iz) => {
        let h = (ix * 1619 + iz * 31337) | 0;
        h = ((h ^ (h >>> 17)) * 0x45d9f3b) | 0;
        h = ((h ^ (h >>> 13)) * 0x45d9f3b) | 0;
        return (h ^ (h >>> 16)) | 0;
    };

    // 8 gradient vectors (cardinal + diagonal) for richer variance
    const GRADS = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[-1,1],[1,-1],[-1,-1]];

    const gradDot = (ix, iz, fx, fz) => {
        const g = GRADS[((hash(ix, iz) >>> 0) % 8 + 8) % 8];
        const len = Math.SQRT2;  // normalize diagonals
        return (g[0] * fx + g[1] * fz) / len;
    };

    const smoothstep = (t) => t * t * (3 - 2 * t);

    const noise2 = (px, pz) => {
        const ix = Math.floor(px), iz = Math.floor(pz);
        const fx = px - ix, fz = pz - iz;
        const ux = smoothstep(fx), uz = smoothstep(fz);
        const v00 = gradDot(ix,   iz,   fx,     fz);
        const v10 = gradDot(ix+1, iz,   fx-1.0, fz);
        const v01 = gradDot(ix,   iz+1, fx,     fz-1.0);
        const v11 = gradDot(ix+1, iz+1, fx-1.0, fz-1.0);
        return (1 - ux) * ((1 - uz) * v00 + uz * v01) +
                    ux  * ((1 - uz) * v10 + uz * v11);
    };

    // FBM accumulation over the frequency-scaled, phase-offset input
    let value = 0;
    let amplitude = 1.0;
    let frequency = 1.0;
    for (let i = 0; i < OCTAVES; i++) {
        const px = (x + PHASE_X) * BASE_FREQ * frequency;
        const pz = (z + PHASE_Z) * BASE_FREQ * frequency;
        value += amplitude * noise2(px, pz);
        frequency *= LACUNARITY;
        amplitude *= GAIN;
    }

    // Distance from origin controls height amplitude:
    //   0-200 units: flat at -10 (smooth transition zone below floor)
    //   200-1000 units: gentle rolling hills, ramp up to ~100 units deep
    //   1000-4500 units: massive mountains, up to ~450 units deep
    const dist = Math.sqrt(x * x + z * z);
    const distScale = Math.max(0, dist - 200) / 800;
    const distAmp = Math.min(distScale * distScale, 1.0) * 450;

    // Terrain sits at or below y=-10 (well below the floor at y=0)
    return -10 + Math.abs(value) * distAmp;
}

/**
 * Pure. Terrain mesh: a flat grid of (gridSize x gridSize) quads with
 * heights displaced by terrainHeight(). Normals are computed analytically
 * from finite differences of neighboring heights.
 *
 * Vertex format: [px, py, pz, nx, ny, nz] — 6 floats per vertex.
 * Total vertices = gridSize^2 * 6 (two triangles per quad, 3 verts each).
 *
 * Args:
 *     gridSize (number): Number of cells per side (default 200 → 200×200 quads)
 *     extent (number): Half-extent of the grid in world units (default 4500)
 *
 * Returns:
 *     Float32Array: interleaved vertex data
 *
 * >>> terrainMeshVertices(2, 100).length % 6
 * 0
 * >>> terrainMeshVertices(2, 100).length
 * 144
 */
export function terrainMeshVertices(gridSize = 200, extent = 4500) {
    const cellSize = (2 * extent) / gridSize;
    const N = gridSize + 1;  // vertices per side

    // Pre-compute heights and normals for all grid vertices
    const heights = new Float32Array(N * N);
    for (let iz = 0; iz < N; iz++) {
        for (let ix = 0; ix < N; ix++) {
            const wx = -extent + ix * cellSize;
            const wz = -extent + iz * cellSize;
            heights[iz * N + ix] = terrainHeight(wx, wz);
        }
    }

    // Compute smooth normals via finite differences
    const normals = new Float32Array(N * N * 3);
    for (let iz = 0; iz < N; iz++) {
        for (let ix = 0; ix < N; ix++) {
            // Sample neighbors (clamped to grid)
            const ixL = Math.max(0, ix - 1), ixR = Math.min(N - 1, ix + 1);
            const izD = Math.max(0, iz - 1), izU = Math.min(N - 1, iz + 1);
            const dx = (heights[iz * N + ixR] - heights[iz * N + ixL]) /
                       ((ixR - ixL) * cellSize);
            const dz = (heights[izU * N + ix] - heights[izD * N + ix]) /
                       ((izU - izD) * cellSize);
            // Normal = normalize(-dx, 1, -dz)
            const len = Math.sqrt(dx * dx + 1 + dz * dz);
            const ni = iz * N + ix;
            normals[ni * 3 + 0] = -dx / len;
            normals[ni * 3 + 1] = 1 / len;
            normals[ni * 3 + 2] = -dz / len;
        }
    }

    // Emit triangles: two per quad cell
    const verts = new Float32Array(gridSize * gridSize * 6 * 6);
    let vi = 0;

    const pushVert = (ix, iz) => {
        const wx = -extent + ix * cellSize;
        const wz = -extent + iz * cellSize;
        const wy = heights[iz * N + ix];
        const ni = iz * N + ix;
        verts[vi++] = wx;
        verts[vi++] = wy;
        verts[vi++] = wz;
        verts[vi++] = normals[ni * 3 + 0];
        verts[vi++] = normals[ni * 3 + 1];
        verts[vi++] = normals[ni * 3 + 2];
    };

    for (let iz = 0; iz < gridSize; iz++) {
        for (let ix = 0; ix < gridSize; ix++) {
            // CCW winding when viewed from above (surface faces +Y)
            pushVert(ix,     iz);
            pushVert(ix+1, iz+1);
            pushVert(ix+1, iz);

            pushVert(ix,   iz);
            pushVert(ix,   iz+1);
            pushVert(ix+1, iz+1);
        }
    }

    return verts;
}

/**
 * Pure. Fullscreen quad for display pass.
 * WebGPU convention: v=0 at top, v=1 at bottom.
 *
 * >>> quadVertices().length
 * 24
 */
export function quadVertices() {
    return new Float32Array([
        -1, -1,  0, 1,
         1, -1,  1, 1,
         1,  1,  1, 0,
        -1, -1,  0, 1,
         1,  1,  1, 0,
        -1,  1,  0, 0,
    ]);
}
