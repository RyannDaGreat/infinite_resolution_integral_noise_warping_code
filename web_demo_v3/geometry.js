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
 * Pure. Beveled (chamfered) unit box with smooth normals for specular highlights.
 * Accepts world-space bevel width and target half-extents so the chamfer is
 * uniform in world space after the instance transform scales the unit box.
 *
 * Geometry: 6 inset faces + 12 edge bevel strips + 8 corner triangles = 44 triangles.
 * Normals: face vertices get face normal; GPU interpolation across bevel strips
 * creates a smooth rounded-edge appearance.
 *
 * Args:
 *     worldBevel (number): Desired uniform bevel width in world units
 *     halfExtents (number[3]): Target [hx, hy, hz] the box will be scaled to.
 *         Per-axis local bevel = worldBevel / (2 * halfExtent[axis]).
 *
 * Returns:
 *     Float32Array: 132 vertices x 6 floats = 792 floats
 *
 * >>> beveledBoxVertices(0.04, [1, 1, 1]).length
 * 792
 */
export function beveledBoxVertices(worldBevel = 0.04, halfExtents = [1, 1, 1]) {
    const h = 0.5;
    // Per-axis local bevel, clamped to prevent degeneracy
    const b = halfExtents.map(he => Math.min(worldBevel / (2 * he), 0.15));
    const hb = b.map(bi => h - bi);  // inset half-extent per axis

    const verts = [];
    const push = (p, n) => verts.push(p[0], p[1], p[2], n[0], n[1], n[2]);
    const tri = (p0, n0, p1, n1, p2, n2) => { push(p0, n0); push(p1, n1); push(p2, n2); };

    // --- 6 Faces (inset by per-axis bevel) ---
    // Each face at axis `a` uses hb of the OTHER two axes for its inset corners.
    const faces = [
        [[0,0,1],  [[-hb[0],-hb[1],h],[hb[0],-hb[1],h],[hb[0],hb[1],h],[-hb[0],hb[1],h]]],
        [[0,0,-1], [[hb[0],-hb[1],-h],[-hb[0],-hb[1],-h],[-hb[0],hb[1],-h],[hb[0],hb[1],-h]]],
        [[1,0,0],  [[h,-hb[1],hb[2]],[h,-hb[1],-hb[2]],[h,hb[1],-hb[2]],[h,hb[1],hb[2]]]],
        [[-1,0,0], [[-h,-hb[1],-hb[2]],[-h,-hb[1],hb[2]],[-h,hb[1],hb[2]],[-h,hb[1],-hb[2]]]],
        [[0,1,0],  [[-hb[0],h,hb[2]],[hb[0],h,hb[2]],[hb[0],h,-hb[2]],[-hb[0],h,-hb[2]]]],
        [[0,-1,0], [[-hb[0],-h,-hb[2]],[hb[0],-h,-hb[2]],[hb[0],-h,hb[2]],[-hb[0],-h,hb[2]]]],
    ];
    const idx = [0, 1, 2, 0, 2, 3];
    for (const [normal, corners] of faces) {
        for (const i of idx) {
            push(corners[i], normal);
        }
    }

    // --- 12 Edge bevel strips ---
    // Each edge shared by axis a1 and axis a2, running along axis a3.
    // Vertex normals = parent face normal → GPU interpolates across strip.
    for (let a1 = 0; a1 < 3; a1++) {
        for (let a2 = a1 + 1; a2 < 3; a2++) {
            const a3 = 3 - a1 - a2;
            for (const s1 of [-1, 1]) {
                for (const s2 of [-1, 1]) {
                    const n1 = [0, 0, 0]; n1[a1] = s1;
                    const n2 = [0, 0, 0]; n2[a2] = s2;

                    const v = [];
                    for (const s3 of [-1, 1]) {
                        // Vertex on face a1: at a1=±h, inset along a2 by hb[a2]
                        const p1 = [0, 0, 0];
                        p1[a1] = s1 * h; p1[a2] = s2 * hb[a2]; p1[a3] = s3 * hb[a3];
                        // Vertex on face a2: at a2=±h, inset along a1 by hb[a1]
                        const p2 = [0, 0, 0];
                        p2[a1] = s1 * hb[a1]; p2[a2] = s2 * h; p2[a3] = s3 * hb[a3];
                        v.push({ p: p1, n: [...n1] }, { p: p2, n: [...n2] });
                    }

                    // Determine winding: cross product should point outward
                    const e1 = v[2].p.map((x, i) => x - v[0].p[i]);
                    const e2 = v[1].p.map((x, i) => x - v[0].p[i]);
                    const cross = [
                        e1[1] * e2[2] - e1[2] * e2[1],
                        e1[2] * e2[0] - e1[0] * e2[2],
                        e1[0] * e2[1] - e1[1] * e2[0],
                    ];
                    const outward = [0, 0, 0]; outward[a1] = s1; outward[a2] = s2;
                    const dot = cross[0] * outward[0] + cross[1] * outward[1] + cross[2] * outward[2];

                    if (dot > 0) {
                        tri(v[0].p, v[0].n, v[2].p, v[2].n, v[3].p, v[3].n);
                        tri(v[0].p, v[0].n, v[3].p, v[3].n, v[1].p, v[1].n);
                    } else {
                        tri(v[0].p, v[0].n, v[1].p, v[1].n, v[3].p, v[3].n);
                        tri(v[0].p, v[0].n, v[3].p, v[3].n, v[2].p, v[2].n);
                    }
                }
            }
        }
    }

    // --- 8 Corner triangles ---
    for (const sx of [-1, 1]) {
        for (const sy of [-1, 1]) {
            for (const sz of [-1, 1]) {
                const px = [sx * h, sy * hb[1], sz * hb[2]]; const nx = [sx, 0, 0];
                const py = [sx * hb[0], sy * h, sz * hb[2]]; const ny = [0, sy, 0];
                const pz = [sx * hb[0], sy * hb[1], sz * h]; const nz = [0, 0, sz];

                const e1 = py.map((x, i) => x - px[i]);
                const e2 = pz.map((x, i) => x - px[i]);
                const cross = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                const dot = cross[0] * sx + cross[1] * sy + cross[2] * sz;

                if (dot > 0) {
                    tri(px, nx, py, ny, pz, nz);
                } else {
                    tri(px, nx, pz, nz, py, ny);
                }
            }
        }
    }

    return new Float32Array(verts);
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
