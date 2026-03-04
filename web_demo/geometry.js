/**
 * Vertex data for 3D primitives and fullscreen quads.
 * Direct port of game_demo/geometry.py.
 */

/**
 * Pure. Returns cube vertex data: position, normal, per-face color.
 * @returns {Float32Array} 36 vertices * 9 floats (px,py,pz, nx,ny,nz, r,g,b)
 */
export function cubeVertices() {
    // (normal, color, 4 corners) — 2 triangles via indices [0,1,2, 0,2,3]
    const faces = [
        [[0,0,1],  [1.0,0.3,0.3], [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]],       // Front  +Z
        [[0,0,-1], [0.3,1.0,0.3], [[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1]]],    // Back   -Z
        [[1,0,0],  [0.3,0.3,1.0], [[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]]],        // Right  +X
        [[-1,0,0], [1.0,1.0,0.3], [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]]],   // Left   -X
        [[0,1,0],  [1.0,0.3,1.0], [[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]]],       // Top    +Y
        [[0,-1,0], [0.3,1.0,1.0], [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]]],   // Bottom -Y
    ];
    const verts = [];
    const idx = [0, 1, 2, 0, 2, 3];
    for (const [normal, color, corners] of faces) {
        for (const i of idx) {
            const p = corners[i];
            verts.push(
                p[0]*0.5, p[1]*0.5, p[2]*0.5,
                normal[0], normal[1], normal[2],
                color[0], color[1], color[2],
            );
        }
    }
    return new Float32Array(verts);
}

/**
 * Pure. Returns floor quad vertex data at height y.
 * Same format as cube: (px,py,pz, nx,ny,nz, r,g,b).
 * @param {number} y - Floor height (default -1.5)
 * @param {number} size - Half-extent (default 5.0)
 * @returns {Float32Array} 6 vertices * 9 floats
 */
export function floorVertices(y = -1.5, size = 5.0) {
    const s = size;
    const n = [0, 1, 0];
    const c = [0.35, 0.35, 0.4];
    // CCW winding from above (+Y)
    const corners = [
        [-s, y, -s],
        [-s, y,  s],
        [ s, y,  s],
        [ s, y, -s],
    ];
    const verts = [];
    for (const i of [0, 1, 2, 0, 2, 3]) {
        const p = corners[i];
        verts.push(p[0], p[1], p[2], n[0], n[1], n[2], c[0], c[1], c[2]);
    }
    return new Float32Array(verts);
}

/**
 * Pure. Returns fullscreen quad: two triangles covering [-1,1] clip space.
 * @returns {Float32Array} 6 vertices * 4 floats (px, py, u, v)
 */
export function quadVertices() {
    return new Float32Array([
        -1, -1, 0, 0,
         1, -1, 1, 0,
         1,  1, 1, 1,
        -1, -1, 0, 0,
         1,  1, 1, 1,
        -1,  1, 0, 1,
    ]);
}
