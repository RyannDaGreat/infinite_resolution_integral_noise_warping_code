/**
 * Vertex data for 3D primitives and fullscreen quads.
 * Adapted from V1 with WebGPU texture coordinate convention.
 */

/**
 * Pure. Returns cube vertex data: position, normal, per-face color.
 * @returns {Float32Array} 36 vertices * 9 floats (px,py,pz, nx,ny,nz, r,g,b)
 *
 * >>> cubeVertices().length
 * 324
 */
export function cubeVertices() {
    const faces = [
        [[0,0,1],  [1.0,0.3,0.3], [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]],
        [[0,0,-1], [0.3,1.0,0.3], [[1,-1,-1],[-1,-1,-1],[-1,1,-1],[1,1,-1]]],
        [[1,0,0],  [0.3,0.3,1.0], [[1,-1,1],[1,-1,-1],[1,1,-1],[1,1,1]]],
        [[-1,0,0], [1.0,1.0,0.3], [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1]]],
        [[0,1,0],  [1.0,0.3,1.0], [[-1,1,1],[1,1,1],[1,1,-1],[-1,1,-1]]],
        [[0,-1,0], [0.3,1.0,1.0], [[-1,-1,-1],[1,-1,-1],[1,-1,1],[-1,-1,1]]],
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
 * Pure. Returns floor quad at height y.
 * @param {number} y - Floor height (default -1.5)
 * @param {number} size - Half-extent (default 5.0)
 * @returns {Float32Array} 6 vertices * 9 floats
 *
 * >>> floorVertices().length
 * 54
 */
export function floorVertices(y = -1.5, size = 5.0) {
    const s = size;
    const n = [0, 1, 0];
    const c = [0.35, 0.35, 0.4];
    const corners = [[-s,y,-s], [-s,y,s], [s,y,s], [s,y,-s]];
    const verts = [];
    for (const i of [0, 1, 2, 0, 2, 3]) {
        const p = corners[i];
        verts.push(p[0], p[1], p[2], n[0], n[1], n[2], c[0], c[1], c[2]);
    }
    return new Float32Array(verts);
}

/**
 * Pure. Fullscreen quad for display pass.
 * WebGPU convention: v=0 at top, v=1 at bottom (Y-down for textures).
 * @returns {Float32Array} 6 vertices * 4 floats (px, py, u, v)
 *
 * >>> quadVertices().length
 * 24
 */
export function quadVertices() {
    return new Float32Array([
        // clip (x,y)   tex (u,v)
        -1, -1,  0, 1,  // bottom-left  → tex bottom
         1, -1,  1, 1,  // bottom-right → tex bottom
         1,  1,  1, 0,  // top-right    → tex top
        -1, -1,  0, 1,
         1,  1,  1, 0,
        -1,  1,  0, 0,  // top-left     → tex top
    ]);
}
