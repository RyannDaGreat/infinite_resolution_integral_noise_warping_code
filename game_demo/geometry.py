"""Vertex data for 3D primitives and fullscreen quads."""

import numpy as np


def cube_vertices():
    """
    Pure function. Returns cube vertex data: position, normal, per-face color.

    Returns:
        np.ndarray: Shape [36, 9] float32 — (px, py, pz, nx, ny, nz, r, g, b).

    Examples:
        >>> cube_vertices().shape
        (36, 9)
    """
    faces = [
        # (normal, color, 4 corners) — 2 triangles via indices [0,1,2, 0,2,3]
        ([0, 0, 1], [1.0, 0.3, 0.3], [(-1,-1, 1),( 1,-1, 1),( 1, 1, 1),(-1, 1, 1)]),  # Front  +Z
        ([0, 0,-1], [0.3, 1.0, 0.3], [( 1,-1,-1),(-1,-1,-1),(-1, 1,-1),( 1, 1,-1)]),  # Back   -Z
        ([1, 0, 0], [0.3, 0.3, 1.0], [( 1,-1, 1),( 1,-1,-1),( 1, 1,-1),( 1, 1, 1)]),  # Right  +X
        ([-1,0, 0], [1.0, 1.0, 0.3], [(-1,-1,-1),(-1,-1, 1),(-1, 1, 1),(-1, 1,-1)]),  # Left   -X
        ([0, 1, 0], [1.0, 0.3, 1.0], [(-1, 1, 1),( 1, 1, 1),( 1, 1,-1),(-1, 1,-1)]),  # Top    +Y
        ([0,-1, 0], [0.3, 1.0, 1.0], [(-1,-1,-1),( 1,-1,-1),( 1,-1, 1),(-1,-1, 1)]),  # Bottom -Y
    ]
    verts = []
    for normal, color, corners in faces:
        pts = [np.array(c, dtype=np.float32) * 0.5 for c in corners]
        n = np.array(normal, dtype=np.float32)
        c = np.array(color, dtype=np.float32)
        for i in [0, 1, 2, 0, 2, 3]:
            verts.append(np.concatenate([pts[i], n, c]))
    return np.array(verts, dtype=np.float32)


def floor_vertices(y=-1.5, size=5.0):
    """
    Pure function. Returns a floor quad at height y, centered at origin.

    Same vertex format as cube: (px, py, pz, nx, ny, nz, r, g, b).
    Normal points up (+Y). Gray color.

    Args:
        y (float): Floor height. Default -1.5 (2 cube-heights below center).
        size (float): Half-extent of floor quad.

    Returns:
        np.ndarray: Shape [6, 9] float32.

    Examples:
        >>> floor_vertices().shape
        (6, 9)
    """
    n = np.array([0, 1, 0], dtype=np.float32)
    c = np.array([0.35, 0.35, 0.4], dtype=np.float32)
    s = size
    # CCW winding when viewed from above (+Y) so front-face isn't culled
    corners = [
        np.array([-s, y, -s], dtype=np.float32),
        np.array([-s, y,  s], dtype=np.float32),
        np.array([ s, y,  s], dtype=np.float32),
        np.array([ s, y, -s], dtype=np.float32),
    ]
    verts = []
    for i in [0, 1, 2, 0, 2, 3]:
        verts.append(np.concatenate([corners[i], n, c]))
    return np.array(verts, dtype=np.float32)


def quad_vertices():
    """
    Pure function. Returns fullscreen quad: two triangles covering [-1,1] clip space.

    Returns:
        np.ndarray: Shape [6, 4] float32 — (px, py, u, v).

    Examples:
        >>> quad_vertices().shape
        (6, 4)
    """
    return np.array([
        [-1, -1, 0, 0],
        [ 1, -1, 1, 0],
        [ 1,  1, 1, 1],
        [-1, -1, 0, 0],
        [ 1,  1, 1, 1],
        [-1,  1, 0, 1],
    ], dtype=np.float32)
