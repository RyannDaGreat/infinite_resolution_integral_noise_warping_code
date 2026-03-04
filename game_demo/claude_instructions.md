# Real-Time GLSL Noise Warping Game Demo

## Goal
Real-time interactive 3D demo rendering Gaussian noise warped by ground-truth motion vectors from 3D geometry. Two modes: GLSL (GPU warp + CPU regaussianize) and Taichi (full particle-based GWTF).

## Algorithm: NN Backward Warp + Regaussianize

### Step 1: Nearest-Neighbor Warp (GPU, fragment shader)
1. Each dest pixel reads its motion vector (screen-space velocity in UV)
2. Computes source UV = current UV - motion vector
3. Snaps to nearest source pixel (integer coords)
4. Reads previous noise at that pixel (exact copy)
5. OOB → fresh PCG Gaussian noise

### Step 2: Regaussianize (CPU, numpy)
After NN warp, expansion creates duplicates (many dests reading same source). Regaussianize restores Gaussianity:
1. Group pixels by identical channel-0 value (duplicates from same source have identical values)
2. For each group of count n:
   - Divide noise by sqrt(n) → corrects inflated variance
   - Generate foreign noise, compute per-group mean, subtract → zero-sum noise
   - Add zero-sum noise → breaks duplicates apart while preserving group sum
3. Result: independent N(0,1) at every pixel

**Proof of correctness:** If no duplicates, counts=1 everywhere, noise passes unchanged. If duplicates exist, dividing by sqrt(n) and adding zero-sum noise restores both variance and independence.

**Source:** `regaussianize()` in `rp/git/CommonSource/noise_warp.py` lines 298-336.

### Why NOT bilinear interpolation
- Bilinear = weighted sum of 4 Gaussians → shrinks variance, creates blur
- The reference Taichi impl uses bilinear for AREA DISTRIBUTION (scatter), not value interpolation
- NN preserves distribution exactly (each pixel = direct copy of one source)

### Why regaussianize CANNOT be in a fragment shader
- Requires `unique` (grouping by value) → global scatter operation
- Requires `index_add_` (summing within groups) → scatter-sum
- Fragment shaders can only read textures, not do scatter/gather across all pixels
- Would need compute shaders (OpenGL 4.3+, unavailable on macOS)

## Hardware & Constraints
- macOS, Apple M4 Pro, Metal 4
- OpenGL 4.1 only (GLSL 4.10) — no compute shaders
- Python 3.14.3

## Stack
- moderngl + pygame: Rendering + window/input
- PyGLM: Matrix math
- numpy: Noise init + regaussianize
- fire: CLI

## Render Pipeline

### GLSL mode (default): GPU warp + CPU regaussianize
```
1. Save previous MVP matrices
2. Update scene (rotate cube, camera)
3. SCENE PASS → MRT FBO:
   - location 0: RGBA8 color
   - location 1: RG32F motion vectors (UV-scale screen-space velocity)
4. NOISE WARP PASS → ping-pong FBO (GPU):
   - Nearest-neighbor backward warp only
   - OOB → fresh PCG Gaussian noise
5. CPU REGAUSSIANIZE:
   - Read warped noise back from GPU
   - Run numpy regaussianize()
   - Upload result back to GPU
6. DISPLAY PASS → screen
```

### Taichi mode: Full particle-based GWTF
```
1. Save previous MVP matrices
2. Update scene
3. SCENE PASS → MRT FBO (same as GLSL)
4. Read motion vectors from GPU → CPU
5. Taichi _warp_step (scatter + Brownian bridge + normalization)
6. Upload result to GPU
7. DISPLAY PASS → screen
```

## Motion Vector Computation
- Vertex shader: project each vertex with current MVP AND previous MVP
- Fragment shader: `motion = (curr_ndc - prev_ndc) * 0.5` (NDC to UV scale)
- Background pixels: cleared to (0, 0) = no motion

## File Structure
```
game_demo/
├── main.py              # Entry point, game loop, FPS camera, --mode flag
├── renderer.py          # ModernGL setup, FBOs, ping-pong, shader loading
├── geometry.py          # Cube + fullscreen quad vertex data
├── regaussianize.py     # Numpy regaussianize (port from noise_warp.py)
├── taichi_warp.py       # Taichi GWTF wrapper (uses rp.git.CommonSource)
├── shaders/
│   ├── scene.vert       # Dual MVP projection
│   ├── scene.frag       # MRT: color + motion vectors
│   ├── noise_warp.vert  # Fullscreen quad passthrough
│   ├── noise_warp.frag  # NN backward warp only (regaussianize is CPU-side)
│   ├── display.vert     # Fullscreen quad passthrough
│   └── display.frag     # Noise/color/motion visualization
├── test_gaussian.py     # Statistical validation
├── claude_instructions.md
└── concerns.md
```

## Gaussian PRNG in Shader
PCG hash + Box-Muller transform. Seed per pixel per frame: `pixel_x * 7919 + pixel_y * 6271 + frame * 104729`.

## Controls
- WASD / Space / Shift: Move camera
- Mouse (click to capture): Look around
- 1-4: Display modes (noise, color, motion, side-by-side)
- Q / ESC: Quit
- `--mode glsl` (default) or `--mode taichi`

## Key Decisions & Lessons Learned

### CRITICAL: Regaussianize is a specific algorithm
- It is NOT "replace duplicates with fresh noise"
- It is NOT "blend warped + fresh noise based on Jacobian"
- It IS: group by identical values → divide by sqrt(count) → add zero-sum foreign noise
- The function lives in `rp/git/CommonSource/noise_warp.py` at `regaussianize()`
- Helper functions: `unique_pixels()`, `sum_indexed_values()`, `indexed_to_image()`

### NN warp is discrete and quantized
- Source coordinates are integer pixel indices, not floating-point UVs
- No bilinear interpolation, no fractional weights, no variance correction from weights
- Weight is always 1 (exact copy) or 0 (OOB → fresh noise)

### Fragment shaders cannot do scatter/gather
- OpenGL 4.1 on macOS = no compute shaders
- Regaussianize requires global unique + scatter-sum → must be CPU-side
- This means a GPU→CPU→GPU roundtrip each frame (perf cost)

### Motion vectors verified correct
- Rotation axis visible as dark spot in magnitude visualization
- Magnitude increases outward from rotation axis
- ~17-22% of pixels have non-zero motion (matches cube screen coverage)

### Cube normals verified correct
- Cross-product of edges matches declared normals, all 6 faces OK

### Taichi on macOS
- Must use CPU backend (Metal has unreliable float atomics, no sparse SNodes)
- Imported from `rp.git.CommonSource.inf_int_noise_warp`

### Flow conventions
- GPU motion vectors: (mv_x, mv_y) in UV space, row 0 = bottom (OpenGL), Y-up
- Taichi expects: pixel space, row 0 = top, (dx, dy) where dy is positive-down
- Conversion: flipud, multiply by resolution, negate dy
