# Web Demo: Real-Time Noise Warping in the Browser

## Goal
Port the Python game demo (moderngl + pygame + taichi) to a self-contained WebGL 2 + JS web page at 512x512 for Chrome on Mac.

## Algorithm
The **particle-based GWTF** (Algorithm 3 from the ICLR 2025 paper) warps Gaussian white noise through motion vectors while preserving spatial uncorrelation via Brownian bridge stochastic sampling.

### 4-Phase Kernel (from `inf_int_noise_warp.py::_particle_warp_kernel`)
1. **Clear**: Zero out buffer_field, pixel_area_field, ticket_serial_field
2. **Backward map**: For each dest pixel, compute bilinear weights to 4 source pixels in the deformation map. Store "tickets" (raveled source index + weight) via atomic counters (MAX_TICKETS=24 per pixel)
3. **Brownian bridge**: For each source pixel, iterate its tickets, sample Brownian bridge B(t2)|B(t1)=q, scatter curr_value back to dest pixels, accumulate pixel_area
4. **Normalize**: For each dest pixel: if pixel_area > 0, noise = buffer/sqrt(pixel_area); else fresh randn

## Architecture
- **GPU (WebGL 2)**: Scene rendering (cube + floor) with MRT for color + motion vectors; display pass
- **CPU (JS)**: Particle warp kernel (ported from taichi); all noise state lives in JS arrays
- **No noise_warp shader** — warp is CPU-side
- **No regaussianize** — the taichi algo handles this intrinsically via Brownian bridge

## File Structure
```
web_demo/
  index.html           - Canvas, controls overlay, gl-matrix CDN
  main.js              - Entry point, game loop, FPSCamera, SpinningCube
  renderer.js          - WebGL 2 context, FBOs, programs, draw calls
  geometry.js          - Vertex data as Float32Arrays
  particle_warp.js     - JS port of taichi 4-phase kernel
  shaders.js           - 4 shader sources (scene.vert/frag, display.vert/frag)
  claude_instructions.md - This file
  concerns.md          - Progress log
```

## Key Differences from Python

| Aspect | Python (moderngl) | JS (WebGL 2) |
|---|---|---|
| Matrix upload | `bytes(glm.transpose(m))` | `uniformMatrix4fv(loc, false, m)` — NO transpose |
| MRT activation | Automatic from FBO config | Must call `gl.drawBuffers()` explicitly |
| Depth buffer | `ctx.depth_texture()` | `gl.createRenderbuffer` + DEPTH_COMPONENT24 |
| Float ext | Implicit | Must check `EXT_color_buffer_float` |
| Noise warp | Taichi kernel on CPU | JS port of same kernel on CPU |
| PRNG | `ti.randn()` | Box-Muller + Mulberry32 PRNG |

## Shaders Needed (4, not 6)
1. `scene.vert` — transforms vertices, computes curr/prev clip positions for motion vectors
2. `scene.frag` — MRT output: color (RGBA8) + motion vectors (RG32F)
3. `display.vert` — fullscreen quad passthrough
4. `display.frag` — visualization modes (noise, color, motion, side-by-side, raw)

## Motion Vector Pipeline
- GPU scene shader outputs motion vectors in UV displacement units: `(curr_ndc - prev_ndc) * 0.5`
- JS reads via `gl.readPixels` → Float32Array [H*W*2] in (mv_x, mv_y) format
- JS converts to deformation map: `identity_cc - flow_rc` where flow is in (row, col) pixel space
- Conversion: `flow_row = -mv_y * H`, `flow_col = mv_x * W` (flip Y, scale to pixels)
- Then: `deform_row = identity_row + mv_y * H`, `deform_col = identity_col - mv_x * W`

## Self-Validation / Testing
- Console-log mean/std of noise every 60 frames — should be mean~0, std~1
- On-screen stats overlay showing fps, mean, std, max group size
- Optional: statistical test on startup with zero flow (output should match input noise)

## Lessons Learned
(append-only — never remove)
