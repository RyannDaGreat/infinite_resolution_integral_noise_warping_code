# Fluid Noise Warp — Web Demo

## Goal
Combine PavelDoGreat's WebGL Fluid Simulation (Navier-Stokes on GPU) with the
Infinite Resolution Integral Noise Warping algorithm (ICLR 2025). The fluid sim's
velocity field serves as optical flow to drive the particle warp kernel.

## Algorithm
1. **Fluid sim** (GPU): curl → vorticity → divergence → pressure → gradient subtract → advection
2. **Velocity readback**: blit velocity FBO to noise-resolution FBO (LINEAR upscale), then readPixels
3. **Particle warp** (CPU): 4-phase kernel (clear, backward map, Brownian bridge, normalize)
4. **Upload + display**: warped noise → texture → display shader with 6 modes

## Display Modes
- 0: **noise** — warped Gaussian noise (n/5 + 0.5, or CDF for uniform)
- 1: **scene** — raw fluid dye colors
- 2: **S+N** — scene + noise overlay (adjustable opacity)
- 3: **dither** — threshold scene luminance with noise CDF
- 4: **motion** — velocity field visualization
- 5: **raw** — raw noise values

## Controls
- Resolution (noise warp): [P] cycles 1024/512/256/128
- Greyscale: [G], Uniform: [U], Retina: [T], Interpolation: [I]
- Rounding: [O] cycles None/All/>1
- Blue noise: [B] toggle, [N] iteration count
- Threshold: [H] toggle + slider
- Lock noise: [L], Pause: [Space]
- Fluid params: sim resolution, dye resolution, curl, dissipation, pressure
- Splat: click/drag on canvas, or click Splat button

## File Structure
- `index.html` — canvas + control panel + mode bar
- `script.js` — self-contained: fluid sim + particle warp + display
- `claude_instructions.md` — this manifest

## Key Architecture Decisions
- Single-file JS (no modules) for simplicity and GitHub Pages deployment
- Fluid sim is a direct port of PavelDoGreat's WebGL Fluid Simulation (MIT license)
- Particle warp is a direct port of V1's `particle_warp.js`
- Display shader handles all 6 modes in a single fragment shader
- Velocity is upscaled to noise resolution via LINEAR-filtered blit before readPixels

## Lessons Learned
- (will be filled in as issues are discovered)
