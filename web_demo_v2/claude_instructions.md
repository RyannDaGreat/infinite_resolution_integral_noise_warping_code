# Web Demo V2 — WebGPU Compute Particle Warp

## Algorithm
Infinite-resolution integral noise warping implemented entirely on GPU via WebGPU compute shaders. Zero CPU-GPU copies per frame — the only CPU work is ~10 floats of matrix math + command buffer submission.

### Data Flow
```
GPU render (MRT: color+motion) → build deformation → clear → backward map → brownian bridge → normalize → [blue noise post-process] → display
```
All storage buffers stay GPU-resident. No readPixels, no texSubImage2D.

### Particle Warp Pipeline (per frame)
1. **Scene render** — MRT: rgba8 color + rgba32float motion vectors + depth
2. **Build deformation** — motion texture → deformation buffer (dx,dy per pixel)
3. **Clear** — zero intermediate buffers
4. **Backward map** — per-dest-pixel, bilinear weights → atomicAdd ticket slots on source pixels
5. **Brownian bridge** — per-source-pixel, sequential ticket processing with Box-Muller PRNG. Float atomics via CAS loop. 1-ticket fast path skips bridge entirely.
6. **Normalize** — per-dest-pixel: if pixelArea > 0, scale by 1/sqrt(area); else fresh Gaussian noise. pixelArea recomputed from deformation+totalRequest (no atomic needed).
7. **Blue noise (optional post-process)** — alternating projections: Gaussian high-pass + histogram matching. 10 iterations.
8. **Display** — reads noise from storage buffer, supports greyscale + Gaussian→uniform display flags.

### Blue Noise — CRITICAL CONSTRAINT
**Blue noise is POST-PROCESSING ONLY.** It must NEVER feed back into the warp. The warp must always receive clean Gaussian noise.

Implementation: backup noiseBuf → run blue noise (modifies noiseBuf for display) → display → restore noiseBuf from backup.

### Blue Noise Algorithm
Port of `rp.convert_to_blue_noise` — iterative high-pass with σ_hp rescaling:
1. **For each iteration (2, 5, or 10 — user configurable):**
   a. Separable Gaussian high-pass: blur H→V→subtract = `x - gaussian_blur(x, sigma)`
   b. Rescale by `1/σ_hp[i]` to maintain unit variance

Key insight: **For Gaussian inputs, histogram matching is equivalent to dividing by σ_hp.** This eliminates all sorting infrastructure (histogram count, prefix sum, scatter sort, scatter match — 4 shaders removed).

The per-iteration sigma table is **resolution-independent** (verified 256-2048, max diff < 0.002) because `sigma_spatial = cutoff_divider / (2π)` depends only on `cutoff_divider`, not resolution:
```javascript
const BN_INV_SIGMA_TABLE = [
    1.083608, 1.035389, 1.023437, 1.017777, 1.014435,
    1.012213, 1.010624, 1.009426, 1.008490, 1.007737,
];
```

Validation: rank correlation vs rp reference = 0.999994, std after 10 iters = 0.9994.

**Greyscale optimization**: When greyscale mode is on, blur processes only channel 0 (scalar reads) — ~75% less memory bandwidth.

### Gaussian→Uniform Display
Uses Abramowitz & Stegun erf approximation (error < 1.5e-7) to apply normal CDF, mapping Gaussian values to uniform [0,1] for better visualization. Toggle via U key or button.

## File Structure
```
web_demo_v2/
├── index.html          # Canvas, toolbar with all toggle buttons, stats overlay
├── main.js             # Entry point, game loop, camera, cube, settings persistence
├── renderer.js         # WebGPU device, pipelines, bind groups, dispatch, profiling
├── geometry.js         # Vertex data (cube, floor, quad)
├── shaders.js          # All WGSL shaders (scene, display, 5 warp passes, 5 blue noise passes)
├── test_blue_noise.mjs # Puppeteer test for blue noise
├── sweep_wg.mjs        # Workgroup size benchmark
└── claude_instructions.md  # This manifest
```

## UI Controls
All settings saved to localStorage (`iinw_v2_settings`), restored on page load.

| Button | Hotkey | Description |
|--------|--------|-------------|
| Resolution | R | Cycle 2048→1024→512→256 (recreates renderer) |
| Blue Noise | B | Toggle blue noise post-processing |
| BN×N | N | Cycle blue noise iterations: 2→5→10 (fewer = faster) |
| Grey | G | Greyscale display (channel 0 only, enables scalar blue noise) |
| Uniform | U | Gaussian→Uniform display via normal CDF |
| Retina | T | Toggle retina/HiDPI CSS scaling |
| Interp | I | Toggle CSS interpolation: bilinear vs nearest-neighbor |
| Reset | — | Reset all settings to defaults |

## Performance (M4 Mac, 2048×2048)
- Without blue noise: ~22ms total GPU (~46fps)
- With blue noise (10 iter, vec4): ~20 dispatches, memory-bandwidth limited
- With blue noise (2 iter, greyscale): ~4 dispatches, much faster
- Brownian bridge is the compute bottleneck (~12ms at 2048²)

## Key Constraints
- Float atomics in WGSL via atomicCompareExchangeWeak CAS loop on u32 reinterpretation
- MAX_TICKETS = 24 per source pixel (masterField + areaField = ~50MB at 2048²)
- Stats readback every 60 frames via async mapAsync — doesn't stall render
- GPU timestamp queries for per-phase profiling (when available)

## Lessons Learned
- Gaussian lookup table optimization FAILED: random table access causes cache misses (slower), and table doesn't capture distribution tails (std drops to ~0.945, visible blur). Reverted.
- Moving pixelArea from atomic scatter (brownian) to deterministic recomputation (normalize) improved brownian min-latency significantly.
- 1-ticket fast path in brownian: marginal impact since most pixels have ~4 tickets (bilinear).
- Blue noise MUST be post-processing only — feeding it back corrupts the warp (std elevated, visible artifacts).
- **Sorting → σ_hp rescaling**: For Gaussian inputs, histogram matching = dividing by σ_hp. Eliminated 4 shaders, ~284 lines. Per-iteration sigma table is resolution-independent.
- **σ_hp is NOT constant**: drifts from 0.923 (iter 0) to 0.992 (iter 9) as spectrum evolves from white to blue. Using a fixed constant causes std to blow up to 1.78. Must use per-iteration table.
- **Greyscale scalar path**: Processing only channel 0 reduces blur memory bandwidth by ~75%.
