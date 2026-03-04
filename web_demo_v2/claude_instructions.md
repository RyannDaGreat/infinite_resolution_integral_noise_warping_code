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
Port of `rp.convert_to_blue_noise` — alternating projections:
1. **Sort initial values** (counting sort: quantize → atomicAdd histogram → prefix sum → scatter)
2. **For each iteration:**
   a. Separable Gaussian high-pass: blur H→V→subtract = `x - gaussian_blur(x, sigma)`
   b. Counting sort on high-passed values
   c. Histogram matching: assign sorted target values by high-pass rank

Key math: FFT high-pass `H(u,v) = 1 - exp(-d²/(2D₀²))` is equivalent to spatial `x - gaussian_blur(x, sigma)` where `sigma = N/(2π·D₀)`. At cutoff_divider=8, sigma≈1.27, kernel radius=4 (9 taps). Much cheaper than 2D FFT.

### Blue Noise LUT Optimization (PLANNED)
The sorting step is expensive. Since the distribution is large (~4M samples at 2048²), the mapping from Gaussian to blue noise values should be stable across frames. Plan:
1. Run full blue noise conversion on first frame (or a reference frame)
2. Record the value→value mapping as a lookup table (e.g., 4096 entries)
3. On subsequent frames, apply the LUT instead of sorting — O(1) per pixel instead of O(N) total
4. The LUT should be generated programmatically at init time, not hardcoded
5. Since blue noise doesn't feed back into warp, approximate distribution is acceptable

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
| Blue Noise | B | Toggle blue noise post-processing (10 iterations) |
| Grey | G | Greyscale display (show channel 0 only) |
| Uniform | U | Gaussian→Uniform display via normal CDF |
| Retina | T | Toggle retina/HiDPI CSS scaling |
| Interp | I | Toggle CSS interpolation: bilinear vs nearest-neighbor |
| Reset | — | Reset all settings to defaults |

## Performance (M4 Mac, 2048×2048)
- Without blue noise: ~22ms total GPU (~46fps)
- With blue noise (10 iter): ~62ms total GPU (~16fps)
- Brownian bridge is the compute bottleneck (~12ms at 2048²)

## Key Constraints
- Float atomics in WGSL via atomicCompareExchangeWeak CAS loop on u32 reinterpretation
- MAX_TICKETS = 24 per source pixel (masterField + areaField = ~50MB at 2048²)
- Prefix sum for histogram is serial (1 thread per channel, 4096 bins) — could parallelize but negligible cost
- Stats readback every 60 frames via async mapAsync — doesn't stall render
- GPU timestamp queries for per-phase profiling (when available)

## Lessons Learned
- Gaussian lookup table optimization FAILED: random table access causes cache misses (slower), and table doesn't capture distribution tails (std drops to ~0.945, visible blur). Reverted.
- Moving pixelArea from atomic scatter (brownian) to deterministic recomputation (normalize) improved brownian min-latency significantly.
- 1-ticket fast path in brownian: marginal impact since most pixels have ~4 tickets (bilinear).
- Blue noise MUST be post-processing only — feeding it back corrupts the warp (std elevated, visible artifacts).
