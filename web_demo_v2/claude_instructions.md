# Web Demo V2 — WebGPU Compute Particle Warp

## Algorithm

### Problem
Given a 3D scene rendered in real-time, we want to generate **temporally coherent noise** — noise that moves with the scene's motion but remains spatially uncorrelated (white Gaussian) at every frame. Naive approaches fail: simply advecting a noise texture through optical flow introduces spatial correlation (blurring), while generating fresh noise each frame destroys temporal coherence (flickering).

The key insight from Deng et al. ("Infinite-Resolution Integral Noise Warping for Diffusion Models", ICLR 2025) is to model noise as a **continuous white-noise field** integrated over pixel areas. When pixels deform under motion, the integral regions change shape, but the underlying continuous field is unchanged. The algorithm samples from the conditional distribution of the deformed integrals given the previous frame's values, using **Brownian bridge** stochastic sampling to decompose each pixel's contribution into independent Gaussian increments.

**Reference**: Deng et al., ICLR 2025. [Paper](https://openreview.net/forum?id=Y6LPWBo2HP), [Code](https://github.com/yitongdeng-projects/infinite_resolution_integral_noise_warping_code). This implementation uses the **particle-based variant** (Algorithm 3), which treats pixels as particles distributed to grid cells via bilinear weighting. The paper also describes a grid-based variant (Algorithm 2) that models deformed regions as octagons — not used here because it breaks with non-injective deformation maps.

### Mathematical Foundation

**Continuous model.** Let W(x) be a white-noise field on R². A pixel p integrates W over its area A_p:

    noise[p] = (1/√|A_p|) ∫_{A_p} W(x) dx

For unit-area pixels, this is standard Gaussian white noise: mean 0, variance 1, spatially uncorrelated.

**Deformation.** Under optical flow, pixel p at time t maps to a deformed region at time t+1. The deformed region overlaps multiple destination pixels. We need to compute what fraction of each source pixel's "noise mass" goes to each destination pixel, then sample new noise values that are (a) consistent with the source values and (b) still white Gaussian.

**Bilinear backward mapping.** For each destination pixel d, the deformation map points to a fractional position in source space. Bilinear interpolation distributes d's request across 4 neighboring source pixels with weights w₁...w₄ (summing to 1). Each source pixel s accumulates a total request `total_area[s] = Σ w_i` from all destination pixels that land near it.

**Brownian bridge.** The crucial step: for each source pixel s with current noise value x and K destination requests (with normalized weights t₁...t_K, Σt_k = 1), we need K values that sum to x·√(total_area) and are each Gaussian with the correct variance. This is exactly a **Brownian bridge** B(t) conditioned on B(0)=0 and B(1)=x:

    Sample B(t₂) | B(t₁) = q:
        μ = ((1-t₂)/(1-t₁))·q + ((t₂-t₁)/(1-t₁))·x
        σ² = (t₂-t₁)(1-t₂)/(1-t₁)
        B(t₂) = μ + σ·z,  z ~ N(0,1)

    The increment B(t₂) - B(t₁) is the contribution from source s to the destination pixel associated with request k.

**Normalization.** After scattering, each destination pixel d has accumulated contributions from its source pixels. The accumulated variance equals `pixel_area[d]` (the sum of all fractional weights assigned to d). Dividing by `√pixel_area[d]` restores unit variance. Destination pixels with zero coverage (disoccluded regions) receive fresh Gaussian noise.

### Implementation: WebGPU Compute Pipeline

All computation runs on GPU via WebGPU compute shaders. Zero CPU-GPU copies per frame — the only CPU work is ~10 floats of matrix math + command buffer submission.

#### Data Flow
```
GPU render (MRT: color+motion) → build deformation → clear → backward map → brownian bridge → normalize → [blue noise post-process] → display
```
All storage buffers stay GPU-resident. No readPixels, no texSubImage2D.

#### Pipeline Phases (per frame)
1. **Scene render** — MRT: rgba8 color + rgba32float motion vectors + depth
2. **Build deformation** — motion texture → deformation buffer (dx,dy per pixel). Maps each pixel to its source position: `source_pos = pixel_center - motion_vector`.
3. **Clear** — zero intermediate buffers (buffer, pixelArea, ticketCount)
4. **Backward map** — per-dest-pixel: bilinear interpolation of source position yields 4 (src, weight) pairs. Each pair claims a ticket slot on its source pixel via `atomicAdd(&ticketCount[src], 1)`, then writes the dest index and weight into `masterField[src, ticket]` and `areaField[src, ticket]`.
5. **Brownian bridge** — per-source-pixel: reads all K tickets sequentially, computes `total_request = Σ weights`, then samples the Brownian bridge (Box-Muller PRNG for Gaussian samples) to produce K increments. Each increment is scattered back to its destination pixel via float atomics (CAS loop on u32 reinterpretation). Also accumulates `pixel_area[dest] += normalized_weight`.
6. **Normalize** — per-dest-pixel: if `pixelArea > 0`, scale by `1/√(pixelArea)` to restore unit variance; else fill with fresh Gaussian noise. pixelArea is recomputed deterministically from the deformation map (no atomic needed — avoids contention from phase 5).
7. **Blue noise (optional post-process)** — converts white Gaussian noise to blue Gaussian noise for perceptually better visualization. See below.
8. **Display** — reads noise from storage buffer, supports greyscale + Gaussian→uniform display flags.

### Blue Noise Post-Processing

#### CRITICAL CONSTRAINT
**Blue noise is POST-PROCESSING ONLY.** It must NEVER feed back into the warp. The warp must always receive clean white Gaussian noise — blue noise input would violate the Brownian bridge's assumption that the source noise is spatially uncorrelated.

Implementation: backup noiseBuf → run blue noise (modifies noiseBuf for display) → display → restore noiseBuf from backup.

#### Algorithm
Port of `rp.convert_to_blue_noise` — iterative spectral reshaping. Each iteration applies a Gaussian high-pass filter (removing low-frequency content) then rescales to maintain unit variance:

1. **For each iteration i ∈ {0, ..., N-1} (N = 2, 5, or 10, user configurable):**
   a. Separable Gaussian high-pass: `hp(x) = x - gaussian_blur(x, σ_spatial)` where `σ_spatial = cutoff_divider / (2π)`
   b. Rescale: `x ← hp(x) / σ_hp[i]` to restore unit variance

**Why rescaling replaces histogram matching.** The original algorithm uses histogram matching (sorting + rank-mapping) after each high-pass to restore the marginal distribution. But for **Gaussian inputs**, high-pass filtering produces a Gaussian output (linear combination of Gaussians is Gaussian), so histogram matching to N(0,1) is equivalent to simply dividing by the standard deviation σ_hp. This eliminates all sorting infrastructure (histogram count, prefix sum, scatter sort, scatter match — 4 shaders, ~284 lines removed).

**Why σ_hp varies per iteration.** The high-pass filter's output std depends on the input spectrum. At iteration 0, the input is white noise (flat spectrum) and the high-pass removes significant energy (σ_hp ≈ 0.923). By iteration 9, the spectrum is already blue (little low-frequency content to remove), so σ_hp ≈ 0.992. Using a fixed constant causes variance to drift — after 10 iterations with a fixed σ_hp, std blows up to ~1.78.

The per-iteration sigma table is **resolution-independent** (verified 256–2048, max diff < 0.002) because `σ_spatial = cutoff_divider / (2π)` depends only on `cutoff_divider`, not resolution:
```javascript
const BN_INV_SIGMA_TABLE = [  // 1/σ_hp[i] for i = 0..9
    1.083608, 1.035389, 1.023437, 1.017777, 1.014435,
    1.012213, 1.010624, 1.009426, 1.008490, 1.007737,
];
```

Validation: rank correlation vs `rp.convert_to_blue_noise` reference = 0.999994, std after 10 iters = 0.9994.

**Greyscale optimization**: When greyscale mode is on, blur processes only channel 0 (scalar reads) — ~75% less memory bandwidth.

### Gaussian→Uniform Display
Uses Abramowitz & Stegun erf approximation (error < 1.5e-7) to apply the normal CDF Φ(x), mapping Gaussian values to uniform [0,1] for better visualization. Toggle via U key or button.

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
| Resolution | P | Cycle 2048→1024→512→256 (recreates renderer). Not R — Cmd+R is browser refresh. |
| Blue Noise | B | Toggle blue noise post-processing |
| BN×N | N | Cycle blue noise iterations: 2→5→10 (fewer = faster) |
| Grey | G | Greyscale display (channel 0 only, enables scalar blue noise) |
| Uniform | U | Gaussian→Uniform display via normal CDF |
| Retina | T | Toggle retina/HiDPI CSS scaling |
| Interp | I | Toggle CSS interpolation: bilinear vs nearest-neighbor. Default: nearest, because bilinear blurs pixel boundaries and defeats pixel-perfect visualization. |
| Round | O | Cycle motion rounding: None→All→>1. **Why**: tests how the warp behaves with quantized (integer-pixel) motion — simulates game-like scenarios where objects move by whole pixels. "Round All" snaps all motion to nearest pixel. "Round >1" preserves sub-pixel motion but quantizes larger displacements. Applied in the build-deformation shader before the warp. |
| Threshold | H+slider | Toggle on/off (H key), slider sets cutoff [0.0, 1.0]. Pixels above threshold are white, below are black. Applied after display flags — with Uniform on, it's implicitly percentile-based (CDF maps Gaussian to [0,1]). **Why**: simulates dithering — reveals blue noise spatial quality (even spread vs white noise clumping). Single shader comparison, zero cost. |
| Reset | — | Reset all settings to defaults |

### Canvas Sizing — Pixel-Perfect Constraint
The canvas CSS size MUST exactly match the render resolution (or render/DPR for retina). Any border, padding, or sizing mismatch causes the browser to interpolate pixels, creating missed scanlines and moiré artifacts. For this reason:
- Canvas uses `outline` (paints outside the box) instead of `border` (paints inside, shrinking content area)
- No padding on the canvas element
- CSS width/height are set to exactly `W/dpr × H/dpr` pixels

### Settings Persistence
Settings are saved to localStorage with a version number (`SETTINGS_VERSION`). When defaults change, bumping the version force-resets all users to new defaults. This ensures default changes (like switching from bilinear to nearest interpolation) actually take effect. Hotkeys skip when Cmd/Ctrl is held, preventing browser shortcuts (Cmd+R = refresh) from triggering UI toggles.

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
