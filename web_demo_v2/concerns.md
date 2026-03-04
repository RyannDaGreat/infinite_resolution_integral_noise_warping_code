# Web Demo V2 — Concerns & Progress

## 2026-03-04 — Session 2

### Blue noise backup/restore fix
- **Issue**: Blue noise modified noiseBuf in-place, corrupting next frame's warp input (std elevated to 1.164)
- **Fix**: Added `copyBufferToBuffer(noiseBuf → bnBackupBuf)` before blue noise, `copyBufferToBuffer(bnBackupBuf → noiseBuf)` after display
- **Status**: Implemented, needs verification

### UI controls batch
- Added: resolution toggle (R), retina toggle (T), interpolation toggle (I), greyscale (G), uniform display (U), reset button
- All settings persisted to localStorage
- Hint bar restyled with dark translucent background to match stats bar

### Blue noise LUT optimization idea (from user)
- User suggests: instead of per-frame counting sort, empirically derive a value→value mapping table
- At 2048² (4M samples), the Gaussian→blue-noise mapping should be extremely stable
- Could be a 4096-entry LUT generated at init time
- Since blue noise is post-processing only and doesn't accumulate, approximate distribution is fine
- Implementation: record mapping from one reference blue noise conversion, apply as LUT on subsequent frames
- LUT generation code must be included (Python or JS), not hardcoded values
- **Status**: Not started. Recorded in manifest.

### Gaussian→Uniform display
- User wants proper visualization: apply inverse normal CDF to map Gaussian values to [0,1]
- Implemented using Abramowitz & Stegun erf approximation (error < 1.5e-7)
- Toggle via U key or button

### Blue noise sorting → σ_hp rescaling optimization
- **Key insight**: For Gaussian inputs, histogram matching = dividing by σ_hp
- **But**: σ_hp is NOT constant across iterations. It evolves from 0.923 (iter 0) to 0.992 (iter 9)
  as the spectrum shifts from white to blue
- **Solution**: Pre-computed per-iteration sigma table (10 values), resolution-independent
- **Validation results** (GPU-matched radius-4 kernel vs rp.convert_to_blue_noise):
  - Rank correlation: 0.999994 — essentially identical spatial pattern
  - Power spectrum: overlapping curves — same frequency characteristics
  - Std after 10 iters: 0.9994 (table) vs 1.0000 (reference)
  - Cross-resolution max diff: 0.001 (256-2048), confirming resolution independence
- **Impact**: Removed 4 sorting shaders + buffers, reduced dispatches from 54→20 per frame
- **Status**: Validated and implemented

### Known risks
- Resolution toggle destroys and recreates renderer — may cause brief visual glitch
- Greyscale is currently display-only; actual single-channel compute would save ~4x on blue noise
- `destroy()` method may not release all GPU resources if async operations are in flight

### Pending tasks
- Greyscale compute optimization (single-channel blue noise)
- WASM fallback
- Continue BULLDOG optimization
