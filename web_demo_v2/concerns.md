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

### Known risks
- Blue noise at 2048² is slow (~62ms, 16fps). LUT optimization could fix this.
- Resolution toggle destroys and recreates renderer — may cause brief visual glitch
- Greyscale is currently display-only; actual single-channel compute would save ~4x on blue noise sorting
- `destroy()` method may not release all GPU resources if async operations are in flight

### Pending tasks
- Verify blue noise backup/restore produces clean Gaussian stats (mean~0, std~1.0)
- Verify blue noise GPU output matches Python reference numerically
- Implement blue noise LUT optimization
- Spacebar pause feature
- Write .claude_todo.md sync
