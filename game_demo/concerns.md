# Concerns Log

## 2026-03-03: Initial implementation
- Created full pipeline: scene pass (MRT) → noise warp (nearest-neighbor) → display
- Corrected algorithm from bilinear interpolation to nearest-neighbor per user guidance
  - Bilinear interpolation of noise values shrinks variance and creates blur
  - Reference Taichi impl uses bilinear for AREA DISTRIBUTION (scatter), not value interpolation
  - Nearest-neighbor preserves distribution exactly (weight=1, no variance correction needed)
- Known limitation: expansion creates spatial correlation (unavoidable without scatter/compute)
- Using PCG hash + Box-Muller for shader PRNG
- OpenGL 4.1 constraint: no compute shaders, fragment-only pipeline

## 2026-03-03: Regaussianization + motion vector fixes
- Added Jacobian-based regaussianization to noise warp shader
  - Computes full 2x2 Jacobian determinant of backward warp via finite differences
  - det(J) < 1 → expansion → mix warped with fresh noise: `warped * sqrt(share) + fresh * sqrt(1 - share)`
  - Maintains unit variance (verified: mean≈0, std≈1 over 60 frames)
- Verified motion vectors are correct:
  - Dark spot at rotation axis confirms proper per-vertex velocity computation
  - Magnitude increases outward from rotation axis (correct for rigid rotation)
  - ~17-22% of pixels have non-zero motion (matches cube screen coverage)
- Cube normals verified correct via cross-product check
- Increased cube rotation speed (1→3 rad/s base, targets 1.5-6) for more visible effect
- Visual result: cube silhouette visible as spiral pattern in noise, edges get fresh noise from discontinuity in motion field

## 2026-03-03: Wrong regaussianization attempts
- Jacobian-based approach (det(J) < 1 → mix warped + fresh) was WRONG: linear blending creates intermediate values, not discrete NN behavior
- Binary duplicate detection (check neighbor source coords → fresh if duplicate) was WRONG: this is not what regaussianize does
- User correction: "That is not regaussianization. Look inside noisewarp.py."

## 2026-03-03: Real regaussianize implemented
- Found actual `regaussianize()` in `rp/git/CommonSource/noise_warp.py` lines 298-336
- Algorithm: group by identical channel-0 values → divide by sqrt(count) → add zero-sum foreign noise
- Cannot be implemented in fragment shader (requires scatter: unique, index_add_)
- Solution: CPU-side post-processing with numpy port in `regaussianize.py`
- Pipeline: GPU NN warp → read back → numpy regaussianize → upload → display
- Also created `taichi_warp.py` for Taichi GWTF comparison mode
- Added `--mode` flag to main.py: "glsl" (default) or "taichi"
- Simplified noise_warp.frag to NN-only (removed all regaussianization attempts from shader)

## 2026-03-04: Motion vector bug found and fixed — CRITICAL
- **Bug:** PyGLM's `bytes(mat4)` produces ROW-MAJOR data, but OpenGL/GLSL expects COLUMN-MAJOR
- All 4 matrices (model, view_proj, prev_model, prev_view_proj) were effectively transposed on the GPU
- Effect: motion vectors were wrong sign, wrong magnitude, wrong direction
- The cube still "looked OK" because rotation matrices are orthogonal (transpose ≈ inverse rotation), and the projection gave plausible-looking results by accident
- Previous visual check ("rotation axis visible as dark spot") was misleading — the overall pattern looked right but magnitudes/directions were wrong
- **Fix:** `bytes(glm.transpose(matrix))` before writing to shader uniforms
- Verified with 7 unit tests: translations ±X/±Y all match CPU predictions with 0.0000 relative error
- Added `--mode image` for visual flow verification (warp a JPEG by motion vectors)
- Default resolution reduced to 200x150 (1/4 of 800x600) for faster CPU regaussianize
