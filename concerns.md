# Concerns — Infinite-Resolution Integral Noise Warping

## 2026-03-03

### Initial run `python test.py -n bear` — FAILED

**Error**: `TaichiRuntimeError: Pointer SNode is not supported on this backend.`
**Location**: `src/warp_particle.py:110`
**Root cause**: Metal backend doesn't support sparse pointer SNodes.
**Fix**: try/except fallback to dense layout with reduced max_entries=512.

### Metal f64 random — FAILED

**Error**: `rand only support 32-bit type` (SPIR-V codegen)
**Root cause**: Metal SPIR-V doesn't support f64 random generation.
**Fix**: Use f32 on Metal, f64 on CUDA/CPU. Added configurable `fp` parameter.

### Metal f32 NaN — FAILED (non-deterministic)

**Error**: NaN/inf values appearing non-deterministically in output frames.
**Investigation**:
1. CPU f32 with same code: works perfectly (zero NaN, correct stats)
2. Metal f32: non-deterministic NaN (4/5 runs clean, 1/5 with NaN for 10 frames)
3. Added `ti.sync()` between split kernels: reduced but did not eliminate NaN
4. Confirmed via research (Taichi #1174): Metal lacks memory order enforcement
5. Tried int32 fixed-point atomics: eliminated NaN but introduced variance blowup (std=3.0 instead of 1.0)
6. Tested Metal int32 atomics in isolation: they work correctly

**Root cause**: Taichi Metal backend has unreliable float atomic operations (CAS-based emulation that drops updates under contention). Additionally, inter-kernel synchronization is incomplete.

**Research findings** (from Taichi GitHub issues):
- #8762: Metal 4GB memory limit causes silent data corruption
- #1174: Metal lacks memory fence equivalent to CUDA __threadfence()
- #8745: Field operations produce incorrect results on Metal
- Metal lacks native float atomics; must use CAS loop

### Final solution: CPU on macOS

**Decision**: Force CPU backend on macOS. Metal Taichi is too unreliable for this algorithm (requires scatter-pattern float atomic accumulation).

**Performance**: 81 frames 512x512 in ~3.5s on M4 (multithread CPU). Acceptable.
**Correctness**: Zero NaN, mean≈0, std≈1.0 on all frames. Verified across multiple runs.

### Brownian bridge endpoint guard

Added `denom > eps` check in `sample_brownian_bridge` to avoid division by zero when `1-t1` is near zero. This is mathematically correct (bridge endpoint B(1)=x is deterministic) and prevents potential f32 blowup.
