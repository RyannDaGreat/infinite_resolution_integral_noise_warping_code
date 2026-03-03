# Infinite-Resolution Integral Noise Warping — Manifest

## Algorithm / Goal

ICLR 2025 paper implementation. Takes optical flow maps as input, outputs spatially-uncorrelated white noise images that are temporally-coherent when animated. Each frame looks like random white noise in isolation, but playing frames as video shows motion matching the input flow.

**Core method**: Inverse bilinear warping + Brownian bridge stochastic sampling via Taichi.

## File Structure

```
test.py              — Entry point. Loads flow, runs warping loop, saves .npy + .jpg outputs.
src/utils.py         — Taichi utility functions: fill_noise, ravel/unravel index, sample_brownian_bridge.
src/warp_particle.py — Core algorithm: ParticleWarper class + particle_warp_kernel.
data/{bear,lucia,soapbox}/flows.npy — Sample flow maps [num_frames, H, W, 2].
logs/                — Output directory (generated at runtime).
```

## Usage

```bash
pip install taichi
python test.py -n bear           # Full run (auto-selects CPU on macOS)
python test.py -n bear -f 10     # Limit to 10 frames
python test.py -n bear --cpu     # Force CPU backend
```

`-n` selects experiment name, `-m` selects mode (default: "particle"), `-f` limits frames.

## Key Architecture

1. `test.py` inits Taichi (auto-selects CPU on macOS, GPU+CUDA elsewhere), loads flow, creates `ParticleWarper(H, W, 4, fp=fp)`
2. Per frame: computes deformation = identity_cc - flow, sets warper fields, calls `warper.run()`
3. `particle_warp_kernel` runs 4-phase kernel:
   - **Clear**: zero out buffer, pixel_area, ticket_serial
   - **Backward map**: each source pixel distributes weighted requests to up to 4 target pixels via atomic ticket system
   - **Bridge sample**: per target pixel, normalizes weights, samples via Brownian bridge, sends contributions back to source pixels
   - **Normalize**: `noise = buffer / sqrt(pixel_area)` preserves variance; unassigned pixels get fresh random noise

## Dependencies

- `taichi` (GPU/CPU compute)
- `numpy` (array I/O)
- `PIL` (JPEG output)

## Critical Constraints

- **Sparse SNodes (`ti.root.pointer`)**: Required by original code. Only supported on CUDA. Dense fallback added for non-CUDA.
- **macOS/Metal**: CPU backend forced — Metal has unreliable float atomics, no sparse SNodes, no f64 random
- Flow convention: `(r, c)` where r=top→bottom, c=left→right
- `max_entries = 10000` (CUDA/sparse) or `512` (dense fallback) per pixel for sparse field allocation
- Original tested on: Windows 11, CUDA 11.8, Python 3.10.9, Taichi 1.7.3

## Known Issues / Lessons Learned

- **macOS Metal backend is unreliable for this workload**: Float atomics use CAS emulation that non-deterministically drops updates. Inter-kernel synchronization is missing. No sparse SNodes. No f64 operations. Extensive testing (split kernels, ti.sync, int32 fixed-point atomics) could not produce reliable results on Metal.
- **Brownian bridge endpoint singularity**: When `1-t1` approaches 0, division blows up. Guarded with `denom > eps` check (returns bridge endpoint value `x` directly). Affects f32 more than f64 but is a latent bug in both.
- **CPU on Apple Silicon is fast enough**: M4 multithread CPU runs 81 frames of 512x512 in ~3.5s wall time (all cores used via Taichi).
- **Dense fallback max_entries=512**: Bear data needs at most 23 entries/pixel. 512 is conservative.

## Performance

| Backend | 81 frames 512x512 | Correct |
|---------|-------------------|---------|
| CPU f64 (M4) | ~3.5s | Yes |
| Metal f32 | ~1.4s | No (non-deterministic NaN/inf) |
| CUDA f64 | N/A (no CUDA on macOS) | Expected: Yes |

## Verification

Output should satisfy per-frame: mean ≈ 0, std ≈ 1.0 (white Gaussian noise).

## Success Criteria

- `python test.py -n bear` runs to completion, producing `logs/bear/particle/warped_noise.npy` and visualization JPEGs
- Output noise is spatially white (per-frame) and temporally coherent (across frames)
- Zero NaN/inf in output
