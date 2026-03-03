"""
Simple API for infinite-resolution integral noise warping.

    >>> import numpy as np
    >>> flows = [np.zeros((64, 64, 2)) for _ in range(5)]
    >>> init = np.random.randn(64, 64, 4)
    >>> noises = list(warp_noise(init, flows))
    >>> len(noises)  # init + 5 warped frames
    6
"""

import sys
import os
import platform
import numpy as np

# Taichi init must happen before importing warp_particle (which defines @ti.kernel decorators
# that compile lazily, but field construction in ParticleWarper requires init).
import taichi as ti

_taichi_initialized = False

def _ensure_taichi_init():
    """
    Initializes Taichi if not already done.
    macOS: CPU backend (Metal is unreliable for this algorithm).
    Linux/Windows: CUDA GPU backend.

    Side effects: calls ti.init() once per process.
    """
    global _taichi_initialized
    if _taichi_initialized:
        return
    if platform.system() == 'Darwin':
        ti.init(arch=ti.cpu, debug=False, default_fp=ti.f64, random_seed=0)
    else:
        ti.init(arch=ti.gpu, device_memory_GB=4.0, debug=False, default_fp=ti.f64, random_seed=0)
    _taichi_initialized = True

# Add src/ to path for warp_particle imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


def warp_noise(init_noise, flows):
    """
    Warp noise through a sequence of optical flow fields.

    Each output frame is spatially-uncorrelated white noise that follows the
    motion described by the input flows. Uses Brownian bridge sampling to
    preserve noise statistics through warping.

    Args:
        init_noise: np.ndarray of shape [H, W, C] — initial Gaussian noise.
        flows: iterable of np.ndarray, each [H, W, 2] — optical flow per frame.
               Convention: (row, col) where row=top→bottom, col=left→right.
               Can be a generator.

    Yields:
        np.ndarray of shape [H, W, C] — warped noise frames.
        First yield is init_noise. Subsequent yields are warped by each flow.

    Examples:
        >>> # With a list of flows
        >>> for frame in warp_noise(init_noise, flow_list):
        ...     process(frame)
        >>> # With a generator of flows
        >>> for frame in warp_noise(init_noise, load_flows()):
        ...     process(frame)
    """
    _ensure_taichi_init()
    from warp_particle import ParticleWarper

    H, W, C = init_noise.shape

    # Cell-center identity map
    ii, jj = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing='ij')
    identity_cc = np.stack((ii, jj), axis=-1)

    warper = ParticleWarper(H, W, C, fp=ti.f64)

    yield init_noise

    prev_noise = init_noise
    for flow in flows:
        warper.set_deformation(identity_cc - flow)
        warper.set_noise(prev_noise)
        warper.run()
        prev_noise = warper.noise_field.to_numpy()
        yield prev_noise
