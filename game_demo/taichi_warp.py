"""
Taichi GWTF wrapper for GPU motion vector integration.

Runs the real particle-based GWTF algorithm on CPU, reading motion vectors
from the GPU scene pass and uploading results back to a GPU noise texture.

Pre-allocates all numpy buffers to avoid per-frame allocation overhead.
"""

import numpy as np


class TaichiWarper:
    """
    Wraps the Taichi-based GWTF warp for use with GPU-generated motion vectors.

    Not pure: allocates Taichi fields, mutates internal state each frame.
    Pre-allocates reusable numpy buffers for zero-allocation per-frame path.

    Args:
        width (int): Framebuffer width.
        height (int): Framebuffer height.
        channels (int): Number of noise channels.

    Examples:
        >>> # tw = TaichiWarper(400, 300, 4)
    """

    def __init__(self, width, height, channels=4):
        from rp.git.CommonSource.inf_int_noise_warp import _make_warper

        self.width = width
        self.height = height
        self.channels = channels

        self.warper, self.identity_cc = _make_warper(height, width, channels)

        # Initial noise in image coordinates (row 0 = top)
        self.prev_noise = np.random.randn(height, width, channels)

        # Pre-allocate reusable buffers
        self._deformation = np.empty((height, width, 2), dtype=np.float64)
        self._gpu_result = np.empty((height, width, channels), dtype=np.float32)

    def get_init_noise_for_gpu(self):
        """
        Return initial noise flipped for OpenGL upload (row 0 = bottom).

        Returns:
            np.ndarray: [H, W, C] float32, row 0 = bottom of screen.

        Examples:
            >>> # noise = tw.get_init_noise_for_gpu()
        """
        return np.flipud(self.prev_noise).astype(np.float32)

    def step(self, motion_gpu):
        """
        Run one GWTF warp step using GPU motion vectors.

        Not pure: mutates internal warper state and pre-allocated buffers.

        Args:
            motion_gpu (np.ndarray): [H, W, 2] motion vectors from GPU in
                (mv_x, mv_y) UV space, row 0 = bottom of screen (OpenGL order).
                mv_x = horizontal displacement (positive right).
                mv_y = vertical displacement (positive UP in screen coords).

        Returns:
            np.ndarray: [H, W, C] float32 warped noise, flipped for GPU upload
                (row 0 = bottom).

        Examples:
            >>> # result = tw.step(motion_from_gpu)
        """
        H, W = self.height, self.width
        deform = self._deformation

        # Build deformation map in-place: identity_cc - flow_rc
        # GPU motion: (mv_x, mv_y) UV space, row 0 = bottom, Y-up
        # Taichi: (row, col) pixel space, row 0 = top
        # flow_rc = (dy_pixels, dx_pixels) where dy = -mv_y * H, dx = mv_x * W
        # deform = identity_cc - flow_rc
        motion_flipped = motion_gpu[::-1]  # view, no copy — flipud
        # row component: identity_row - (-mv_y * H) = identity_row + mv_y * H
        np.multiply(motion_flipped[:, :, 1], H, out=deform[:, :, 0])
        deform[:, :, 0] += self.identity_cc[:, :, 0]
        # col component: identity_col - (mv_x * W) = identity_col - mv_x * W
        np.multiply(motion_flipped[:, :, 0], -W, out=deform[:, :, 1])
        deform[:, :, 1] += self.identity_cc[:, :, 1]

        # Run Taichi kernel (inlined from _warp_step to avoid extra copies)
        self.warper.set_deformation(deform)
        self.warper.set_noise(self.prev_noise)
        self.warper.run()
        self.prev_noise = self.warper.noise_field.to_numpy()

        # Flip result to OpenGL order (row 0 = bottom) into pre-allocated buffer
        np.copyto(self._gpu_result, self.prev_noise[::-1], casting="same_kind")

        return self._gpu_result
