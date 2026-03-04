"""
Taichi GWTF wrapper for GPU motion vector integration.

Runs the real particle-based GWTF algorithm on CPU, reading motion vectors
from the GPU scene pass and uploading results back to a GPU noise texture.
"""

import numpy as np


class TaichiWarper:
    """
    Wraps the Taichi-based GWTF warp for use with GPU-generated motion vectors.

    Not pure: allocates Taichi fields, mutates internal state each frame.

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

        Not pure: mutates internal warper state.

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
        from rp.git.CommonSource.inf_int_noise_warp import _warp_step

        # Convert GPU motion vectors to Taichi flow format:
        # GPU: row 0 = bottom, UV space, Y-up
        # Taichi: row 0 = top, pixel space, (dx, dy) where dy is positive-down
        motion_img = np.flipud(motion_gpu)  # row 0 = top (image coords)

        flow_dx = motion_img[:, :, 0] * self.width   # UV → pixels, horizontal
        flow_dy = -motion_img[:, :, 1] * self.height  # UV → pixels, flip Y
        flow_dxdy = np.stack([flow_dx, flow_dy], axis=-1)

        self.prev_noise = _warp_step(
            self.warper, self.identity_cc, self.prev_noise, flow_dxdy
        )

        # Flip back to OpenGL order for GPU upload
        return np.flipud(self.prev_noise).astype(np.float32)
