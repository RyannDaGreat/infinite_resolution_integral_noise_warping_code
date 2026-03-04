"""ModernGL rendering: context, FBOs, shader programs, draw calls."""

import moderngl
import numpy as np
import glm
from pathlib import Path

from geometry import cube_vertices, quad_vertices


def load_shader(name):
    """
    Pure function. Reads a GLSL shader file from the shaders/ directory.

    Args:
        name (str): Shader filename (e.g. "scene.vert").

    Returns:
        str: Shader source code.

    Examples:
        >>> # load_shader("scene.vert")
    """
    return (Path(__file__).parent / "shaders" / name).read_text()


class Renderer:
    """
    Manages ModernGL context, MRT framebuffer, noise ping-pong, and draw calls.

    Not pure: holds GPU state, renders to framebuffers and screen.

    Args:
        width (int): Framebuffer width in pixels.
        height (int): Framebuffer height in pixels.

    Examples:
        >>> # r = Renderer(800, 600)
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        self.frame_count = 0

        self._init_shaders()
        self._init_geometry()
        self._init_fbos()
        self._init_noise()

    def _init_shaders(self):
        self.scene_prog = self.ctx.program(
            vertex_shader=load_shader("scene.vert"),
            fragment_shader=load_shader("scene.frag"),
        )
        self.noise_warp_prog = self.ctx.program(
            vertex_shader=load_shader("noise_warp.vert"),
            fragment_shader=load_shader("noise_warp.frag"),
        )
        self.display_prog = self.ctx.program(
            vertex_shader=load_shader("display.vert"),
            fragment_shader=load_shader("display.frag"),
        )

    def _init_geometry(self):
        cube_vbo = self.ctx.buffer(cube_vertices().tobytes())
        self.cube_vao = self.ctx.vertex_array(
            self.scene_prog,
            [(cube_vbo, "3f 3f 3f", "in_position", "in_normal", "in_color")],
        )

        quad_data = quad_vertices().tobytes()
        quad_vbo_nw = self.ctx.buffer(quad_data)
        self.noise_quad_vao = self.ctx.vertex_array(
            self.noise_warp_prog,
            [(quad_vbo_nw, "2f 2f", "in_position", "in_texcoord")],
        )
        quad_vbo_disp = self.ctx.buffer(quad_data)
        self.display_quad_vao = self.ctx.vertex_array(
            self.display_prog,
            [(quad_vbo_disp, "2f 2f", "in_position", "in_texcoord")],
        )

    def _init_fbos(self):
        W, H = self.width, self.height

        # Scene MRT: color (RGBA8) + motion vectors (RG32F) + depth
        self.color_tex = self.ctx.texture((W, H), 4)
        self.motion_tex = self.ctx.texture((W, H), 2, dtype="f4")
        self.motion_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.depth_tex = self.ctx.depth_texture((W, H))
        self.scene_fbo = self.ctx.framebuffer(
            color_attachments=[self.color_tex, self.motion_tex],
            depth_attachment=self.depth_tex,
        )

        # Noise ping-pong (RGBA32F)
        self.noise_tex = [
            self.ctx.texture((W, H), 4, dtype="f4"),
            self.ctx.texture((W, H), 4, dtype="f4"),
        ]
        for t in self.noise_tex:
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.noise_fbo = [
            self.ctx.framebuffer(color_attachments=[self.noise_tex[0]]),
            self.ctx.framebuffer(color_attachments=[self.noise_tex[1]]),
        ]
        self.noise_idx = 0  # index of the WRITE target

    def _init_noise(self):
        """Upload initial Gaussian noise to both ping-pong textures."""
        noise = np.random.randn(self.height, self.width, 4).astype(np.float32)
        data = noise.tobytes()
        self.noise_tex[0].write(data)
        self.noise_tex[1].write(data)

    def render_scene(self, model, view_proj, prev_model, prev_view_proj):
        """
        Render the cube to the MRT framebuffer (color + motion vectors).

        Not pure: writes to scene_fbo GPU textures.

        Args:
            model (glm.mat4): Current model matrix.
            view_proj (glm.mat4): Current view-projection matrix.
            prev_model (glm.mat4): Previous frame's model matrix.
            prev_view_proj (glm.mat4): Previous frame's view-projection matrix.
        """
        self.scene_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        # PyGLM bytes() is row-major; OpenGL expects column-major → transpose
        self.scene_prog["u_model"].write(bytes(glm.transpose(model)))
        self.scene_prog["u_view_proj"].write(bytes(glm.transpose(view_proj)))
        self.scene_prog["u_prev_model"].write(bytes(glm.transpose(prev_model)))
        self.scene_prog["u_prev_view_proj"].write(bytes(glm.transpose(prev_view_proj)))

        self.cube_vao.render()

    def warp_noise(self):
        """
        Noise warp pass: nearest-neighbor backward warp from motion vectors.

        Reads motion_tex + prev noise, writes new noise via ping-pong.
        Discrete nearest-neighbor lookup — no bilinear interpolation.
        OOB source pixels get fresh PCG Gaussian noise.

        Not pure: reads/writes GPU textures.
        """
        src_idx = 1 - self.noise_idx
        dst_idx = self.noise_idx

        self.noise_fbo[dst_idx].use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        self.motion_tex.use(location=0)
        self.noise_tex[src_idx].use(location=1)

        self.noise_warp_prog["motion_tex"].value = 0
        self.noise_warp_prog["prev_noise_tex"].value = 1
        self.noise_warp_prog["u_resolution"].value = (
            float(self.width),
            float(self.height),
        )
        self.noise_warp_prog["u_frame"].value = self.frame_count

        self.noise_quad_vao.render()

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        self.noise_idx = 1 - self.noise_idx

    def display(self, mode=0):
        """
        Render final output to the screen.

        Not pure: writes to default framebuffer.

        Args:
            mode (int): 0=noise, 1=color, 2=motion vectors, 3=side-by-side.
        """
        self.ctx.screen.use()
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # The noise texture we just wrote to (after ping-pong swap)
        read_idx = 1 - self.noise_idx
        self.noise_tex[read_idx].use(location=0)
        self.color_tex.use(location=1)
        self.motion_tex.use(location=2)

        self.display_prog["noise_tex"].value = 0
        self.display_prog["color_tex"].value = 1
        self.display_prog["motion_tex"].value = 2
        self.display_prog["u_display_mode"].value = mode

        self.display_quad_vao.render()

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

        self.frame_count += 1

    def read_motion(self):
        """
        Read motion vector texture back to CPU.

        Not pure: reads from GPU.

        Returns:
            np.ndarray: Shape [H, W, 2] float32 in (mv_x, mv_y) UV space,
                row 0 = bottom of screen (OpenGL order).

        Examples:
            >>> # mv = renderer.read_motion()
        """
        data = self.motion_tex.read()
        return np.frombuffer(data, dtype=np.float32).reshape(
            self.height, self.width, 2
        )

    def upload_noise(self, noise):
        """
        Upload a CPU noise array to the current write noise texture.

        Not pure: writes GPU texture, swaps ping-pong index.

        Args:
            noise (np.ndarray): [H, W, 4] float32, row 0 = bottom (OpenGL order).

        Examples:
            >>> # renderer.upload_noise(np.random.randn(H, W, 4).astype(np.float32))
        """
        dst_idx = self.noise_idx
        self.noise_tex[dst_idx].write(noise.astype(np.float32).tobytes())
        self.noise_idx = 1 - self.noise_idx
        self.frame_count += 1

    def read_noise(self):
        """
        Read the current noise texture back to CPU.

        Not pure: reads from GPU.

        Returns:
            np.ndarray: Shape [H, W, 4] float32.

        Examples:
            >>> # noise = renderer.read_noise()
        """
        read_idx = 1 - self.noise_idx
        data = self.noise_tex[read_idx].read()
        return np.frombuffer(data, dtype=np.float32).reshape(
            self.height, self.width, 4
        )
