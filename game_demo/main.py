"""
Real-time GLSL noise warping demo.

Renders a spinning cube with ground-truth motion vectors from 3D geometry,
then warps Gaussian noise using nearest-neighbor backward warp + regaussianize.

Modes:
    glsl   — GPU nearest-neighbor warp + CPU regaussianize (default)
    taichi — Full Taichi particle-based GWTF
    image  — Warp a JPEG by cube's motion vectors (visual flow verification)

Controls:
    WASD / Space / Shift  — Move camera
    Mouse (click to capture, ESC to release) — Look around
    1  — Noise view
    2  — Color view
    3  — Motion vector view
    4  — Side-by-side (color | noise)
    5  — Raw image view (for image mode)
    Q / ESC — Quit
"""

import pygame
import numpy as np
import glm
import fire

from renderer import Renderer
from regaussianize import regaussianize


def load_image_for_gpu(path, width, height):
    """
    Load a JPEG/PNG and prepare it for GPU upload as an RGBA32F texture.

    Not pure: reads file from disk.

    Args:
        path (str): Path to image file.
        width (int): Target width.
        height (int): Target height.

    Returns:
        np.ndarray: [H, W, 4] float32 in [0, 1], row 0 = bottom (OpenGL order).

    Examples:
        >>> # img = load_image_for_gpu("dog.jpg", 800, 600)
    """
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((width, height))
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3], row 0 = top
    arr = np.flipud(arr)  # row 0 = bottom (OpenGL)
    alpha = np.ones((height, width, 1), dtype=np.float32)
    return np.concatenate([arr, alpha], axis=-1)  # [H, W, 4]


def make_test_pattern(width, height):
    """
    Pure function. Generate a colored gradient test pattern.

    4 colored quadrants with a grid overlay for verifying warp direction.

    Args:
        width (int): Image width.
        height (int): Image height.

    Returns:
        np.ndarray: [H, W, 4] float32 in [0, 1], row 0 = bottom (OpenGL order).

    Examples:
        >>> p = make_test_pattern(100, 100)
        >>> p.shape
        (100, 100, 4)
    """
    y = np.linspace(0, 1, height)[:, None]
    x = np.linspace(0, 1, width)[None, :]
    r = x * np.ones_like(y)
    g = y * np.ones_like(x)
    b = (1 - x) * np.ones_like(y)
    # Grid overlay every 32 pixels
    gx = (np.arange(width) % 32 < 2)[None, :]
    gy = (np.arange(height) % 32 < 2)[:, None]
    grid = (gx | gy).astype(np.float32) * 0.3
    r = np.clip(r - grid, 0, 1).astype(np.float32)
    g = np.clip(g - grid, 0, 1).astype(np.float32)
    b = np.clip(b - grid, 0, 1).astype(np.float32)
    a = np.ones((height, width), dtype=np.float32)
    return np.stack([r, g, b, a], axis=-1)  # already row 0 = bottom for OpenGL


class FPSCamera:
    """
    First-person camera with WASD + mouse look.

    Not pure: stores mutable position/rotation state.
    """

    def __init__(self, position=None, yaw=-90.0, pitch=-10.0):
        self.position = glm.vec3(position or (0, 1, 4))
        self.yaw = yaw
        self.pitch = pitch
        self.speed = 3.0
        self.sensitivity = 0.15

    def process_mouse(self, dx, dy):
        """Update yaw/pitch from mouse movement delta."""
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def process_keys(self, keys, dt):
        """Move camera based on held keys."""
        fwd = self._forward()
        right = glm.normalize(glm.cross(fwd, glm.vec3(0, 1, 0)))
        v = self.speed * dt
        if keys[pygame.K_w]:
            self.position += fwd * v
        if keys[pygame.K_s]:
            self.position -= fwd * v
        if keys[pygame.K_a]:
            self.position -= right * v
        if keys[pygame.K_d]:
            self.position += right * v
        if keys[pygame.K_e] or keys[pygame.K_SPACE]:
            self.position.y += v
        if keys[pygame.K_q] or keys[pygame.K_LSHIFT]:
            self.position.y -= v

    def _forward(self):
        """Compute forward direction from yaw/pitch."""
        ry = glm.radians(self.yaw)
        rp = glm.radians(self.pitch)
        return glm.normalize(glm.vec3(
            glm.cos(rp) * glm.cos(ry),
            glm.sin(rp),
            glm.cos(rp) * glm.sin(ry),
        ))

    def view_matrix(self):
        """
        Returns:
            glm.mat4: View matrix.

        Examples:
            >>> # cam = FPSCamera(); cam.view_matrix()
        """
        f = self._forward()
        return glm.lookAt(self.position, self.position + f, glm.vec3(0, 1, 0))


class SpinningCube:
    """
    Cube with smooth random axis/speed changes.

    Not pure: stores mutable rotation state.
    """

    def __init__(self):
        self.angle = 0.0
        self.axis = glm.normalize(glm.vec3(1, 1, 0))
        self.speed = 1.5
        self.target_axis = self._random_axis()
        self.target_speed = np.random.uniform(0.75, 3.0)
        self.change_timer = 0.0
        self.change_interval = 3.0
        self.prev_model = glm.mat4(1.0)

    def _random_axis(self):
        v = np.random.randn(3)
        v /= np.linalg.norm(v) + 1e-8
        return glm.vec3(*v)

    def update(self, dt):
        """Save previous model matrix, then update rotation."""
        self.prev_model = self.model_matrix()

        t = min(dt * 2.0, 1.0)
        self.axis = glm.normalize(glm.mix(self.axis, self.target_axis, t))
        self.speed += (self.target_speed - self.speed) * t
        self.angle += self.speed * dt

        self.change_timer += dt
        if self.change_timer > self.change_interval:
            self.change_timer = 0.0
            self.target_axis = self._random_axis()
            self.target_speed = np.random.uniform(0.75, 3.0)
            self.change_interval = np.random.uniform(2.0, 5.0)

    def model_matrix(self):
        """
        Returns:
            glm.mat4: Current rotation matrix.

        Examples:
            >>> # SpinningCube().model_matrix()
        """
        return glm.rotate(glm.mat4(1.0), self.angle, self.axis)


def run(width=200, height=150, fps=60, mode="glsl", image=None):
    """
    Launch the interactive noise warping demo.

    Not pure: creates window, runs game loop, handles input.

    Args:
        width (int): Window width.
        height (int): Window height.
        fps (int): Target framerate.
        mode (str): "glsl", "taichi", or "image".
        image (str): Path to JPEG/PNG for image mode. If None, uses test pattern.

    Examples:
        >>> # run()
        >>> # run(mode="image", image="dog.jpg")
    """
    assert mode in ("glsl", "taichi", "image"), f"mode must be 'glsl', 'taichi', or 'image', got {mode!r}"

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True
    )

    pygame.display.set_mode(
        (width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    )
    pygame.display.set_caption(f"Noise Warping Demo [{mode}]")
    clock = pygame.time.Clock()

    renderer = Renderer(width, height)
    camera = FPSCamera()
    cube = SpinningCube()

    # Mode-specific init
    taichi_warper = None
    if mode == "taichi":
        from taichi_warp import TaichiWarper
        taichi_warper = TaichiWarper(width, height, channels=4)
        renderer.upload_noise(taichi_warper.get_init_noise_for_gpu())
    elif mode == "image":
        if image is not None:
            img_data = load_image_for_gpu(image, width, height)
        else:
            print("No --image path given, using test pattern")
            img_data = make_test_pattern(width, height)
        renderer.upload_noise(img_data)

    proj = glm.perspective(glm.radians(60.0), width / height, 0.1, 100.0)
    prev_view_proj = proj * camera.view_matrix()

    mouse_captured = False
    # Default display mode: 4 (raw) for image mode, 0 (noise) otherwise
    display_mode = 4 if mode == "image" else 0
    running = True

    while running:
        dt = min(clock.tick(fps) / 1000.0, 0.1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if mouse_captured:
                        pygame.event.set_grab(False)
                        pygame.mouse.set_visible(True)
                        mouse_captured = False
                    else:
                        running = False
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    display_mode = event.key - pygame.K_1
            elif event.type == pygame.MOUSEBUTTONDOWN and not mouse_captured:
                pygame.event.set_grab(True)
                pygame.mouse.set_visible(False)
                mouse_captured = True
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                pygame.display.set_mode(
                    (width, height),
                    pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
                )
                renderer.resize(width, height)
                proj = glm.perspective(
                    glm.radians(60.0), width / height, 0.1, 100.0
                )
                if mode == "taichi":
                    taichi_warper = TaichiWarper(width, height, channels=4)
                    renderer.upload_noise(taichi_warper.get_init_noise_for_gpu())
                elif mode == "image":
                    if image is not None:
                        img_data = load_image_for_gpu(image, width, height)
                    else:
                        img_data = make_test_pattern(width, height)
                    renderer.upload_noise(img_data)
            elif event.type == pygame.MOUSEMOTION and mouse_captured:
                camera.process_mouse(event.rel[0], event.rel[1])

        camera.process_keys(pygame.key.get_pressed(), dt)
        cube.update(dt)

        view_proj = proj * camera.view_matrix()
        model = cube.model_matrix()

        # Scene pass (always GPU)
        renderer.render_scene(model, view_proj, cube.prev_model, prev_view_proj)

        if mode == "glsl":
            # GPU nearest-neighbor warp
            renderer.warp_noise()
            # CPU regaussianize: read back → regaussianize → upload
            warped_hwc = renderer.read_noise()  # [H, W, 4] float32, row 0 = bottom
            warped_chw = warped_hwc.transpose(2, 0, 1)  # [C, H, W]
            regaussed_chw, _ = regaussianize(warped_chw)
            regaussed_hwc = regaussed_chw.transpose(1, 2, 0)  # [H, W, C]
            renderer.upload_noise(regaussed_hwc)
        elif mode == "taichi":
            # Read motion vectors from GPU, run Taichi GWTF, upload result
            motion = renderer.read_motion()  # [H, W, 2] float32, row 0 = bottom
            result = taichi_warper.step(motion)  # [H, W, C] float32, row 0 = bottom
            renderer.upload_noise(result)
        elif mode == "image":
            # Just NN warp, no regaussianize — visual verification of flow
            renderer.warp_noise()

        renderer.display(mode=display_mode)

        prev_view_proj = view_proj
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    fire.Fire(run)
