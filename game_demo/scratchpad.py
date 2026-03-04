"""Test both GLSL and Taichi modes end-to-end: Gaussian stats + no duplicates."""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pygame
import numpy as np
import glm

pygame.init()
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
W, H = 200, 150
pygame.display.set_mode((W, H), pygame.OPENGL | pygame.DOUBLEBUF)

from renderer import Renderer
from regaussianize import regaussianize


def run_glsl_test(n_frames=30):
    print("\n=== GLSL Mode Test (%d frames) ===" % n_frames)
    renderer = Renderer(W, H)
    proj = glm.perspective(glm.radians(60.0), W / H, 0.1, 100.0)
    view = glm.lookAt(glm.vec3(0, 0, 3), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

    angle = 0.0
    speed = 3.0
    axis = glm.normalize(glm.vec3(1, 1, 0))
    prev_model = glm.mat4(1.0)
    prev_vp = proj * view

    for frame in range(n_frames):
        dt = 1.0 / 60.0
        prev_model_save = glm.rotate(glm.mat4(1.0), angle, axis)
        angle += speed * dt
        model = glm.rotate(glm.mat4(1.0), angle, axis)
        vp = proj * view

        renderer.render_scene(model, vp, prev_model_save, prev_vp)
        renderer.warp_noise()

        # CPU regaussianize
        warped = renderer.read_noise()  # [H, W, 4]
        warped_chw = warped.transpose(2, 0, 1)
        regaussed_chw, counts = regaussianize(warped_chw)
        regaussed = regaussed_chw.transpose(1, 2, 0)
        renderer.upload_noise(regaussed)

        prev_vp = vp

        if frame in [0, 9, 19, 29]:
            noise = regaussed
            mean = noise.mean()
            std = noise.std()
            n_unique_ch0 = len(np.unique(noise[:, :, 0]))
            n_pixels = W * H
            dup_ratio = 1.0 - n_unique_ch0 / n_pixels

            print("  Frame %2d: mean=%+.4f std=%.4f unique_ch0=%d/%d (%.1f%% dups)" % (
                frame, mean, std, n_unique_ch0, n_pixels, 100 * dup_ratio))

            assert abs(mean) < 0.15, "FAIL: |mean| = %.4f > 0.15" % abs(mean)
            assert 0.8 < std < 1.2, "FAIL: std = %.4f outside [0.8, 1.2]" % std
            assert dup_ratio < 0.01, "FAIL: %.1f%% duplicates after regaussianize" % (100 * dup_ratio)

    print("  GLSL mode: ALL CHECKS PASSED")


def run_taichi_test(n_frames=30):
    print("\n=== Taichi Mode Test (%d frames) ===" % n_frames)
    from taichi_warp import TaichiWarper

    renderer = Renderer(W, H)
    warper = TaichiWarper(W, H, channels=4)
    renderer.upload_noise(warper.get_init_noise_for_gpu())

    proj = glm.perspective(glm.radians(60.0), W / H, 0.1, 100.0)
    view = glm.lookAt(glm.vec3(0, 0, 3), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

    angle = 0.0
    speed = 3.0
    axis = glm.normalize(glm.vec3(1, 1, 0))
    prev_model = glm.mat4(1.0)
    prev_vp = proj * view

    for frame in range(n_frames):
        dt = 1.0 / 60.0
        prev_model_save = glm.rotate(glm.mat4(1.0), angle, axis)
        angle += speed * dt
        model = glm.rotate(glm.mat4(1.0), angle, axis)
        vp = proj * view

        renderer.render_scene(model, vp, prev_model_save, prev_vp)

        # Read motion vectors and run Taichi GWTF
        motion = renderer.read_motion()
        result = warper.step(motion)
        renderer.upload_noise(result)

        prev_vp = vp

        if frame in [0, 9, 19, 29]:
            noise = result
            mean = noise.mean()
            std = noise.std()
            print("  Frame %2d: mean=%+.4f std=%.4f" % (frame, mean, std))

            assert abs(mean) < 0.15, "FAIL: |mean| = %.4f > 0.15" % abs(mean)
            assert 0.7 < std < 1.3, "FAIL: std = %.4f outside [0.7, 1.3]" % std

    print("  Taichi mode: ALL CHECKS PASSED")


# Run tests
run_glsl_test()
run_taichi_test()

pygame.quit()
print("\n=== All end-to-end tests passed ===")
