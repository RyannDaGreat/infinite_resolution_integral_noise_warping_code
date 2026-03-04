"""Full test suite: motion vectors + GLSL + Taichi end-to-end."""

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

identity = glm.mat4(1.0)


def cpu_motion(point, model, prev_model, vp):
    p = glm.vec4(*point, 1.0)
    cc = vp * model * p
    pc = vp * prev_model * p
    cn = glm.vec2(cc.x / cc.w, cc.y / cc.w)
    pn = glm.vec2(pc.x / pc.w, pc.y / pc.w)
    d = (cn - pn) * 0.5
    return float(d.x), float(d.y)


def center_motion(mv, W, H):
    cy, cx = H // 2, W // 2
    patch = mv[cy - 3:cy + 3, cx - 3:cx + 3]
    mask = np.any(np.abs(patch) > 1e-6, axis=-1)
    if mask.sum() < 3:
        return None
    return patch[mask].mean(axis=0)


# === Motion vector tests ===
print("=== Motion Vector Tests ===\n")
renderer = Renderer(W, H)
proj = glm.perspective(glm.radians(60.0), W / H, 0.1, 100.0)
view = glm.lookAt(glm.vec3(0, 0, 3), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
vp = proj * view

tests = [
    ("No motion", identity, identity, "zero"),
    ("+X", glm.translate(identity, glm.vec3(0.2, 0, 0)), identity, "+x"),
    ("+Y", glm.translate(identity, glm.vec3(0, 0.2, 0)), identity, "+y"),
    ("-X", glm.translate(identity, glm.vec3(-0.2, 0, 0)), identity, "-x"),
    ("-Y", glm.translate(identity, glm.vec3(0, -0.2, 0)), identity, "-y"),
]

for name, curr, prev, check in tests:
    renderer.render_scene(curr, vp, prev, vp)
    mv = renderer.read_motion()
    g = center_motion(mv, W, H)
    if check == "zero":
        assert np.abs(mv).max() < 1e-5, "FAIL"
        print("  %-10s PASS (max |mv| < 1e-5)" % name)
    elif check == "+x":
        assert g[0] > 0 and abs(g[1]) < 0.005, "FAIL"
        print("  %-10s PASS (mv_x=%+.6f)" % (name, g[0]))
    elif check == "+y":
        assert g[1] > 0 and abs(g[0]) < 0.005, "FAIL"
        print("  %-10s PASS (mv_y=%+.6f)" % (name, g[1]))
    elif check == "-x":
        assert g[0] < 0, "FAIL"
        print("  %-10s PASS (mv_x=%+.6f)" % (name, g[0]))
    elif check == "-y":
        assert g[1] < 0, "FAIL"
        print("  %-10s PASS (mv_y=%+.6f)" % (name, g[1]))

# === GLSL end-to-end ===
print("\n=== GLSL Mode (30 frames) ===\n")
renderer2 = Renderer(W, H)
view2 = glm.lookAt(glm.vec3(0, 1, 4), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
vp2 = proj * view2
angle = 0.0
axis = glm.normalize(glm.vec3(1, 1, 0))
prev_vp2 = vp2

for frame in range(30):
    dt = 1.0 / 60.0
    prev_m = glm.rotate(identity, angle, axis)
    angle += 1.5 * dt
    curr_m = glm.rotate(identity, angle, axis)
    renderer2.render_scene(curr_m, vp2, prev_m, prev_vp2)
    renderer2.warp_noise()
    w = renderer2.read_noise().transpose(2, 0, 1)
    rg, _ = regaussianize(w)
    renderer2.upload_noise(rg.transpose(1, 2, 0))
    prev_vp2 = vp2
    if frame in [0, 29]:
        m, s = rg.mean(), rg.std()
        u = len(np.unique(rg[0].ravel()))
        print("  Frame %2d: mean=%+.4f std=%.4f unique_ch0=%d/%d" % (frame, m, s, u, W * H))
        assert abs(m) < 0.15 and 0.8 < s < 1.2, "FAIL"

print("  GLSL PASSED")

# === Taichi end-to-end ===
print("\n=== Taichi Mode (30 frames) ===\n")
from taichi_warp import TaichiWarper

renderer3 = Renderer(W, H)
tw = TaichiWarper(W, H, channels=4)
renderer3.upload_noise(tw.get_init_noise_for_gpu())
angle = 0.0
prev_vp3 = vp2

for frame in range(30):
    dt = 1.0 / 60.0
    prev_m = glm.rotate(identity, angle, axis)
    angle += 1.5 * dt
    curr_m = glm.rotate(identity, angle, axis)
    renderer3.render_scene(curr_m, vp2, prev_m, prev_vp3)
    motion = renderer3.read_motion()
    result = tw.step(motion)
    renderer3.upload_noise(result)
    prev_vp3 = vp2
    if frame in [0, 29]:
        m, s = result.mean(), result.std()
        print("  Frame %2d: mean=%+.4f std=%.4f" % (frame, m, s))
        assert abs(m) < 0.15 and 0.7 < s < 1.3, "FAIL"

print("  Taichi PASSED")

pygame.quit()
print("\n=== ALL TESTS PASSED ===")
