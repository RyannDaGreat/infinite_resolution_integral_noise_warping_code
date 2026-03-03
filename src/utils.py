import os
import taichi as ti
import numpy as np
import math

eps = 1.e-6

@ti.func
def get_randn_like(noise):
    return ti.Vector([ti.randn() for _ in range(noise.n)])

@ti.kernel
def fill_noise(result: ti.template()):
    for I in ti.grouped(result):
        result[I] = ti.Vector([ti.randn() for _ in range(result.n)])

@ti.func
def unravel_index(index, n, m): # n for num rows, m for num cols
    s = index // m
    t = index % m
    return s, t

@ti.func
def ravel_index(s, t, n, m):
    return s * m + t

# brownian bridge B(t) = W(t) - tW(1) + t * x
# sample B(t2) | B(t1) = q
# When t1 ≈ 1, the bridge is at its endpoint and B(1) = x deterministically.
# Guard (1-t1) > eps to avoid division by zero in f32.
@ti.func
def sample_brownian_bridge(x, t1, t2, q):
    sample = x # B(1) = x (endpoint condition)
    denom = 1. - t1
    if t2 < 1. and denom > eps:
        mu = (1.-t2)/denom * q + (t2-t1)/denom * x
        var = (t2-t1) * (1-t2) / denom
        z = get_randn_like(x)
        sample = z * ti.math.sqrt(var) + mu
    return sample
