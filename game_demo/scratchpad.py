# Validate noise functions produce reasonable terrain

import math

FOREST_CENTER_X = 0
FOREST_CENTER_Z = 200
FOREST_HALF_EXTENT = 75
FOREST_AMPLITUDE = 40
FOREST_FREQ = 0.025
FOREST_SEED = 31337
FOREST_GRID_SPACING = 2.0

def hashCell(a, b, seed):
    # Wang hash, matching the JS int32 math
    a = int(a)
    b = int(b)
    seed = int(seed)
    # Use Python's arbitrary precision but mask to 32 bits like JS
    s = (seed ^ (a * 1619 + b * 31337)) & 0xFFFFFFFF
    s = (s ^ (s >> 16)) * 0x45d9f3b & 0xFFFFFFFF
    s = (s ^ (s >> 16)) * 0x45d9f3b & 0xFFFFFFFF
    s = (s ^ (s >> 16)) & 0xFFFFFFFF
    return s / 0xFFFFFFFF

def valueNoise2D(x, z, seed):
    ix, iz = math.floor(x), math.floor(z)
    fx, fz = x - ix, z - iz
    ux = fx * fx * (3 - 2 * fx)
    uz = fz * fz * (3 - 2 * fz)
    v00 = hashCell(ix,     iz,     seed)
    v10 = hashCell(ix + 1, iz,     seed)
    v01 = hashCell(ix,     iz + 1, seed)
    v11 = hashCell(ix + 1, iz + 1, seed)
    return v00 + ux * (v10 - v00) + uz * (v01 - v00) + ux * uz * (v00 - v10 - v01 + v11)

def fbm2D(x, z, seed, octaves=6, lacunarity=2.0, gain=0.5):
    value = 0
    amplitude = 1.0
    totalAmp = 0
    freq = 1.0
    for i in range(octaves):
        value += amplitude * valueNoise2D(x * freq, z * freq, seed + i * 7919)
        totalAmp += amplitude
        amplitude *= gain
        freq *= lacunarity
    return value / totalAmp

def terrainHeight(wx, wz):
    lx = wx - FOREST_CENTER_X
    lz = wz - FOREST_CENTER_Z
    raw = fbm2D(lx * FOREST_FREQ, lz * FOREST_FREQ, FOREST_SEED)
    shaped = raw ** 1.4
    h = shaped * FOREST_AMPLITUDE
    edgeFadeX = max(0, 1 - abs(lx) / (FOREST_HALF_EXTENT * 0.85))
    edgeFadeZ = max(0, 1 - max(0, -lz) / 40)
    fade = min(edgeFadeX, edgeFadeZ) ** 1.5
    return h * fade

# Test some heights
print("Terrain height samples:")
test_points = [
    (0, 200, "center"),
    (0, 150, "south edge"),
    (0, 250, "north"),
    (50, 200, "east"),
    (-50, 200, "west"),
    (75, 200, "far east (edge)"),
    (0, 125, "just inside south boundary"),
    (0, 120, "outside boundary (should fade)"),
]
heights = []
for wx, wz, label in test_points:
    h = terrainHeight(wx, wz)
    heights.append(h)
    print(f"  ({wx:+4d}, {wz:+4d}) [{label}]: h={h:.2f}")

print(f"\nHeight range: {min(heights):.2f} - {max(heights):.2f}")
print(f"Expected max: {FOREST_AMPLITUDE}")

# Count how many terrain cells would be placed (h > 0.1)
step = FOREST_GRID_SPACING
half = FOREST_HALF_EXTENT
count_total = 0
count_active = 0
heights_grid = []
for ix_i in range(int(-half/step), int(half/step)+1):
    for iz_i in range(int(-half/step), int(half/step)+1):
        ix = ix_i * step
        iz = iz_i * step
        wx = FOREST_CENTER_X + ix
        wz = FOREST_CENTER_Z + iz
        h = terrainHeight(wx, wz)
        count_total += 1
        if h >= 0.1:
            count_active += 1
            heights_grid.append(h)

print(f"\nTerrain grid: {count_total} total, {count_active} active (h >= 0.1)")
print(f"Active height range: {min(heights_grid):.2f} - {max(heights_grid):.2f}")
print(f"Total instances from terrain: {count_active}")
print(f"Within 8192 instance budget: {count_active < 7000}")

# Check where terrain starts appearing (south boundary fade)
print("\nSouth edge fade-in (at X=0):")
for wz in range(120, 175, 5):
    h = terrainHeight(0, wz)
    print(f"  Z={wz}: h={h:.3f}")
