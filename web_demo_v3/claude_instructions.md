# Web Demo V3 — Physics Domino Demo with Noise Warp

## Goal
A first-person physics sandbox where the player walks around a large floor, shoots spheres from a cannon (mouse click), and knocks over domino chains. The noise warp algorithm visualizes temporally coherent noise over the entire physics scene — same algorithm as V2, but with dozens of independently moving rigid bodies instead of a single spinning cube.

## Why This Demo
More fun, more cool, more interesting to play with than the spinning cube. Eventually working toward a game.

## Algorithm

### Noise Warp (identical to V2)
See `../web_demo_v2/claude_instructions.md` for the full mathematical description. In brief:
- Model noise as a continuous white-noise field integrated over pixel areas
- When pixels move (optical flow), use **Brownian bridge** sampling to split each source pixel's noise into fractional contributions to destination pixels
- Normalize by `1/√(pixelArea)` to restore unit variance; fill disoccluded pixels with fresh Gaussian noise
- Optional **blue noise** post-processing (iterative Gaussian high-pass + σ_hp rescaling)

### What's New in V3: Instanced Rendering + Physics

**Instanced rendering**: One storage buffer of instance transforms, one draw call per geometry type (box, sphere). Scales to 50+ objects without per-object bind group swaps.

**Instance data layout** (per instance, 144 bytes):
```
struct InstanceData {
    model:     mat4x4f,   // current frame transform
    prevModel: mat4x4f,   // previous frame transform (for motion vectors)
    color:     vec4f,     // instance color (RGBA)
}
```

**Motion vectors**: computed identically to V2 but per-instance:
```
currNDC = (viewProj * instances[instance_index].model * position).xy / w
prevNDC = (prevViewProj * instances[instance_index].prevModel * position).xy / w
motion = (currNDC - prevNDC) * 0.5
```

**Rapier.js**: Rust/WASM physics engine. Loaded from jsdelivr CDN via dynamic `import()`.

**Raw WebGPU** (no Three.js): Keeps the existing MRT pipeline for motion vectors.

## Interaction Design

### Player
- **First person**: FPS camera with pointer lock, WASD movement, mouse look
- **Physics body**: Dynamic capsule (half-height 0.5, radius 0.3) with locked rotation. Gravity pulls player down, floor collision prevents falling through.
- **Jump**: Spacebar applies upward impulse when grounded. Grounded = raycast downward detects floor within threshold.
- **Eye height**: Camera at body position + 0.7 offset (eyes at ~1.5m)
- **Implementation note**: Dynamic body (not kinematic) so gravity and floor collision work automatically.

### Shooting
- **Trigger**: Mouse click while pointer locked (mousedown on document for pointer lock compatibility)
- **Projectile**: Blue sphere (radius 0.3, density 30) spawned 1.5 units ahead of camera
- **Velocity**: 15 units/sec in camera forward direction
- **Max projectiles**: 20 active. Oldest recycled when limit reached.
- **Spawn offset**: 1.5 units forward to clear player capsule collider

### Dominoes
- **Half-extents**: (0.75, 1.5, 0.18) → full 1.5 × 3.0 × 0.36 (3x scale for dramatic cascades)
- **Thin axis**: Z (fall direction is along Z)
- **Spacing**: 2.5 units along Z (wide gaps for dramatic fall arcs)
- **Arrangement**: 60 dominoes in a straight line along -Z, starting at z=0
- **Material**: friction=0.4, restitution=0.1 (low bounce, realistic topple)
- **Gravity**: -25 m/s² (stronger than Earth for snappy physics feel)

### Floor
- **Size**: 400×0.1×400 units (effectively infinite from player perspective)
- **Physics**: Fixed cuboid collider, friction=0.5
- **Visual**: Gray box instance, top surface at y=0

### Camera Defaults
- **Position**: (0, 1.5, 4) looking toward (0, 0.5, 0) — sees the domino line from slightly above
- **Movement speed**: 5 units/sec

## World Layout

The world is a large 400×400 floor with distinct game areas in each cardinal direction:

| Area | Position | Description |
|------|----------|-------------|
| **Domino Alley** | Center (X≈0, Z=0 to -150) | 60 dominos in a line. Starting area. |
| **The Labyrinth** | West (X≈-100, Z≈0) | Procedural stone maze (15×15). Treasure chest in center. Ivy overgrowth. |
| **The Contraption** | East (X≈+100, Z≈0) | Marble machine theme park. Staircases, ramps, rails, spinning wheels, rope chains. |
| **Sky Steps** | South (X≈0, Z≈-200) | Platforming tower. Randomized cube platforms + ramps. Provably solvable path to flag at top. |

Each area has a wooden **signpost** at its entrance: a cylinder pole with a rectangle sign textured with wood (from `assets/wood_color.jpg`). Signs are labeled with the area name.

### Signposts
- **Geometry**: Cylinder (pole, ~0.15 radius, ~2.5m tall) + box (sign, ~1.5×0.6×0.05)
- **Texture**: Wood color map from `assets/wood_color.jpg` (CC0, ambientCG WoodFloor051)
- **Normal map**: `assets/wood_normal.jpg` for surface detail
- **Text**: Area name rendered as contrasting dark text on the sign face (procedural or texture)
- One signpost at the entrance to each area, facing the player spawn

### Dominoes
- **Visual detail**: Pips have **normal-mapped indentation** (concave bowls via procedural normal perturbation in shader). Center divider line is also indented (V-groove normal perturbation).

## File Structure
```
web_demo_v3/
├── index.html              # Canvas, toolbar (mode bar + settings buttons), CDN imports
├── main.js                 # Entry point, game loop, module wiring, input routing
├── physics.js              # Rapier world, floor, dominoes, player capsule, shooting, maze, marble machine, tower
├── scene.js                # Instance buffer management, FPSCamera, transform sync
├── renderer.js             # WebGPU device, pipelines, instanced MRT, warp, blue noise, display, sky
├── ui.js                   # Settings persistence, buttons, hotkeys, mode bar, stats overlay
├── shaders.js              # All WGSL (instanced scene + sky, display 6 modes, 5 warp passes, blue noise)
├── geometry.js             # Box, beveled box, sphere, quad vertex data (position + normal, 6 floats/vert)
├── assets/
│   ├── wood_color.jpg      # CC0 wood plank texture (ambientCG WoodFloor051, 1K)
│   └── wood_normal.jpg     # CC0 wood normal map (OpenGL convention)
├── claude_instructions.md  # This manifest
└── concerns.md             # Live progress log
```

### Module Responsibilities

**`physics.js`** — Owns Rapier world and all rigid bodies.
- `initPhysics()` → creates world, floor, dominoes, player capsule
- `stepPhysics(dt)` → advances simulation
- `shootSphere(pos, dir)` → spawns projectile
- `resetScene()` → destroys and recreates all bodies
- `getTransforms()` → returns array of {position, rotation} per body
- **Note**: Physics knows nothing about rendering. Clean testing boundary.

**`scene.js`** — Bridges physics ↔ rendering. Manages instance data.
- `SceneManager` class: tracks renderable objects, maps physics bodies to instance slots
- Builds Float32Array of InstanceData each frame from physics transforms
- Stores previous-frame transforms for motion vectors
- `FPSCamera` class (extracted from V2)
- **Note**: Decouples physics body indexing from GPU buffer layout.

**`renderer.js`** — All GPU work. Largest file (~800 lines).
- WebGPU device init, textures, storage buffers
- Instanced scene render pipeline (box + sphere draw calls)
- All 5 warp compute passes (identical to V2)
- Blue noise post-processing (identical to V2)
- Display render pass (identical to V2)
- Profiling (GPU timestamps + stats readback)
- **Note**: Warp pipeline shares 8+ buffers with renderer — kept in same file to avoid artificial coupling.

**`ui.js`** — All UI state management.
- Settings persistence (localStorage with version)
- Button creation and hotkey handling
- Stats overlay formatting
- Modifier key guards (Cmd/Ctrl don't trigger hotkeys)
- **Note**: UI logic was tangled with game logic in V2's main.js. Cleaner to separate.

**`main.js`** — Glue. Wires modules, runs game loop.
- `init()`: physics → scene → renderer → UI
- `frame()`: physics.step → scene.updateTransforms → renderer.frame → UI.updateStats
- Pointer lock management, input routing
- **Note**: Keeps each module focused. Main.js is the only file that imports everything.

## UI Controls
All settings saved to localStorage (`iinw_v3_settings`), restored on page load.

| Button | Hotkey | Description |
|--------|--------|-------------|
| Resolution | P | Cycle 2048→1024→512→256 |
| Blue Noise | B | Toggle blue noise post-processing |
| BN×N | N | Cycle blue noise iterations: 2→5→10 |
| Grey | G | Greyscale display |
| Uniform | U | Gaussian→Uniform via normal CDF |
| Retina | T | Toggle HiDPI scaling |
| Interp | I | Bilinear vs nearest-neighbor |
| Round | O | Motion rounding: None→All→>1 |
| Threshold | H | Toggle threshold visualization |
| Lock Noise | L | Freeze noise warp — snapshot current noise including blue noise |
| Reset Scene | R | Reset physics (re-stand all dominoes) |
| Slow-Mo | M | Toggle 0.25x physics speed |
| Reset Settings | — | Reset all UI to defaults |

### Per-Mode Settings
- **S+N mode (mode 3)**: Noise opacity slider appears (0.0–1.0, default 0.25). Controls how much noise is blended over the scene.
- **Stars mode (mode 7)**: Star count slider (log scale 10²–10⁶, default 10⁴) and AA toggle button appear. AA has no hotkey — letter keys conflict with WASD movement.

### Display Modes (clickable toolbar + keys 1-6)
| Key | Name | Description |
|-----|------|-------------|
| 1 | Noise | Raw warp noise (default) |
| 2 | Scene | Blinn-Phong rendered scene with 2-light shading |
| 3 | S+N | Scene + noise overlay (opacity controlled by slider) |
| 4 | Dither | B&W dithering: threshold scene luminance with noise CDF |
| 5 | Motion | Motion vector visualization (RG = motion × 5 + 0.5) |
| 6 | Raw | Raw noise with threshold and display flags |
| 7 | Stars | Star warp: N points advected by the motion field, per-frame uniform (see Star Warp Mode section) |

### Star Warp Mode (mode 7 / index 6)

The point-process analog of the noise warp: N white point "stars" ride the motion
field, and every frozen frame is indistinguishable from N uniform-random points. No
flow can create persistent over/under-density. (Full algorithm derivation, proofs, and
the v1-Jacobian-failure history live in the StarWarp project manifest one level above
this repo; this section documents the GPU integration.)

Per frame in Stars mode (noise-warp compute passes are SKIPPED — the modes are
mutually exclusive displays, freeing the ~12 ms warp budget):

1. `clearBuffer(starDensityBuf)`
2. **starSplat** (W·H threads): each pixel forward-advects unit mass by its motion
   vector — pixel flow = (+motion.r·W, −motion.g·H) — and bilinearly splats it into
   the density buffer via CAS float atomics. Result E: 1 = unchanged, 0 = uncovered,
   >1 = pile-up. E is the transported density (NOT the Jacobian, which misses
   translations/fold-overs).
3. **starScanRows** (H threads): deficit = max(1 − E, 0); per-row inclusive prefix
   sums (last column = row total).
4. **starScanCdf** (1 thread): prefix over row totals → row CDF; last entry = total.
5. **starUpdate** (N threads): advect star by bilinearly-sampled motion (at its OLD
   position); death = out-of-frame OR strength exhausted. Each star carries a
   **star strength** q ~ U[0,1) (assigned at birth, stored in starStrengthBuf,
   stride-1 f32 alongside the stride-2 position buffer): crowding erodes it,
   q *= max(E, 1) with E sampled at the NEW position, and the star dies when
   q >= 1. Death consumes NO RNG (deterministic given the flow; inverse-CDF
   sampling of the old "survive w.p. 1/max(E,1)" coin — statistics identical,
   proven + simulated in the StarWarp manifest/concerns one level up, 2026-07-05).
   Dead stars respawn via 2-level CDF binary search (row, then column within row)
   + tent jitter, reflected at borders, with fresh q from the PCG RNG. If total
   deficit ≈ 0, respawn uniformly (quota: star count is exactly N every frame).
   Do NOT threshold a fixed q per frame instead of eroding — survivors become
   immune to repeat thinning and contraction collapses all stars into a clump.
6. **starRender**: N·6 vertices (a quad per star) into starTex (rgba16float,
   additive blend); display mode 6 reads starTex over a black background.

**Antialiasing (toggleable, default ON)**: AA renders each star as a tent-kernel
footprint in LINEAR light; the display pass applies the sRGB OETF at the very end.
The unit tent is a partition of unity, so total luminance per star is exactly
invariant under subpixel position — no brightness flicker as stars cross pixels.
(Summing coverage in sRGB space would make a split star look ~half as bright.)
Tent radius = max(1, round(W/1024)) texels, integer to keep the exactness. AA OFF
draws hard weight-1 quads (retro single-pixel look); 0/1 are sRGB fixed points so
both paths share one display shader.

- **Star ids + emoji view (added 2026-07-05); graveyard REMOVED same day**:
  - **Ids**: starMetaBuf interleaves {q: f32, id: u32} per star (seeded
    0..MAX_STARS−1); fresh births mint ids from starCountersBuf[0] (atomic,
    starts at MAX_STARS); counters[1] counts deaths (diagnostics, published with
    window.__starStats; the stats readback also publishes window.__starIds —
    starMetaBuf carries COPY_SRC for this, or the stats frame's submit is
    rejected). WGSL trap: `meta` is a RESERVED keyword — binding is starMeta.
  - **Emoji identity view** (`emoji` button, setting `starEmoji`): 61-glyph atlas
    rasterized at init (OffscreenCanvas 8×8 × 64px → rgba8unorm); starRender picks
    the cell by Knuth hash of id, samples premultiplied; separate pipeline with
    one/one−srcα blending vs the additive white-tent pipeline (two bind groups,
    'auto' layouts). White tents write alpha = coverage so display composites
    stars OVER the optional field background in linear light.
  - **q-color** (`q-color` button, setting `starColorQ`): tints tents by
    turboQ(q) — blue fresh, red near death. Ignored by emoji sprites.
  - **q-size** (`q-size` button + 0–20 px slider, settings `starSizeQ` /
    `starSizeMax`, default 8 px): size = REMAINING LIFE. A star's disc width =
    starSizeMax · (W/1024) · max(1 − q, SIZE_Q_MIN=0.15) texels (slider is
    calibrated at 1024² and scales with resolution, like tent radius/glyphs): fresh stars (q ~ 0) at the
    slider's full width, SHRINKING as crowding erodes q toward death (user:
    "they should get smaller as they die"). Rendered as solid discs with a
    ~1 px antialiased rim — scaling the AA tent kernel instead shows the filter
    itself as a radial-gradient blob (user: "why do they have gradients").
    Composes with emoji AND q-color (orthogonal). History: base-radius scaling
    was "too subtle" → absolute slider; direct q mapping felt backwards →
    inverted.
  - **GRAVEYARD (REMOVED)**: a ghost-ring resurrection system lived here for a few
    hours — full implementation at git tag `graveyard-final`, loud ===== commit
    titles. Removed because in real 3D play births almost never land in the same
    bucket as the ghost they should revive (user verdict after the "cactus test";
    measured story in concerns.md 2026-07-05). Keep 8 storage buffers max per
    stage in starUpdate — the baseline maxStorageBuffersPerShaderStage; exceeding
    it invalidates the pipeline and blacks out the whole frame.
  - StarUniforms back to 16 B {W, H, frameSeed, numStars};
    StarRenderUniforms 44 B (48 B buffer, sizeMaxPx at f32[10]). SETTINGS_VERSION 8.
- **Stereo stars (added 2026-07-05; report §13 has the algorithm + proofs)**:
  `stereo` button cycles OFF / SBS-cross / SBS-parallel / blend / red-blue
  (SETTINGS_VERSION 10). SBS is TRUE 1:1: in the two SBS modes (and Stars
  display only) ui.updateCanvas makes the canvas BACKING STORE 2W×H and the
  display shader maps each half to its eye texture texel-for-texel, full
  height — no downsampling, no CSS stretching, retina-toggle-correct. (Two
  shipped-and-fixed attempts first: squeezed 2:1 SBS, then render-at-half +
  2x-CSS-stretch — "low res and flickery"; the lesson is render at the size
  you display.) Crossed order puts the RIGHT eye's view on the LEFT half for
  cross-eyed fusion; each eye rect gets a 1 px (resolution-scaled) green
  border as a fusion anchor. Display-mode switches re-run updateCanvas (both
  the click and keyboard paths) so leaving Stars restores the 1024 canvas.
  A `swap R/B` toggle flips which eye gets the red vs blue channel in
  anaglyph mode (glasses get worn both ways; setting `stereoSwap`,
  SETTINGS_VERSION 11). `IPD` slider (0–0.5 world
  units, default 0.12) and `FOV` slider (40–120°, applies in mono too — proj is
  rebuilt in main.js when it changes). Two independent star streams (starBufR /
  starMetaBufR, disjoint id ranges, separate birth RNG seed) each advected by
  their eye's temporal motion; per-eye render-time merge = the user's hcat warp
  step with births skipped (provably equivalent — kept-half deficit is 0):
  starSplat REUSED with the cross texture transports the other eye's mass into
  this eye's density, then mergeSelect (2N threads, exactly 8 storage buffers)
  thins own + reprojected-other COPIES by the deterministic q·(1+E) < 1 rule
  into compact per-eye lists drawn via drawIndirect. Four flows: scene pass per
  eye MRT-outputs temporal motion + cross-eye motion (CameraUniforms grew to
  304 B with otherViewProj; mono sets otherViewProj = viewProj so cross ≡ 0;
  sky writes zero disparity). Field background is mono-only. Stats readback
  publishes __starStats.mergedL/R (each ≈ N by the merged-uniformity invariant
  — the test asserts |merged/N − 1| < 6%; also publishes stereoActive).
  Measured (test_stereo.mjs): mergedL/R within 0.3–1.7% of N across all modes,
  static + motion + IPD ∈ {0, 0.5}; mono path bit-identical stats after.
- **Field background (added 2026-07-05)**: `field: OFF / E / deficit` button in the
  Stars settings row — the display shader composites a dimmed (0.35×) turbo colormap
  of the star density buffer UNDER the stars (linear light, then one sRGB encode).
  Density maps E/2 (blue = uncovered, mid green = 1, red = pile-up); deficit maps
  max(1−E, 0). DisplayUniforms.starField u32 (0/1/2, offset 28); starDensityBuf is
  display binding 5 (read-only; stale outside Stars mode but never read there).
  Setting `starField`, SETTINGS_VERSION 5.
- **Variable N**: log-scale slider 10²…10⁶ (decades at 0/25/50/75/100), default 10⁴,
  visible only in Stars mode, persisted. MAX_STARS = 2^20 preallocated (8 MB); N is a
  uniform. All positions are seeded uniform at init, so raising N exposes
  stale-but-uniform stars — spatially valid, artifact-free.
- **Lock [L]** freezes the star field too (star compute skipped, render still runs).
- Buffers: starBuf (MAX_STARS·2 f32), starDensityBuf (N f32, atomic-CAS written),
  starRowPrefixBuf (N f32), starRowCdfBuf (H f32), starUniformBuf, starRenderUniformBuf,
  starTex (rgba16float), _starStagingBuf (stats readback).
- Headless validation: `window.__starStats` = periodic readback of ≤65536 star
  positions → in-bounds fraction + 4×4 coarse-grid min/mean & max/mean.

### Noise Lock Behavior
When lock is engaged:
1. If blue noise is enabled, it's **baked** into the noiseBuf on the transition frame (applied without restore)
2. All subsequent frames skip the entire warp pipeline — noiseBuf is frozen
3. On unlock, the next frame's warp pipeline completely overwrites noiseBuf
This ensures the locked snapshot matches what the user was seeing, including blue noise processing.

### Scene Shading
- **Blinn-Phong** with 2 lights:
  - Light 0: warm directional (1.0, 3.0, 2.0), color (1.0, 0.95, 0.9)
  - Light 1: cool fill (-2.0, 1.0, -1.0), color (0.3, 0.35, 0.5)
- **Ambient**: 0.15 uniform
- **Specular**: shininess=32, intensity 0.3/0.15 per light

### Procedural Textures (in scene fragment shader)
- **Floor**: Voronoi stone pattern — large gray rock cells with dark mortar edges. Scale 0.4, uses world XZ.
- **Dominoes**: Ivory base (0.92, 0.90, 0.85), beveled geometry, two pip sets per face (top/bottom halves like real dominoes, 1-6 dots each), center divider line. Uses **localNormal** for face detection so pips survive rotation.
- **Spheres**: Soccer ball — icosahedron-based 12 pentagon centers, black pentagons on white, seam lines.
- All procedural — no texture files needed. Uses localPos, worldPos, and localNormal passed from vertex shader.

### Player Movement
- **Walk**: 5 units/sec (WASD/arrow keys)
- **Sprint**: 12 units/sec (hold Shift)
- **Jump**: impulse of 6 (Spacebar, grounded only via raycast)
- **Linear damping**: 0.5 (was 5.0 — reduced for better jump feel)

### New controls vs V2
- **R = Reset Scene**, **M = Slow-Mo**, **Click = Shoot**, **Space = Jump**, **Shift = Sprint**, **L = Lock Noise**

## Performance Budget
| Component | Time (2048²) | Notes |
|-----------|--------------|-------|
| Rapier step (100 bodies) | <1ms | WASM, trivial workload |
| Transform upload | <0.1ms | 256 × 144 bytes = 36KB writeBuffer |
| Scene render (instanced) | <1ms | 2 draw calls (boxes + spheres), simple geometry |
| Warp pipeline | ~12ms | Same as V2, dominated by Brownian bridge |
| Blue noise (10 iter) | ~8ms | Same as V2 |
| Display | <0.5ms | Fullscreen quad |
| **Total** | ~14ms (~70fps) | Warp-dominated, same as V2 |

## Risks and Mitigations

### Motion vector quality with many objects
Domino edges produce sharp motion discontinuities (adjacent pixels belong to different objects). The warp handles this via:
- Fresh noise at disoccluded pixels (pixelArea=0)
- Bilinear interpolation smooths transitions at object boundaries
- **Mitigation**: Start with 40 dominoes, profile ticket usage, scale up

### Rapier WASM loading
The @dimforge/rapier3d-compat package embeds WASM as base64 (no separate .wasm file). Loading via jsdelivr ESM CDN may have edge cases.
- **Mitigation**: Test loading early, fall back to script tag if needed

### First-frame motion vectors for new objects
When a sphere is spawned, its first frame has prevModel == model (zero motion). This causes the warp to inject fresh noise for those pixels, which is correct — the sphere wasn't visible before.

## Reuse from V2

### Identical (copied):
- All warp compute shaders (build deformation, clear, backward map, brownian bridge, normalize)
- Blue noise shaders (blur horizontal/vertical)
- Display fragment shader (all modes, threshold, uniform CDF)
- PRNG (PCG hash + Box-Muller)
- Quad vertex data
- Stats readback logic
- Blue noise σ_hp table
- Settings persistence pattern

### Modified:
- Scene vertex/fragment shaders: `instance_index` + storage buffer instead of uniform model matrices
- Renderer: instanced draw calls, instance storage buffer, camera uniform buffer
- Geometry: vertex format changes (no per-vertex color, position+normal only), add sphere, beveled box
- Main loop: physics step + transform sync inserted before render

### New:
- physics.js (Rapier integration, all physics logic)
- scene.js (instance buffer management, camera)
- ui.js (extracted from V2's main.js)
- Sphere geometry generation (CCW winding from outside)
- Beveled box geometry (`beveledBoxVertices(worldBevel, halfExtents)`) — per-axis chamfer compensates for non-uniform instance scaling. 44 triangles: 6 inset faces + 12 edge strips + 8 corner tris.
- Player character (capsule body, jump, sprint, shoot)
- Noise lock (snapshot with blue noise bake on transition)
- Per-mode UI settings (S+N opacity slider)
- localNormal vertex output (untransformed normal for texture mapping on rotated objects)

## Pending Work (Session 3)

### Sky
- **Moon**: Crescent moon opposite the sun. Bright moonlight at night so player can always see. Moon illuminates clouds.
- **Animated clouds**: Clouds move across sky (faster drift than current). Domain-warped FBM.
- **Stars at night**: Already implemented — twinkle, hash-based placement.
- **Day/night cycle**: 5-minute full cycle, starts at solar noon (π/2 offset). DONE.

### Shadows
- **Shadow mapping**: Sun-driven directional shadow map. Toggle button to enable/disable.
- **Soft shadows**: PCF or similar. Not completely black (ambient from sky provides fill).

### World Terrain
- **Massive terrain mesh below floor**: Current 400×400 floor becomes a tall pillar. Add a huge terrain grid mesh underneath/around it (20-50x larger). Near the player, mountains are small; further out, massive mountains. Proper grid mesh distorted by heightmap (not box columns).

### Marble Machine Redesign
- **Proper blueprint**: Current layout is a random jumble. Staircases must connect to platforms, ramps must lead somewhere, clear flow direction (climb up, balls roll down).
- **Deformable rope/chain rendering**: Current chains are stretchy rectangles (cubes on spherical joints). Need proper capsule/cylinder rope segments, or a continuous tube mesh.
- **Chain placement**: Chains currently half-embedded in floor. Must hang properly from platform edges.

### Mountain Terrain (The Wilds)
- **Finer grid**: Current 2.0m spacing → ~0.4-0.5m (5x finer). Makes mountain walkable without constant jumping.
- **May need heightfield collider** for smooth physics (instead of per-column cuboids).
- **MAX_INSTANCES may need increase** to handle more terrain columns.

### Signposts
- **Bigger**: More like billboards than tiny signposts. Large enough to read from a distance.
- **Closer to entrances**: Currently too far away from area entrances.
- **Legible text**: Area names must be readable. Procedural text rendering or texture atlas.
- **All areas**: Every area needs a signpost (currently only The Wilds has one).

### Maze Improvements
- **Seamless stone textures**: Current wall texture looks unrealistic and has visible seams.
- **Ivy geometry**: Actual procedural leaf geometry, not just shader overlay. Vines cascading from wall tops.
- **Treasure chest**: Replace box with a proper 3D chest model (find asset online or procedural).

### Floor
- **Voronoi normal map**: Crevices between stones should be indented (normal perturbation making mortar lines lower than rock surfaces).

### Nighttime Lighting
- **Flashlight**: Player has a cone of light (spotlight) that illuminates nearby objects at night. Forward-facing from camera direction.
- **Glowing soccer balls**: Spheres emit light at night, illuminating surroundings. Blue glow matching their color.
- **Glowing mushrooms**: Mushrooms emit colored light at night (bioluminescence). Different colors per mushroom.
- **User requirement**: "When it's nighttime, give me a flashlight... soccer balls glow... mushrooms should also emit colored light"

### Time of Day Controls
- **4 preset buttons**: Midday, Sunset, Night, Dawn. Each sets the day/night cycle to a specific time.
- **Icons**: Use Iconify icons for each button (sun, sunset, moon, sunrise).
- **Sync**: When user presses a button, the cycle jumps to that time and continues from there.
- **Default**: Cycle starts at midday and auto-progresses.
- **User requirement**: "set to midday, set to night, set to sunset, set to dawn buttons. Iconify icons."

### Sky Warp Noise
- **Toggle button (default ON)**: Sky motion vectors should participate in noise warping. When the camera rotates, the sky "moves" in screen space — those motion vectors should warp the noise too.
- **Implementation**: Sky fragment shader currently outputs `motion = vec4(0,0,0,1)`. To enable sky warping, compute motion vectors for the sky by projecting the same world direction through both current and previous viewProj matrices. Need prevViewProj in sky uniforms.
- **User requirement**: "The sky has to warp the noise too when I look around... by default it should be turned on."

### Architecture Note
- **All visual enhancements (sky, shadows, lighting, textures) are SEPARATE from the noise warp pipeline**. The warp code is its own thing. All this is "fluff on top."
- **User requirement**: "Make sure all these details are separate from the noise rendering code."

### Other
- **Soccer ball texture**: Proper truncated icosahedron (pentagons + hexagons).
- **Floor voronoi normal map**: Crevices/mortar lower than rock surfaces.

## Testing Strategy

### Automated (Puppeteer headless)
1. **Noise stats**: mean≈0, std≈1 (same validation as V2)
2. **Physics loads**: Rapier initializes without error
3. **Instance count**: >0 instances rendered (dominoes exist)
4. **Screenshot**: capture noise view for visual validation
5. **Star warp** (in test_headless.mjs): switch to Stars mode, validate
   window.__starStats — 100% in-bounds, 4×4 grid min/mean > 0.8, max/mean < 1.2
6. **Star warp under motion** (test_stars_motion.mjs): walk/strafe for 5 s in
   Stars mode (exercises deaths/births), re-validate uniformity; AA on/off
   screenshots; N = 10⁶ stress (must hold 60 fps and stay uniform)

### Manual
1. WASD movement + mouse look works
2. Click shoots spheres
3. Spheres knock over dominoes
4. Domino chain cascades
5. Space jumps
6. Display modes (1-6) switch correctly; mode bar highlights active
7. All UI buttons work
8. R resets scene

## Lessons Learned (from V2, preserved)
- Float atomics via CAS loop on u32 reinterpretation (WGSL has no native f32 atomics)
- `pass` is a WGSL reserved keyword
- Canvas `border` causes pixel misalignment — use `outline` instead
- Cmd+R triggers KeyR before page unload — guard all hotkeys with modifier check
- Settings versioning via SETTINGS_VERSION forces reset when defaults change
- Blue noise MUST be post-processing only — feeding it back corrupts the warp
- Sorting → σ_hp rescaling for Gaussian inputs eliminates 4 shaders
- σ_hp varies per iteration (0.923 → 0.992 as spectrum goes white → blue)
