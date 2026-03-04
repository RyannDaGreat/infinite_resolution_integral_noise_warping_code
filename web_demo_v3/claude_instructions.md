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

## File Structure
```
web_demo_v3/
├── index.html              # Canvas, toolbar (mode bar + settings buttons), CDN imports
├── main.js                 # Entry point, game loop, module wiring, input routing
├── physics.js              # Rapier world, floor, dominoes, player capsule, shooting
├── scene.js                # Instance buffer management, FPSCamera, transform sync
├── renderer.js             # WebGPU device, pipelines, instanced MRT, warp, blue noise, display
├── ui.js                   # Settings persistence, buttons, hotkeys, mode bar, stats overlay
├── shaders.js              # All WGSL (instanced scene, display 6 modes, 5 warp passes, blue noise)
├── geometry.js             # Box, sphere, quad vertex data (position + normal, 6 floats/vert)
├── claude_instructions.md  # This manifest
└── concerns.md             # Live progress log
```

### Future Folder Hierarchy (when complexity warrants it)
```
web_demo_v3/
├── game/                   # Physics engine, scene, geometry
│   ├── physics.js
│   ├── scene.js
│   └── geometry.js
├── warp/                   # Noise warping pipeline
│   ├── renderer.js         # (or split: warp.js + render.js)
│   └── shaders.js
├── fancy/                  # Procedural textures, bevels, soccer ball (isolated)
│   └── fancy.js
├── ui.js
├── main.js
└── index.html
```
Files move into subfolders once their concern has >1 file. Currently all are single-file, so flat is fine.

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
| Reset Scene | R | Reset physics (re-stand all dominoes) |
| Slow-Mo | M | Toggle 0.25x physics speed |
| Reset Settings | — | Reset all UI to defaults |

### Display Modes (clickable toolbar + keys 1-6)
| Key | Name | Description |
|-----|------|-------------|
| 1 | Noise | Raw warp noise (default) |
| 2 | Scene | Blinn-Phong rendered scene with 2-light shading |
| 3 | S+N | Scene + Noise/4 overlay (noise attenuated to not overpower) |
| 4 | Dither | B&W dithering: threshold scene luminance with noise CDF |
| 5 | Motion | Motion vector visualization (RG = motion × 5 + 0.5) |
| 6 | Raw | Raw noise with threshold and display flags |

### Scene Shading
- **Blinn-Phong** with 2 lights:
  - Light 0: warm directional (1.0, 3.0, 2.0), color (1.0, 0.95, 0.9)
  - Light 1: cool fill (-2.0, 1.0, -1.0), color (0.3, 0.35, 0.5)
- **Ambient**: 0.15 uniform
- **Specular**: shininess=32, intensity 0.3/0.15 per light

### Procedural Textures (in scene fragment shader)
- **Floor**: Voronoi stone pattern — large gray rock cells with dark mortar edges. Scale 0.4, uses world XZ.
- **Dominoes**: Bevel (edge darkening via distance-from-edge), per-instance color variation (seeded by instance ID), dot pattern on Z-face (1-6 dots like real dominoes).
- **Spheres**: Soccer ball — icosahedron-based 12 pentagon centers, black pentagons on white, seam lines.
- All procedural — no texture files needed. Uses localPos and worldPos passed from vertex shader.

### New controls vs V2
- **R = Reset Scene**, **M = Slow-Mo**, **Click = Shoot**, **Space = Jump**

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
- Geometry: vertex format changes (no per-vertex color, position+normal only), add sphere
- Main loop: physics step + transform sync inserted before render

### New:
- physics.js (Rapier integration, all physics logic)
- scene.js (instance buffer management, camera)
- ui.js (extracted from V2's main.js)
- Sphere geometry generation
- Player character (capsule body, jump, shoot)

## Testing Strategy

### Automated (Puppeteer headless)
1. **Noise stats**: mean≈0, std≈1 (same validation as V2)
2. **Physics loads**: Rapier initializes without error
3. **Instance count**: >0 instances rendered (dominoes exist)
4. **Screenshot**: capture noise view for visual validation

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
