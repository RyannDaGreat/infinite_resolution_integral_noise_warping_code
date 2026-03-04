# Concerns — Web Demo V3

## 2026-03-04: Initial Build

### Plan
- Physics engine: Rapier.js (WASM, CDN-loaded via @dimforge/rapier3d-compat)
- Rendering: Raw WebGPU, instanced MRT (reuse V2 warp pipeline)
- UI: Vanilla JS with modular file structure (physics.js, scene.js, renderer.js, ui.js, main.js)
- Player: FPS camera on dynamic capsule body, WASD + jump + shoot spheres
- Scene: 40 dominoes on large floor, player shoots spheres to knock them over

### Risks
- Rapier CDN loading: @dimforge/rapier3d-compat via jsdelivr ESM — may need fallback to script tag
- Instance storage buffer alignment: InstanceData must be 16-byte aligned for WGSL
- Motion vector discontinuities at domino boundaries — warp handles via fresh noise injection
- Sphere spawning: first frame has zero motion (prevModel == model) — correct behavior

### Build Order
1. Manifest + concerns (this file)
2. geometry.js (box + sphere + quad)
3. shaders.js (instanced scene + V2 warp/display shaders)
4. physics.js (Rapier world, dominoes, player, shooting)
5. renderer.js (WebGPU, instanced rendering, warp pipeline)
6. scene.js (instance buffer, camera, transform sync)
7. ui.js (settings, buttons, hotkeys)
8. main.js (game loop glue)
9. index.html (canvas, toolbar, Rapier import)
10. test_headless.mjs (Puppeteer validation)
11. Manual testing + fixes

## 2026-03-04: Session 2 — Physics Fixes + Visual Improvements

### Completed
- **Sprint mechanic**: Shift key = 12 units/sec (SPRINT_SPEED). Check both ShiftLeft/ShiftRight in movePlayer().
- **Jump impulse halved**: 12→6 for more realistic feel.
- **Player linear damping reduced**: 5.0→0.5 — was killing jump arc.
- **Sphere damping added**: linear=0.3, angular=0.3 — balls no longer roll forever.
- **Sphere winding fixed**: Triangles were CW from outside → swapped to CCW. Was causing normals to appear inverted (backward spin illusion).
- **Beveled box geometry**: `beveledBoxVertices(worldBevel, halfExtents)` — per-axis chamfer = worldBevel/(2*halfExtent[axis]). 44 triangles (6 inset faces + 12 edge strips + 8 corner tris). Renderer passes domino half-extents [0.75, 1.5, 0.18].
- **Split draw calls**: Floor uses flat boxVB (instance 0), dominoes use bevelBoxVB (instances 1+).
- **Domino texture rewrite**: Uses `localNormal` (untransformed) for face detection — pips survive rotation. Two pip sets per face (top/bottom halves like real dominoes). Removed random per-instance color — all standard ivory (0.92, 0.90, 0.85). Center divider line.
- **Noise lock button [L]**: Freezes warp pipeline. Blue noise baked on transition frame.
- **S+N opacity slider**: Per-mode settings — slider appears only in mode 3 (S+N). Default 0.25.
- **Lock snapshot fix**: On lock transition, blue noise is baked into noiseBuf (applied without restore). Restore guarded with `!noiseLocked` to prevent stale backup from overwriting snapshot.

### Bugs Found and Fixed
- **Pips disappear on rotated dominoes**: Shader used world-space normal for face detection. When dominoes rotate, the world normal changes. Fixed by adding `localNormal` to vertex shader output (location 7).
- **Sphere normals inverted**: Triangle winding was CW from outside. Cross-product confirmed inward-pointing normals. Fixed by swapping vertex order in both triangles per quad.
- **Jump terminal velocity too fast**: Linear damping=5.0 aggressively dampened Y velocity mid-jump. Reduced to 0.5.
- **Spheres roll forever**: No linear/angular damping on sphere rigid bodies. Added setLinearDamping(0.3).setAngularDamping(0.3).
- **Non-uniform bevel**: Unit box bevel=0.04 gets stretched by instance transform. Fixed by generating bevel at final dimensions (worldBevel/halfExtent per axis).
- **Blue noise not locked**: Lock skipped blue noise application AND the restore ran unconditionally outside the lock check, overwriting locked noise with stale backup. Fixed with justLocked transition detection + guarded restore.

### In Progress (Background Agents)
- **Soccer ball texture**: Agent researching proper truncated icosahedron pattern (pentagons + hexagons from icosahedron vertices)
- **Rounded bevel**: Agent implementing proper multi-segment rounded bevel (wwwtyro algorithm) with V-groove support
- **Procedural sky**: Agent implementing Rayleigh+Mie atmospheric scattering for sunset sky

### Pending
- Procedural clouds (Perlin-Worley 3D, not cheap Perlin)
- Dithering options: Floyd-Steinberg, Bayer matrix (N=2,4,8,16), RGB mode
- Center V-groove on domino geometry
- Floor reflectivity (specular)
- Voronoi floor center dots fix

### Design Decisions
- **User requirement**: "Just make fucking dominoes" — no random colors, standard ivory dominos with proper pips
- **User requirement**: Bevel is geometry, not shader — beveled mesh with smooth normals for specular highlights
- **User requirement**: Lock should snapshot current display including blue noise — hence bake-on-transition design
- **User requirement**: Per-mode settings (opacity slider only visible in S+N mode)
- **User requirement**: Sprint mechanic (Shift to run faster)
- **Bevel approach**: "Generate at final dimensions" per bevel research. Avoids non-uniform scale issues. Blender's bevel modifier has the same problem — their answer is "Apply Scale" which is equivalent.
