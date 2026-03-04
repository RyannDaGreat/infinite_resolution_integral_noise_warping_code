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
