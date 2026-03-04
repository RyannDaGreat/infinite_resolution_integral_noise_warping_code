# Concerns / Progress Log

## 2026-03-04

### Reading source files
- Read all 6 Python shaders (scene.vert/frag, noise_warp.vert/frag, display.vert/frag)
- Read renderer.py, geometry.py, main.py, regaussianize.py, taichi_warp.py
- Read inf_int_noise_warp.py (the core taichi kernel)
- Decision: port taichi 4-phase particle warp to JS, skip noise_warp shader + regaussianize
- Only need 4 shaders (scene pair + display pair), not 6

### Starting implementation
- Creating: index.html, shaders.js, geometry.js, particle_warp.js, renderer.js, main.js

### All files created and initial headless test
- All 6 JS files + index.html written
- gl-matrix loaded from CDN, had to destructure `window.glMatrix` for `mat4`, `vec3`, `glMatrix`
- Puppeteer headless test PASSES: FPS=42, mean=0.002, std=1.001
- All 5 display modes render correctly (verified via screenshots)
- Scene: cube with colored faces + gray floor + directional lighting ✓
- Motion vectors: visible on cube, static on floor ✓
- Noise stats: N(0,1) preserved ✓
- Only non-issue: 404 for favicon.ico (harmless)

### Lessons learned
- WebGL 2 headless in Puppeteer requires `--use-gl=angle` (SwiftShader doesn't support WebGL 2)
- gl-matrix 3.4.3 CDN exports to `window.glMatrix`, not top-level globals
- readPixels on canvas after swapBuffers returns [0,0,0,0] without `preserveDrawingBuffer: true`
- MRT with float textures requires scene.frag to output vec2 for motion but the texture must be RGBA32F for readPixels compatibility
