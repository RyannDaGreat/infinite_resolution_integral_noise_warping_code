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

## 2026-07-04: Session — Star Warp Mode (branch star_warp_mode)

### Plan
- New display mode 7 "Stars": N points advected by the motion field with the
  splat-density death/birth algorithm (per-frame uniform star field — see the
  StarWarp project manifest above this repo for the algorithm's derivation and the
  v1 Jacobian failure history).
- All-GPU: splat (CAS float atomics) → row prefix scans → CDF → per-star update with
  2-level CDF binary-search births. Noise warp passes skipped while in Stars mode.
- Variable N via log slider (10^2..10^6, default 10^4), MAX_STARS = 2^20 preallocated.
- Toggleable AA: linear-light tent splat into rgba16float starTex + sRGB OETF at
  display → per-star luminance exactly invariant under subpixel position.
- SETTINGS_VERSION 3 → 4 (new starCount/starAA defaults; forces settings reset).

### Risks
- CAS float-atomic contention in starSplat at 2048² (same idiom as brownian pass).
- Headless WebGPU may be unavailable (existing test already tolerates).

### Results (same day)
- All modules parse; WGSL compiles clean (no console errors beyond pre-existing
  favicon 404).
- Headless (real WebGPU, 1024²): noise mode unchanged (mean 0.000, std 1.000);
  Stars mode 10k stars: 100% in-bounds, 4×4 grid min/mean 0.949, max/mean 1.062.
- Under 5 s of walking+strafing (deaths/births firing): min/mean 0.891,
  max/mean 1.066 — uniform under motion. PASS.
- N = 1,000,000 stars: 60 fps, GPU total 7.3 ms, min/mean 0.968, max/mean 1.029.
  At 1024² that's ~95% pixel occupancy — visually the thresholded-noise limit.
- Fixed during implementation: blue-noise restore had to be gated on !starsMode —
  in Stars mode no backup is taken that frame, so the unconditional restore would
  have copied a STALE backup over noiseBuf (same bug class as the lock-restore
  bug from Session 2 — restore paths must be gated by every warp-skipping mode).

### GitHub Pages deploy (same day)
- Site appeared stale after pushing Stars mode: the Pages build had been BROKEN
  since Mar 8 — commit 55547ca ("hhhhh") committed web_demo_fluid/node_modules
  as a symlink to ../web_demo/node_modules (not in the repo tree → dangling).
  Legacy Jekyll Pages refuses symlinks: every build since errored ("Page build
  failed"), leaving the live site frozen at the Mar 5 build.
- Fix: git rm the symlink, .gitignore node_modules (369b5bd). Build went green.
- Live validation (headless WebGPU against the deployed URL): noise mean 0.0004
  / std 0.9999; stars 100% in-bounds, min/mean 0.918, max/mean 1.067 under
  motion. PASS.
- Lesson: never commit node_modules symlinks — legacy Pages/Jekyll dies on them
  and the failure is silent unless you check the Pages build API/settings.

## 2026-07-05: Stars mode death rule → star strength (v3)

- starUpdate no longer rolls a survival coin. Each star has a strength q ~ U[0,1)
  (new starStrengthBuf, stride-1 f32, binding 6 on starUpdate only); crowding
  multiplies it (q *= max(E,1) at the new position) and the star dies at q >= 1.
  Statistically identical to the old coin (inverse-CDF of the death time; proof +
  head-to-head simulation in the outer StarWarp manifest/concerns, 2026-07-05).
  Death is now RNG-free; PCG is used only for respawn draws. Render + stats paths
  untouched (positions stay stride-2).
- Do NOT threshold a fixed q per frame instead of eroding: survivors become immune
  to repeat thinning; contraction collapses all stars into a clump (measured).
- Verified headless: noise mode unchanged (mean 0.0004, std 0.9998); stars 10k
  100% in-bounds 0.902/1.051; motion test 0.928/1.080; 1M stars 60 fps 0.980/1.046.
- Fixed test_headless.mjs: `const URL` shadowed the global URL constructor,
  breaking --serve. Renamed PAGE_URL.

## 2026-07-05: Stars-mode background field view (turbo)

- User: "view the fields below the stars (maybe the turbo colormap)". Added
  `field: OFF / E / deficit` button to the Stars settings row: display shader
  (mode 6) composites a dimmed (0.35x) turbo colormap of the star density buffer
  under the stars in linear light before the single sRGB encode. Density view maps
  E/2 (0 uncovered = blue, 1 = mid green, >=2 pile-up = red); deficit view maps
  max(1-E,0). New DisplayUniforms field starField (u32, offset 28 — buffer was
  already 32 B); starDensityBuf bound read-only at display binding 5 (stale in
  non-Stars modes, never read there). SETTINGS_VERSION 4 -> 5 (new starField key).
- Verified headless: test_headless + test_stars_motion still pass; scripted
  click-through OFF->E->deficit->OFF renders with zero console errors; screenshot
  with E field shows uniform mid-turbo green under a static camera (E == 1
  everywhere) as expected.

## 2026-07-05: Emoji identity view + graveyard PORTED to WGSL (with two shipped bugs)

- Port implemented per manifest spec: starMetaBuf {q, id}, starCountersBuf
  [id mint, ghost seq], ghostBuf ring (GHOST_CAP 2^20, capacity uniform min(5N, cap)),
  ghostCellHeadBuf + ghostAdvect pass (atomicMax cursor = free MRU), resurrection via
  CAS claim in starUpdate; emoji atlas (61 glyphs, OffscreenCanvas -> rgba8unorm,
  Knuth hash by id) + premultiplied-alpha render pipeline variant; white tents now
  write alpha = coverage so stars composite over the field background.
- BUG 1 (user-caught, black screen): starUpdate initially bound 10 storage buffers;
  the DEFAULT maxStorageBuffersPerShaderStage is 8 -> invalid pipeline -> every
  command buffer containing it rejected at submit -> nothing rendered at all, and
  the stats staging buffer read back zeros (all "stars" at origin: min/mean 0,
  max/mean 16 = one meter cell). Fix: consolidate strengths+ids -> starMetaBuf and
  idCounter+ghostSeq -> starCountersBuf (exactly 8), plus request the adapter's
  maxStorageBuffersPerShaderStage as belt-and-braces.
- BUG 2: `meta` is a WGSL RESERVED keyword -> shader module failed to parse.
  Renamed starMeta.
- TESTING LESSON (why our headless tests missed both): Dawn/WebGPU validation
  errors surface as console type WARNING in headless Chrome, not 'error'.
  test_headless.mjs now promotes warnings matching
  /invalid|error while|exceeds the maximum|validating/i to failures.
- Verified after fix: headless + motion tests pass (10k: 0.902/1.051; motion
  0.936/1.102; 1M @ 60 fps: 0.972/1.041); full toggle sweep (emoji x graveyard x
  field) zero GPU/JS issues; emoji screenshot confirmed (10k distinct glyphs).

## 2026-07-05: "The cactus never comes back" — user-caught resurrection granularity bug

- User walked to a wall (saw a cactus emoji), backed off just until it died, walked
  forward again: the cactus never returned. Diagnosis required GPU instrumentation
  (counters[2]/[3] = deaths/resurrections; window.__starIds identity sample in the
  stats readback — which exposed ANOTHER bug: starMetaBuf lacked COPY_SRC, so the
  readback copy invalidated every 60th frame's entire command buffer).
- Root cause: resurrection matched ghosts at 1-PIXEL cells. The Python reference ran
  at 64×96 where one cell is 1/6144 of the domain; at 1024² one cell is 1/1048576 —
  ~50k ghosts occupy ~5% of cells, so births almost never landed on a ghost.
  Measured on the new test_graveyard_cycle.mjs (scripted walk-back/walk-forward):
  resurrections = 1.3% of deaths, identity recovery 40.5% ON vs 39.1% OFF (~no
  benefit). LESSON: a granularity parameter validated at prototype resolution
  changes SEMANTICS at production resolution — express match radii as
  resolution-relative, and port tests along with the algorithm.
- Fix: bucket the ghost MRU grid at ghostBucket = max(1, W/128) px (~8 px at 1024²,
  matching the Python test's RELATIVE granularity; ≤1% of screen width displacement,
  where the Python fine-grid tests showed no measurable bias). Result:
  resurrections 24.6% of deaths, identity recovery 64.3% vs 38.8% OFF.
  Remaining gap to Python's 84.7% is expected physics: walking kills ~60% of stars
  per leg, edge deaths are unrecoverable (content truly leaves the frame), and 61k
  cycle deaths overflowed the 50k ring (5×N capacity, the user's tunable knob).
- test_graveyard_cycle.mjs added as a permanent test (asserts resurrections fire and
  ON beats OFF on identity retention).

## 2026-07-05 (evening): graveyard REMOVED; q-color + q-size views added

- User verdict after playing: "it never goes into the same bin" — removed the ghost
  ring, ghostAdvect pass, resurrection, graveyard button, and test_graveyard_cycle.mjs.
  Full implementation at git tag `graveyard-final` (loud commit titles). Kept ids,
  emoji view, deaths counter, id readback. StarUniforms back to 16 B; counters
  shrank to [id mint, deaths]; SETTINGS_VERSION 7.
- New Stars-row toggles, orthogonal and composable: `q-color` (turbo tint by
  strength: blue fresh → red near death) and `q-size` (footprint scaled by
  max(q, 0.15): smaller q = smaller star; also scales emoji sprites; with AA the
  non-integer tent radius gives up exact brightness invariance — diagnostic view).
- Verified: headless + motion tests pass (10k 0.904/1.051; 1M @ 60 fps 0.978/1.043);
  q-color, q-color+q-size, emoji+q-size sweeps render with zero GPU/JS issues
  (screenshots checked). Removal gotcha: my first strip attempt sliced shaders.js
  between 'struct StarUniforms' and the FIRST 'fn pcg(' — but other warp shaders
  define their own pcg earlier, producing an empty slice and a catastrophic
  str.replace(''): always anchor slice endpoints (src.index(needle, start)).

## 2026-07-05: q-size max slider (user: "size is too subtle")

- Replaced the base-radius q scaling (max star ~2 px at 1024² — invisible) with an
  absolute size: full width = starSizeMax slider (0–20 px, default 8) × max(q, 0.15).
  Tent support (effRadius) now equals the star's own half-width in q-size mode.
  StarRenderUniforms + sizeMaxPx (f32[10]); SETTINGS_VERSION 8. Verified: headless
  test passes; 20 px + q-color screenshot checked, zero GPU issues.
