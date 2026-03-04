# WebGPU Physics + Noise Warp Architecture Guide

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GAME LOOP (JavaScript)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────┐       ┌──────────────────┐                           │
│  │  Camera Update   │       │ Physics Step     │                           │
│  │  (WASD/EQ)       │───┐   │ (gravity,        │                           │
│  │  Move position   │   │   │  collision,      │                           │
│  │  Compute matrices│   │   │  constraints)    │                           │
│  └──────────────────┘   │   │  → body.position │                           │
│                         │   │  → body.rotation │                           │
│                         │   └──────────────────┘                           │
│                         │           │                                       │
│                         ▼           ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │   Build Instance Data (CPU-side transform matrix)        │              │
│  │                                                           │              │
│  │   for each physics body {                                │              │
│  │     - Compute: model = mat4(rotation, position)          │              │
│  │     - Store in buffer: [model][prevModel][color]         │              │
│  │     - Copy current → previous for next frame             │              │
│  │   }                                                       │              │
│  └──────────────────────────────────────────────────────────┘              │
│                         │                                                   │
│                         ▼                                                   │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │ GPU: Mapped Buffer Upload                                │              │
│  │ (copyBufferToBuffer)                                     │              │
│  └──────────────────────────────────────────────────────────┘              │
│                         │                                                   │
└─────────────────────────┼───────────────────────────────────────────────────┘
                          │
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GPU RENDER PASS (WebGPU)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │ Vertex Shader (Instanced)                                │              │
│  │                                                           │              │
│  │ @builtin(instance_index) idx                             │              │
│  │ inst = instances[idx]  // Storage buffer lookup          │              │
│  │ worldPos = inst.model * in.position                      │              │
│  │ prevWorldPos = inst.prevModel * in.position             │              │
│  │ out.currClip = viewProj * worldPos                       │              │
│  │ out.prevClip = prevViewProj * prevWorldPos              │              │
│  └──────────────────────────────────────────────────────────┘              │
│                         │                                                   │
│                         ▼                                                   │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │ Fragment Shader (MRT: Color + Motion)                    │              │
│  │                                                           │              │
│  │ // Compute motion vectors relative to camera             │              │
│  │ currNDC = currClip.xy / currClip.w                       │              │
│  │ prevNDC = prevClip.xy / prevClip.w                       │              │
│  │ motion = (currNDC - prevNDC) * 0.5                       │              │
│  │                                                           │              │
│  │ color = computePhongShading(normal, position)            │              │
│  │                                                           │              │
│  │ Output to:                                               │              │
│  │  - colorTarget  (RGBA8)  ← for display                   │              │
│  │  - motionTarget (RG32f)  ← for noise warp                │              │
│  └──────────────────────────────────────────────────────────┘              │
│                         │                                                   │
│                    Color    Motion                                          │
│                    (RGBA)   (RG32f)                                         │
│                      │        │                                             │
│                      ▼        ▼                                             │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │ Compute Shader: Deformation Field (from motion)          │              │
│  │ (Your existing noise warp pipeline)                      │              │
│  │                                                           │              │
│  │ - Read motion vectors                                    │              │
│  │ - Build deformation gradient                             │              │
│  │ - Compute backward map                                   │              │
│  │ - Apply noise warping                                    │              │
│  └──────────────────────────────────────────────────────────┘              │
│                         │                                                   │
│                         ▼                                                   │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │ Display Pass (fullscreen quad)                           │              │
│  │                                                           │              │
│  │ Sample warped color                                      │              │
│  │ Apply post-effects (blue noise, thresholding, etc)       │              │
│  │ Output to canvas                                         │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Motion Vector Computation

### What Happens When Everything Moves

```
Frame N-1:
┌──────────────────────┐
│ Camera Position: P_c │
│ Body Position: P_b   │
│ renders at screen (x₁, y₁)
└──────────────────────┘

              ⬇

Frame N:
┌──────────────────────┐
│ Camera Position: P_c' │ (moved forward)
│ Body Position: P_b'   │ (moved right)
│ renders at screen (x₂, y₂)
└──────────────────────┘

              ⬇

Motion Vector Calculation:
currNDC = (x₂/w₂, y₂/w₂)         // current normalized device coords
prevNDC = (x₁/w₁, y₁/w₁)         // previous normalized device coords
motion = (currNDC - prevNDC)       // relative displacement

              ⬇

Result:
- If camera moved forward and body moved right:
  motion = (body_right_motion - camera_forward_motion)
  = apparent motion relative to camera

- If body static and camera moved:
  motion = -camera_motion (everything appears to shift opposite to camera)

- If body and camera moved same direction/speed:
  motion ≈ 0 (no relative motion)
```

---

## Storage Buffer Memory Layout

### Per-Instance Data Structure

```
Struct InstanceData (144 bytes = 36 floats):

  Offset (bytes)   Offset (floats)   Size      Field
  ──────────────   ───────────────   ────      ─────
  0                0                 64        modelMatrix (4×4 mat4x4f)
  64               16                64        prevModelMatrix (4×4 mat4x4f)
  128              32                16        color (vec4f)
  ──────────────   ───────────────   ────
  144              36                          TOTAL


Storage Buffer Layout (100 bodies):

  Index 0: [model₀ | prevModel₀ | color₀]
  Index 1: [model₁ | prevModel₁ | color₁]
  ...
  Index 99: [model₉₉ | prevModel₉₉ | color₉₉]

Total size: 100 × 144 = 14,400 bytes = ~14 KB
```

### WGSL Struct Definition

```wgsl
struct InstanceData {
    model:      mat4x4f,    // Offset: 0
    prevModel:  mat4x4f,    // Offset: 64
    color:      vec4f,      // Offset: 128
}

@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;
```

### Vertex Shader Access

```wgsl
@vertex fn vs(
    in: VsIn,
    @builtin(instance_index) idx: u32,
) -> VsOut {
    let inst = instances[idx];  // GPU automatically does offset calculation

    // inst.model points to bytes [idx*144 : idx*144+64]
    // inst.prevModel points to bytes [idx*144+64 : idx*144+128]
    // inst.color points to bytes [idx*144+128 : idx*144+144]

    let worldPos = inst.model * vec4f(in.position, 1.0);
    let prevWorldPos = inst.prevModel * vec4f(in.position, 1.0);

    return out;
}
```

---

## Camera Matrix Tracking (Critical for Motion Vectors)

### Frame-by-Frame Evolution

```
BEGIN Frame N-1:
┌─────────────────────────────────────────────────┐
│ Camera:                                         │
│  viewMatrix_N-1 (from position, forward, up)    │
│  projMatrix (unchanged each frame)              │
│  viewProj_N-1 = proj × view                     │
│                                                  │
│ Render:                                         │
│  Vertex: clip = viewProj_N-1 × world_pos        │
│  Fragment: motion = 0 (prev = curr in first)    │
└─────────────────────────────────────────────────┘

                    ⬇

         Save for next frame:
         prevViewProj_saved = viewProj_N-1

                    ⬇

BEGIN Frame N:
┌─────────────────────────────────────────────────┐
│ Camera Movement:                                │
│  processMouse(dx, dy)  → update yaw/pitch       │
│  processKeys(dt)       → update position        │
│  computeViewMatrix()   → new viewMatrix_N       │
│  projMatrix * viewMatrix_N → viewProj_N         │
│                                                  │
│ Render:                                         │
│  Vertex:                                        │
│    currClip = viewProj_N × world_pos_N          │
│    prevClip = prevViewProj_saved × world_pos_N-1│
│                                                  │
│  Fragment:                                      │
│    currNDC = currClip.xy / currClip.w           │
│    prevNDC = prevClip.xy / prevClip.w           │
│    motion = (currNDC - prevNDC)                 │
│                                                  │
│  This motion includes:                          │
│  - Object's own movement                        │
│  - Minus camera's movement (appears as negative)│
│  = RELATIVE MOTION                              │
└─────────────────────────────────────────────────┘

                    ⬇

         Save for next frame:
         prevViewProj_saved = viewProj_N

                    ⬇
         Continue to Frame N+1...
```

### Implementation Pattern

```javascript
class Camera {
    constructor() {
        this.position = [0, 1, 4];
        this.yaw = -90;
        this.pitch = -10;

        // Current frame matrices
        this.viewMatrix = mat4.create();
        this.projMatrix = mat4.create();
        this.viewProjMatrix = mat4.create();

        // Previous frame (for motion vectors)
        this.prevViewProjMatrix = mat4.create();
    }

    updateMatrices(aspect) {
        // Save current → previous BEFORE updating
        mat4.copy(this.prevViewProjMatrix, this.viewProjMatrix);

        // Compute new matrices
        mat4.perspective(this.projMatrix, 45°, aspect, 0.1, 100);
        this._updateViewMatrix();  // from position + yaw/pitch
        mat4.multiply(this.viewProjMatrix, this.projMatrix, this.viewMatrix);
    }
}
```

---

## Physics Integration: Transform Computation

### Cannon.js Body → Transform Matrix

```javascript
function bodyToMatrix(body, out) {
    // Cannon.js stores:
    // - body.position: [x, y, z]
    // - body.quaternion: {x, y, z, w}

    const rot = [body.quaternion.x, body.quaternion.y, body.quaternion.z, body.quaternion.w];
    const pos = [body.position.x, body.position.y, body.position.z];

    mat4.fromRotationTranslation(out, rot, pos);
}

// Per physics step:
class PhysicsWorld {
    step(dt) {
        cannon.world.step(dt);

        for (let i = 0; i < bodies.length; i++) {
            const body = bodies[i];

            // Compute current transform
            const currentMatrix = mat4.create();
            bodyToMatrix(body, currentMatrix);

            // Store in GPU buffer
            bufferData[i*36 + 0:16] = currentMatrix;      // current
            bufferData[i*36 + 16:32] = body.prevMatrix;   // previous

            // Save for next frame
            mat4.copy(body.prevMatrix, currentMatrix);
        }
    }
}
```

---

## Single Draw Call Strategy

### Why Single Draw Call Works

```javascript
// Instead of:
for (let i = 0; i < bodies.length; i++) {
    pass.draw(vertexCount, 1);  // 100 draw calls ❌
}

// Do this:
pass.draw(vertexCount, 100);     // 1 draw call ✓

// GPU hardware:
// - Launches 100 instances
// - Each invocation has @builtin(instance_index) = 0..99
// - Vertex shader indexes storage buffer: instances[idx]
// - Shared vertex buffers across all instances
// - Typical speedup: 5-10x for many small objects
```

### Binding Setup

```javascript
const bindGroup = device.createBindGroup({
    layout: pipelineLayout.getBindGroupLayout(0),
    entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },       // camera
        { binding: 1, resource: { buffer: storageBuffer } },       // instances
        { binding: 2, resource: colorTexView },                    // outputs
        { binding: 3, resource: motionTexView },
    ],
});

const pass = cmdEncoder.beginRenderPass({
    colorAttachments: [
        { view: colorTexView, clearValue: [0.1, 0.1, 0.1, 1], loadOp: 'clear', storeOp: 'store' },
        { view: motionTexView, clearValue: [0, 0, 0, 0], loadOp: 'clear', storeOp: 'store' },
    ],
});

pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.setVertexBuffer(0, geometryBuffer);
pass.setIndexBuffer(indexBuffer, 'uint32');
pass.drawIndexed(geometryIndexCount, 100);  // 100 instances!
```

---

## Performance Characteristics

### Timeline for 100 Rigid Bodies

```
┌────────────────────────────────────────┐
│ Frame Duration (16.67 ms @ 60 FPS)     │
├────────────────────────────────────────┤
│                                         │
│ Physics Simulation:     ~2-3 ms (CPU)  │
│ - Gravity, collision    █░░░░░░░░░░░░░ │
│ - Constraints           █░░░░░░░░░░░░░ │
│ - Update bodies         █░░░░░░░░░░░░░ │
│                                         │
│ Transform Computation:  ~0.2 ms (CPU)  │
│ - Build matrices                 ░░░░░ │
│ - Batch into buffer              ░░░░░ │
│                                         │
│ GPU Upload:             ~0.1 ms        │
│ - Mapped buffer copy             ░░░░░ │
│                                         │
│ GPU Render:             ~1.5 ms (GPU)  │
│ - MRT render pass        ░░░░░░░░░░░░░ │
│ - Single instanced draw  ░░░░░░░░░░░░░ │
│                                         │
│ Compute Warp:           ~2.0 ms (GPU)  │
│ - Noise warp (existing)  ░░░░░░░░░░░░░ │
│                                         │
│ Display Pass:           ~0.5 ms (GPU)  │
│ - Fullscreen quad        ░░░░░░░░░░░░░ │
│                                         │
├────────────────────────────────────────┤
│ Total:                  ~6-7 ms / 16.67│ = 60 FPS ✓
└────────────────────────────────────────┘
```

### Bottleneck Analysis

**CPU-bound:** Physics simulation (2-3 ms for 100 bodies)
**GPU-bound:** Noise warp compute (2.0 ms for your resolution)
**Balanced:** Rendering is minor (1.5 ms)

**To improve:**
- Add more bodies → physics takes longer
- Increase resolution → warp takes longer
- Switch to GPU physics → loses CPU time but needs GPU memory (future optimization)

---

## Vertex Shader Anatomy (Complete Example)

```wgsl
struct Uniforms {
    viewProj:         mat4x4f,
    prevViewProj:     mat4x4f,
}

struct InstanceData {
    model:      mat4x4f,
    prevModel:  mat4x4f,
    color:      vec4f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;

struct VsIn {
    @location(0) position: vec3f,
    @location(1) normal:   vec3f,
}

struct VsOut {
    @builtin(position) position: vec4f,
    @location(0) color:    vec4f,
    @location(1) currClip: vec4f,
    @location(2) prevClip: vec4f,
    @location(3) normal:   vec3f,
}

@vertex fn vs(
    in: VsIn,
    @builtin(instance_index) idx: u32,
) -> VsOut {
    // Get instance-specific data
    let inst = instances[idx];

    // Transform vertex to world space (current & previous)
    let worldPos     = inst.model * vec4f(in.position, 1.0);
    let prevWorldPos = inst.prevModel * vec4f(in.position, 1.0);

    // Project to clip space
    let currClip = uniforms.viewProj * worldPos;
    let prevClip = uniforms.prevViewProj * prevWorldPos;

    // Transform normal to world space
    let worldNormal = (inst.model * vec4f(in.normal, 0.0)).xyz;

    var out: VsOut;
    out.position = currClip;
    out.color = inst.color;
    out.currClip = currClip;
    out.prevClip = prevClip;
    out.normal = worldNormal;

    return out;
}

@fragment fn fs(in: VsOut) -> vec4f {
    // Phong lighting
    let lightDir = normalize(vec3f(1.0, 1.0, 1.0));
    let diff = max(dot(normalize(in.normal), lightDir), 0.3);

    return vec4f(in.color.rgb * diff, 1.0);
}
```

### Fragment Shader in MRT Pipeline

```wgsl
struct FsOut {
    @location(0) color:  vec4f,
    @location(1) motion: vec4f,
}

@fragment fn fs(in: VsOut) -> FsOut {
    var out: FsOut;

    // Color output (location 0)
    let lightDir = normalize(vec3f(1.0, 1.0, 1.0));
    let diff = max(dot(normalize(in.normal), lightDir), 0.3);
    out.color = vec4f(in.color.rgb * diff, 1.0);

    // Motion output (location 1)
    let currNDC = in.currClip.xy / in.currClip.w;
    let prevNDC = in.prevClip.xy / in.prevClip.w;
    let screenMotion = (currNDC - prevNDC) * 0.5;  // scale for storage
    out.motion = vec4f(screenMotion, 0.0, 1.0);

    return out;
}
```

---

## Validation: Correct Motion Vectors

### Test Cases

**Case 1: Static Everything**
```
Camera: stationary
Bodies: stationary
Expected motion: [0, 0]
Actual: currNDC == prevNDC → difference is 0 ✓
```

**Case 2: Camera Moves Forward, Bodies Static**
```
Camera: forward (closer to bodies)
Bodies: stationary
Expected motion: [-small, 0] (bodies appear to shift backward on screen)
Actual: Camera moved forward → prevViewProj differs → prevNDC shifts → motion != 0 ✓
```

**Case 3: Body Moves Right, Camera Static**
```
Camera: stationary
Bodies: move right
Expected motion: [+large, 0] (body appears to move right on screen)
Actual: worldPos changes → currNDC shifts right → motion != 0 ✓
```

**Case 4: Both Move Same (Pursuit)**
```
Camera: forward, following body
Body: moves forward same speed
Expected motion: ≈ [0, 0] (body stays in same screen position)
Actual: Both move equally in same direction → NDC delta ≈ 0 ✓
```

### Debug Visualization

To verify motion vectors are correct, temporarily output them directly:

```wgsl
@fragment fn fs_debug(in: VsOut) -> vec4f {
    let currNDC = in.currClip.xy / in.currClip.w;
    let prevNDC = in.prevClip.xy / in.prevClip.w;
    let motion = (currNDC - prevNDC) * 0.5;

    // Visualize: magnitude = brightness
    let mag = length(motion) * 10.0;  // amplify for visibility
    return vec4f(vec3f(mag), 1.0);
}
```

Expected output:
- Black where nothing moved
- Bright where motion occurred
- Persists only while objects move (not frozen)

---

## Integration Checklist

- [ ] Add `sphereGeometry()`, `cylinderGeometry()` to geometry.js
- [ ] Create `InstanceData` struct in shaders.js with `prevModel`
- [ ] Update vertex shader to use `@builtin(instance_index)` and storage buffer
- [ ] Create storage buffer in renderer.js (size: maxBodies × 144)
- [ ] Implement `PhysicsManager` class with Cannon.js integration
- [ ] Add `bodyToMatrix()` helper function
- [ ] Implement mapped buffer upload pattern
- [ ] Track `prevMatrix` on each physics body
- [ ] Update camera matrix tracking (save `prevViewProj` each frame)
- [ ] Change draw call from `draw(n, 1)` to `draw(n, numBodies)`
- [ ] Test with 10 bodies first, then scale to 100
- [ ] Verify motion vectors with debug visualization
- [ ] Profile with DevTools (GPU timings)

---

## Expected Integration Size

- **Geometry additions:** ~100 lines
- **WGSL modifications:** ~30 lines (add struct, update VS)
- **PhysicsManager:** ~150 lines
- **Renderer updates:** ~50 lines
- **Main loop integration:** ~30 lines
- **Total new code:** ~360 lines (manageable!)

Existing MRT and noise warp code remain untouched.

