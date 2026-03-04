# WebGPU Physics Integration - Quick Reference

## TL;DR Answers

### Q1: Engine vs Custom Renderer?
**KEEP CUSTOM RENDERER.** You already have MRT with motion vectors—exactly what you need. Add Cannon.js separately for physics.

### Q2: Minimum for 100 Rigid Bodies?
- Storage buffer with instance transforms (144 bytes each)
- Single draw call: `draw(vertexCount, 100)`
- Vertex shader indexes buffer via `@builtin(instance_index)`
- Update transforms on CPU, batch upload with mapped buffer

### Q3: Motion Vectors from Physics?
- Store `prevModelMatrix` per body
- Your shader already computes: `(currNDC - prevNDC)` which handles camera motion
- Save transforms from last frame to compute relative velocity

### Q4: Geometry for Dominoes/Pendulum?
- **Box:** Unit cube scaled in transform (reuse existing)
- **Sphere:** UV sphere, 16 segments ≈ 289 vertices
- **Cylinder:** 8 segments, separate caps + sides

### Q5: Camera-Relative Motion Vectors?
- You already do this! Use `prevViewProj` from camera last frame
- Multiply by `prevModel` to get previous clip space
- Divide by W for perspective correction
- Result: relative to camera automatically

---

## Minimal Integration Code

### 1. Add Storage Buffer Struct (WGSL)

```wgsl
struct InstanceData {
    model:      mat4x4f,    // 64 bytes
    prevModel:  mat4x4f,    // 64 bytes
    color:      vec4f,      // 16 bytes
}

@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;
```

### 2. Vertex Shader (Modified)

```wgsl
@vertex fn vs(
    in: VsIn,
    @builtin(instance_index) idx: u32,
) -> VsOut {
    let inst = instances[idx];
    let worldPos = inst.model * vec4f(in.position, 1.0);
    let prevWorldPos = inst.prevModel * vec4f(in.position, 1.0);

    var out: VsOut;
    out.currClip = u.viewProj * worldPos;
    out.prevClip = u.prevViewProj * prevWorldPos;
    out.color = inst.color;
    out.normal = (inst.model * vec4f(in.normal, 0.0)).xyz;
    out.position = out.currClip;
    return out;
}
```

### 3. JavaScript: Physics World + Instance Buffer

```javascript
import CANNON from 'cannon-es';

class PhysicsManager {
    constructor(device) {
        this.device = device;
        this.world = new CANNON.World();
        this.world.gravity.set(0, -9.82, 0);
        this.bodies = [];
        this.instanceBuffer = null;
        this.instanceData = null;
    }

    addBox(x, y, z, halfWidth, halfHeight, halfDepth) {
        const shape = new CANNON.Box(
            new CANNON.Vec3(halfWidth, halfHeight, halfDepth)
        );
        const body = new CANNON.Body({ mass: 1, shape });
        body.position.set(x, y, z);
        this.world.addBody(body);

        // Track previous transform
        body.prevMatrix = mat4.create();
        mat4.identity(body.prevMatrix);

        this.bodies.push(body);
        return body;
    }

    initBuffer(maxBodies = 200) {
        this.instanceBuffer = this.device.createBuffer({
            size: maxBodies * 144,  // InstanceData struct
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
        });

        this.instanceData = new Float32Array(maxBodies * 36);
    }

    step(dt, camera) {
        // Physics
        this.world.step(dt);

        // Update CPU-side instance data
        for (let i = 0; i < this.bodies.length; i++) {
            const body = this.bodies[i];

            // Build current matrix
            const matrix = mat4.create();
            mat4.fromRotationTranslation(
                matrix,
                [body.quaternion.x, body.quaternion.y, body.quaternion.z, body.quaternion.w],
                [body.position.x, body.position.y, body.position.z]
            );

            const offset = i * 36; // 144 bytes / 4 bytes per float

            // Store current model matrix (first 16 floats)
            this.instanceData.set(matrix, offset);

            // Store previous model matrix (next 16 floats)
            this.instanceData.set(body.prevMatrix, offset + 16);

            // Store color (last 4 floats)
            this.instanceData[offset + 32] = 0.5;
            this.instanceData[offset + 33] = 0.5;
            this.instanceData[offset + 34] = 0.5;
            this.instanceData[offset + 35] = 1.0;

            // Save current → previous for next frame
            mat4.copy(body.prevMatrix, matrix);
        }

        // GPU upload using mapped buffer
        this._uploadInstanceData(this.bodies.length);
    }

    _uploadInstanceData(numBodies) {
        const stagingBuffer = this.device.createBuffer({
            size: numBodies * 144,
            usage: GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });

        const range = stagingBuffer.getMappedRange();
        new Float32Array(range).set(this.instanceData.subarray(0, numBodies * 36));
        stagingBuffer.unmap();

        const cmd = this.device.createCommandEncoder();
        cmd.copyBufferToBuffer(
            stagingBuffer, 0,
            this.instanceBuffer, 0,
            numBodies * 144
        );
        this.device.queue.submit([cmd.finish()]);

        stagingBuffer.destroy();
    }

    getBindGroup(device, layout, textureView) {
        return device.createBindGroup({
            layout: layout,
            entries: [
                { binding: 1, resource: { buffer: this.instanceBuffer } },
            ],
        });
    }
}
```

### 4. Render Loop Integration

```javascript
async function gameLoop(renderer, physics, camera) {
    const startTime = performance.now();

    while (true) {
        const now = performance.now();
        const dt = Math.min((now - startTime) / 1000, 0.016); // Cap at 60fps

        // Update camera
        camera.processKeys(dt);
        camera.viewMatrix = camera.viewMatrix();

        // Step physics
        physics.step(dt, camera);

        // Render frame
        renderer.frame({
            model: physics.instanceBuffer,  // or pass instanceData
            viewProj: camera.viewProjMatrix,
            prevModel: physics.prevTransforms,  // managed internally
            prevViewProj: camera.prevViewProjMatrix,
            displayMode: 'motion',  // 'color', 'motion', 'side-by-side', etc.
        });

        await new Promise(r => requestAnimationFrame(r));
    }
}
```

---

## Geometry Utilities

### Box (Domino)
```javascript
export function boxGeometry(width = 1, height = 2, depth = 0.1) {
    // Use existing cubeVertices(), apply scale in transform
    return cubeVertices();
}
```

### Sphere
```javascript
export function sphereGeometry(radius = 0.5, segments = 16) {
    const verts = [];
    for (let r = 0; r <= segments; r++) {
        const phi = Math.PI * r / segments;
        for (let s = 0; s <= segments; s++) {
            const theta = 2 * Math.PI * s / segments;
            const x = Math.sin(phi) * Math.cos(theta) * radius;
            const y = Math.cos(phi) * radius;
            const z = Math.sin(phi) * Math.sin(theta) * radius;

            verts.push(x, y, z,  x, y, z,  0.2, 0.6, 0.2);  // pos, normal, color
        }
    }
    return new Float32Array(verts);
}
```

### Cylinder (Pendulum Rod)
```javascript
export function cylinderGeometry(radius = 0.05, height = 2.0, segments = 8) {
    const verts = [];
    const h_half = height / 2;

    // Top cap
    verts.push(0, h_half, 0, 0, 1, 0, 0.5, 0.5, 0.5);
    for (let i = 0; i <= segments; i++) {
        const angle = 2 * Math.PI * i / segments;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        verts.push(x, h_half, z, 0, 1, 0, 0.5, 0.5, 0.5);
    }

    // Side quads
    for (let h = 0; h < 2; h++) {
        const y = h_half - h * height;
        for (let i = 0; i <= segments; i++) {
            const angle = 2 * Math.PI * i / segments;
            const nx = Math.cos(angle);
            const nz = Math.sin(angle);
            const x = nx * radius;
            const z = nz * radius;
            verts.push(x, y, z, nx, 0, nz, 0.5, 0.5, 0.5);
        }
    }

    // Bottom cap
    verts.push(0, -h_half, 0, 0, -1, 0, 0.5, 0.5, 0.5);
    for (let i = 0; i <= segments; i++) {
        const angle = 2 * Math.PI * i / segments;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        verts.push(x, -h_half, z, 0, -1, 0, 0.5, 0.5, 0.5);
    }

    return new Float32Array(verts);
}
```

---

## Verification Checklist

- [ ] Storage buffer created with `GPUBufferUsage.STORAGE`
- [ ] Instance struct matches shader struct size (144 bytes)
- [ ] Vertex shader accesses `instances[@builtin(instance_index)]`
- [ ] Physics bodies initialized with `prevMatrix`
- [ ] `prevMatrix` updated after each physics step
- [ ] Mapped buffer created and unmapped before GPU submission
- [ ] Motion vectors computed as `(currNDC - prevNDC)`
- [ ] Stationary body shows camera motion in motion texture
- [ ] Moving body shows object + camera motion in motion texture

---

## Performance Tips

1. **Batch updates:** Use mapped buffer + copyBufferToBuffer (~2x faster than writeBuffer)
2. **Single draw call:** Instance count from physics body count, not individual draws
3. **Reuse geometry:** Scale boxes and spheres via transform, don't create variants
4. **Physics timestep:** Use fixed 1/60s or 1/120s, not variable dt (more stable)
5. **Profile:** Check GPU time with `device.queue.onSubmittedWorkDone()`

---

## Minimal Package.json Dependencies

```json
{
  "dependencies": {
    "cannon-es": "^0.20.0",
    "gl-matrix": "^3.4.3"
  },
  "devDependencies": {
    "webpack": "^5.0.0",
    "webpack-cli": "^4.0.0",
    "webpack-dev-server": "^4.0.0"
  }
}
```

---

## Key Files in Your Renderer

Already in place, just needs modification:

- **`shaders.js`** → Add `InstanceData` struct, update vertex shader
- **`renderer.js`** → Create/bind storage buffer
- **`main.js`** → Create `PhysicsManager`, integrate `step()`
- **`geometry.js`** → Add `sphereGeometry()`, `cylinderGeometry()`

Total changes: ~200 lines of code.

---

## Debug Motion Vectors

Display mode to verify motion vectors are correct:

```javascript
// In fragment shader display pass
@fragment fn fs(in: VsOut) -> vec4f {
    let motion = textureSample(motionTexture, sampler, in.texcoord);
    let mag = length(motion.xy) * 10.0; // Amplify for visibility
    return vec4f(vec3f(mag), 1.0);  // Show as grayscale
}
```

Expectations:
- Static camera, static bodies → black
- Static camera, moving bodies → bright
- Moving camera, static bodies → bright (camera motion)
- Moving camera, moving bodies → varies

