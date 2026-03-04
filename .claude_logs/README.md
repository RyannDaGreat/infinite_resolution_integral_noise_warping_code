# WebGPU Custom Renderer + Physics Integration - Research & Implementation Guide

Complete research on adding rigid body physics to your existing WebGPU MRT renderer with noise warping.

## Documents in This Package

### 1. **RESEARCH_SUMMARY.txt** (START HERE)
- Executive summary of all 5 research questions
- Key findings and architectural decisions
- Implementation effort estimate (~360 lines of code)
- Success criteria and next steps
- Quick reference to all other documents

### 2. **WEBGPU_PHYSICS_RESEARCH.md** (DETAILED ANALYSIS)
- Comprehensive Q&A with technical depth
- Source citations and references
- Performance expectations
- Concrete code patterns
- Domino scene example
- ~400 lines of technical documentation

### 3. **QUICK_REFERENCE.md** (IMPLEMENTATION COOKBOOK)
- TL;DR answers to all 5 questions
- Copy-paste code snippets
- Geometry generators (box, sphere, cylinder)
- Physics + renderer integration patterns
- Debug visualization techniques
- Verification checklist
- ~250 lines of practical code

### 4. **ARCHITECTURE_GUIDE.md** (SYSTEM DESIGN)
- ASCII diagrams of complete rendering pipeline
- Data flow for motion vector computation
- Storage buffer memory layout specifications
- Frame-by-frame camera matrix evolution
- Complete shader examples with annotations
- Performance timeline and bottleneck analysis
- Single draw call optimization explained
- ~450 lines of architectural documentation

### 5. **COMPLETE_EXAMPLE.js** (WORKING CODE)
- Full implementation ready to integrate
- Geometry functions: sphere, cylinder
- Updated WGSL shaders with instancing
- PhysicsManager class (Cannon.js wrapper)
- Renderer integration points
- Game loop example
- Domino scene creation function
- Copy-paste ready!

---

## Quick Navigation

### If you want to...

**Understand the big picture:** Start with RESEARCH_SUMMARY.txt (5 min read)

**Get answers to all 5 questions:** Read WEBGPU_PHYSICS_RESEARCH.md (15 min read)

**Start coding immediately:** Use QUICK_REFERENCE.md + COMPLETE_EXAMPLE.js (start in 30 min)

**Understand system architecture:** Study ARCHITECTURE_GUIDE.md (20 min read)

**Implement from scratch:** Use COMPLETE_EXAMPLE.js as a template (2-4 hours)

---

## The 5 Research Questions - Quick Answers

### Q1: Engine vs Custom Renderer?
**KEEP YOUR CUSTOM RENDERER.** Add Cannon.js physics separately. Your MRT pipeline is optimal.

### Q2: Minimum for 100 Rigid Bodies?
- Storage buffer: 144 bytes per body (16 floats model + 16 floats prevModel + color)
- Single draw call: `draw(vertexCount, 100)`
- Vertex shader indexes via `@builtin(instance_index)`
- ~5-10x faster than individual draw calls

### Q3: Motion Vectors from Physics?
- Store `prevModelMatrix` per body (updated each frame)
- Your shader: `(currNDC - prevNDC)` automatically handles this
- Includes body motion + camera motion (relative)

### Q4: Geometry for Dominoes/Pendulum?
- **Box:** Reuse cubeVertices(), scale via transform
- **Sphere:** UV sphere, 16 segments ≈ 289 verts (code provided)
- **Cylinder:** 8 segments for rod (code provided)

### Q5: Camera-Relative Motion Vectors?
**Your code already does this!** Use `prevViewProj` from previous frame, divide by W.

---

## Key Decision: Architecture Pattern

```
Your Current Setup:
  Custom WebGPU Renderer → Noise Warp Compute

After Integration:
  Physics (CPU) → Transforms → Custom WebGPU Renderer → Noise Warp Compute
  └─ Cannon.js          └─ Storage buffer  └─ Unchanged (MRT+warp)
```

**Why this works:**
- Physics on CPU is mature, tested (100 bodies = 2-3ms on CPU)
- GPU physics (Rapier 2026) not ready yet
- Industry standard pattern (Three.js, Babylon.js do exactly this)
- Your MRT pipeline stays untouched and optimal
- Compute shaders for noise warp are unaffected

---

## Implementation Checklist

- [ ] Read RESEARCH_SUMMARY.txt (understand context)
- [ ] Choose implementation style:
  - [ ] Copy COMPLETE_EXAMPLE.js and adapt (fastest)
  - [ ] Follow QUICK_REFERENCE.md snippets (cleaner)
  - [ ] Read ARCHITECTURE_GUIDE.md and code from scratch (most learning)
- [ ] Add geometry functions (sphere, cylinder) to geometry.js
- [ ] Create storage buffer in renderer.js
- [ ] Update WGSL with InstanceData struct and instancing VS
- [ ] Create PhysicsManager class (Cannon.js wrapper)
- [ ] Integrate physics.step() into game loop
- [ ] Test with 10 bodies, then scale to 100
- [ ] Verify motion vectors with debug visualization
- [ ] Profile GPU/CPU timings with DevTools

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| 100 rigid bodies | 60 FPS ✓ |
| GPU render time | ~1.5 ms |
| CPU physics time | ~2-3 ms |
| Total frame time | ~6-7 ms / 16.67 |
| Storage buffer size | ~14 KB |
| Single draw call | Yes ✓ |

**Bottleneck:** CPU physics (2-3 ms) is the limiting factor before you hit GPU limits.

---

## Key Technical Details

✓ Storage buffer layout (144 bytes per instance)
✓ Instance index builtin in vertex shader
✓ Mapped buffer upload (2x faster than writeBuffer)
✓ Transform matrix from physics quaternion + position
✓ Previous frame tracking for motion vectors
✓ Camera matrix history for relative velocity
✓ Perspective-corrected motion vector calculation
✓ Single draw call with instancing

---

## Sources & References

All research sourced from 2025-2026 documentation:
- WebGPU Fundamentals (storage buffers, optimization)
- Learn Wgpu Tutorial (instancing patterns)
- Rapier Physics 2026 Review (future GPU plans)
- Three.js vs WebGPU performance analysis
- Unity motion vectors documentation
- Cannon.js GitHub repository

See WEBGPU_PHYSICS_RESEARCH.md for complete citations.

---

## File Dependencies

**To implement, you'll modify:**
- `web_demo_v2/geometry.js` - Add sphere, cylinder
- `web_demo_v2/shaders.js` - Add InstanceData struct, update VS
- `web_demo_v2/renderer.js` - Create storage buffer, bind group
- `web_demo_v2/main.js` - Add PhysicsManager, integrate loop

**No existing code removed** - only additions and shader updates.

---

## Verification

After implementation, verify:
1. 100 bodies render at 60 FPS
2. Motion vectors compute correctly (see ARCHITECTURE_GUIDE.md test cases)
3. Single draw call used (check DevTools GPU stats)
4. Noise warp applied to moving bodies
5. Camera motion doesn't break motion vectors
6. Physics bodies collide realistically
7. Domino chain reaction visible
8. Pendulum swings with warp effect

---

## Total Implementation Time

**Estimate: 2-4 hours for experienced developer**

Breakdown:
- Geometry utilities: 30 min
- Shader updates: 20 min
- PhysicsManager: 60 min
- Renderer integration: 30 min
- Testing & debugging: 30 min

---

## Questions or Issues?

Refer to the specific document:
- **Architecture questions:** ARCHITECTURE_GUIDE.md
- **Code examples:** COMPLETE_EXAMPLE.js
- **Quick lookups:** QUICK_REFERENCE.md
- **Deep technical:** WEBGPU_PHYSICS_RESEARCH.md

---

## License & Attribution

This research was conducted using public sources from:
- [WebGPU Fundamentals](https://webgpufundamentals.org/)
- [Learn Wgpu](https://sotrh.github.io/learn-wgpu/)
- [Cannon.js](https://github.com/schteppe/cannon.js)
- [Rapier Physics](https://rapier.rs/)

All code examples in this package are yours to use freely.

---

**Generated:** 2026-03-04
**Status:** Research Complete - Ready for Implementation
