# Boffins Room Session #20: Why Does GPU Isotropic Work But Adaptive Crashes?

**Date:** 2026-03-27
**Panel:** Dr. Rashid (GPU systems), Prof. Nguyen (geometry processing), Dr. Okafor (numerical methods)

---

**MODERATOR:** Gentlemen, the GPU isotropic remeshing pass works beautifully — split, collapse, cross-collapse, flip, smooth, BVH project, all clean. But when we feed that output through micro-collapse and run a second pass, the process dies with `double free or corruption` after smooth. Same kernels, same code path, different mesh. What's going on?

---

**Dr. Rashid:** First question — is it *actually* the same code path? Let me check... [reads `op_feature_remesh.cu`]

No. Pass 2 calls `compute_adaptive_sizing()` before the iteration loop. This creates three temporary RXMesh vertex attributes — `_qsum`, `_qcnt`, `_qtmp` — computes per-face quality, scatters to vertices, Laplacian-smooths them, maps to a sizing field, then *removes* all three attributes.

**Prof. Nguyen:** And pass 1 doesn't touch sizing at all?

**Dr. Rashid:** Correct. Pass 1 does `sizing->reset(1.0f, DEVICE)` and never looks at it again. No temp attributes created, no attributes removed.

**Dr. Okafor:** So the smoking gun is `rx.remove_attribute()` after `compute_adaptive_sizing()`. What happens to those attributes when topology changes later?

---

**Dr. Rashid:** Here's my theory. `compute_adaptive_sizing()` runs *before* the iteration loop. It creates attributes, computes quality, removes them. But RXMesh's attribute manager may keep dangling internal bookkeeping — freed device pointers still referenced in some table. Then during the iteration loop, `slice_patches()` tries to migrate or rebind attributes and hits the stale entry.

**Prof. Nguyen:** That's plausible, but let me offer an alternative. The crash happens after smooth, not during topology ops. The sequence is:

```
split (OK) → collapse (-1337 verts, OK) → crosses (-159, OK) → flip (OK) → smooth (OK) → CRASH
```

After smooth, the code calls `rx2.update_host()` then `rx2.export_obj()`. The `update_host()` synchronizes all registered attributes from device to host. If any attribute has inconsistent allocation state from the earlier `remove_attribute()` calls, that's your double-free.

**Dr. Okafor:** Wait — are the temp attributes even the problem? They're removed *before* the loop starts. By the time we split and collapse, they should be gone.

**Dr. Rashid:** "Should be" is doing a lot of heavy lifting there. RXMesh's `remove_attribute` calls `cudaFree` on the device buffer and `free` on the host buffer. But if `add_attribute` with `DEVICE` location didn't allocate a host buffer, then `remove_attribute` might try to free a null or uninitialized host pointer.

---

**Prof. Nguyen:** Let me reframe this. Forget the attributes for a moment. There's a much simpler explanation: **the mesh going into pass 2 is already fragile.**

Pass 1 creates a clean isotropic mesh. Then micro-collapse removes ~40 vertices by collapsing valence-3/4 interior vertices. This creates slightly irregular local topology — faces that share edges with recently-deleted vertices. The compaction after micro-collapse *should* clean this up, but we're going through an OBJ round-trip:

```
GPU pass 1 → export OBJ → Python reads OBJ → feeds to micro_collapse (CPU) → export OBJ → GPU pass 2 reads OBJ
```

If the OBJ export from pass 1 has any degenerate faces (we know Q max is 751!), and micro-collapse doesn't remove all of them, pass 2 inherits a mesh with near-degenerate triangles. The collapse kernel creates cavities around these triangles, and the cavity fill produces invalid geometry that corrupts the patch structure.

**Dr. Okafor:** That's the "garbage in, garbage out" theory. But pass 1 completed fine with those same degenerate triangles...

**Prof. Nguyen:** Pass 1 *created* them. It didn't inherit them. Big difference. When you create a degenerate triangle during collapse, it's in a freshly-filled cavity — the patch structure is locally consistent. When you *load* a mesh with a degenerate triangle and try to build patches around it, the patch decomposition might create a bad patch boundary that crosses the degenerate face.

---

**Dr. Rashid:** I want to test both theories. The attribute theory is easy to verify — just skip `compute_adaptive_sizing()` entirely and run pass 2 with uniform sizing. If it still crashes, it's not the attributes.

**Prof. Nguyen:** And the degenerate-input theory: run micro-collapse, then check Q max of the output. If it's clean (Q max < 50), the mesh isn't the problem.

**Dr. Okafor:** From the summary table, micro-collapse *does* clean things up — Q max goes from 751 to 23 after micro. So the mesh entering pass 2 should be fine.

**Prof. Nguyen:** Then it's the attributes. Or... there's a third possibility.

---

**Prof. Nguyen:** Cross-collapse. Look at the numbers:

```
Pass 1: crosses remove ~100-200 vertices (fine)
Pass 2: crosses remove 159 vertices (V 10292 → 10133)
```

The cross-collapse kernel uses `atomicCAS` conflict resolution and cavity operations. On a mesh that's already been through one pass of split/collapse, the patch structure has been `slice_patches`'d multiple times. Each slice doubles the patch count. By the time crosses run in pass 2, you might have 4x the original patches, each smaller. The one-per-face limit and quality checks might not prevent all conflicts.

**Dr. Rashid:** That's speculative. The collapse in pass 2 removed 1337 vertices without issue. Cross-collapse only removed 159. If the patch structure were corrupted, collapse would have crashed first.

**Prof. Nguyen:** Not necessarily. Collapse uses `CavityOp::EV` (edge-vertex cavity). Cross-collapse uses a *different* cavity pattern — it identifies the diamond around a vertex and collapses the entire thing. The diamond access pattern might touch a corrupted neighbor that EV doesn't.

**Dr. Okafor:** This is getting circular. Let's be empirical.

---

## CONSENSUS: Three Hypotheses, Ranked by Likelihood

### 1. Attribute lifecycle bug (65% confidence)
`compute_adaptive_sizing()` creates and removes temp attributes before the iteration loop. RXMesh's internal attribute registry may retain stale pointers. When `update_host()` or `export_obj()` iterates all attributes for synchronization, it hits the dangling entry.

**Test:** Comment out `compute_adaptive_sizing()`, use uniform sizing for pass 2. If crash disappears, confirmed.

### 2. Accumulated patch fragmentation (25% confidence)
Pass 2 operates on a mesh that's already been through one full remesh cycle + micro-collapse. The RXMeshDynamic constructed from this mesh may have suboptimal patch decomposition. Multiple `slice_patches` calls during the iteration loop further fragment patches until some internal buffer overflows.

**Test:** Increase `capacity_factor` from 3.5 to 5.0 for pass 2 only. Or: construct a fresh RXMeshDynamic from the micro-collapse output (currently it reuses pass 1's rx).

### 3. Cross-collapse diamond corruption (10% confidence)
The cross-collapse cavity pattern accesses a larger neighborhood than edge-collapse. On the already-modified mesh, this might read beyond valid patch boundaries.

**Test:** Skip cross-collapse in pass 2 (`PYRXMESH_NO_CROSSES=1`). If crash disappears, confirmed.

---

## RECOMMENDED FIX

**Short term:** Don't run `compute_adaptive_sizing()`. Pass 2 with uniform sizing is functionally equivalent — the sizing field was computed but never actually used by the split/collapse kernels anyway. Only the flip kernel receives it, and the impact on flip decisions is minimal.

**Medium term:** The CPU "adaptive" pass uses `minAdaptiveMult = maxAdaptiveMult = 1.0` (defaults, never overridden in our code), so the quality-weighted multiplier is always 1.0 — same thresholds as isotropic. BUT: (a) it runs for 15 iterations (not 1), and (b) it has `surfDistCheck` enabled with Hausdorff distance enforcement during collapse/smooth/flip. So GPU pass 2 should match: same kernels, same thresholds, but more iterations and eventually a surface distance check in collapse.

**Long term:** If true adaptive sizing is needed, compute the sizing field *once* before pass 2, store it as a persistent attribute (not temp), and never remove it during the iteration loop.

---

**MODERATOR:** So the boffins say: skip the adaptive sizing computation, run pass 2 as a plain isotropic pass on the micro-collapsed mesh, and it should match the CPU anyway. Dr. Rashid, final word?

**Dr. Rashid:** The funniest part? The sizing field is computed, stored in `sizing2`, passed to `feature_equalize_valences`... and the kernel ignores it. The split and collapse kernels use hardcoded `high2_sq` and `low2_sq`. We're crashing for a feature that doesn't work.

**Prof. Nguyen:** *[laughs]* Classic GPU programming. The bug is in the code you didn't need to write.
