# pyrxmesh — Development Notes

## Architecture

pyrxmesh wraps RXMesh (GPU triangle mesh processing) via a **pipeline separation pattern**:

```
Python (pyrxmesh.Mesh / pyrxmesh.smooth etc.)
  → nanobind (_pyrxmesh.cpp, host compiler)
  → pipeline.h (pure C++ interface, no CUDA)
  → pipeline.cu + op_*.cu (nvcc, contains __device__ lambdas)
  → RXMesh library (git submodule)
```

This is mandatory because RXMesh headers contain CUDA constructs (`__device__`, cooperative_groups, etc.) that cannot be compiled by a host C++ compiler. The nanobind module only sees `pipeline.h`.

## Key Design Decisions

- **Persistent Mesh class**: `pyrxmesh.Mesh(v, f)` constructs RXMeshStatic once (~1s for patching) and reuses it for all operations. This gives 100x speedup over stateless calls for analysis ops (8ms vs 1000ms for vertex normals).

- **Stateless functions kept**: All 15 functions also exist as standalone `pyrxmesh.func(v, f, ...)` for one-off use. They pay the construction cost each time.

- **Dynamic mesh ops in separate .cu files**: QSlim, SEC, Remesh, Delaunay each get their own `op_*.cu` file because RXMesh app headers expect a global `Arg` struct. Using `static struct arg {...} Arg;` prevents ODR violations across TUs.

- **Float precision**: RXMesh uses float32 internally. We accept float64 from Python (numpy default), cast to float32 for GPU, cast back to float64 for output.

- **Triangle-only**: OBJ files with quads/polygons are rejected with a clear error. RXMesh only supports triangle meshes.

## Performance Optimizations (121s → ~48s for 3.6M vert Asian Dragon remesh)

### GPU-native pipeline
Most of the mesh construction pipeline has been moved to GPU:

1. **GPU OBJ parser** (`pipeline.cu`): Parallel text parsing on GPU. 12s → 1s for 374MB OBJ.
2. **GPU sort-scan topology** (`gpu_build_topology.cu`): Edge discovery via thrust sort on packed uint64 keys. 13s → 0.2s.
3. **Approach A thrust ltog** (`gpu_patch_build.cu`): Per-patch local-to-global arrays built via global thrust::sort + thrust::unique. 7s → 0.4s. No shared memory limits.
4. **GPU readback** (`op_remesh.cu`): Extract verts+faces via `run_query_kernel` on GPU. 5s → 2.7s.
5. **Batch build_device** (`rxmesh.cpp`): 17 bulk cudaMalloc (one per buffer type) instead of 340k individual calls. 8s → 1.5s.
6. **cudaMallocAsync pool** (`rxmesh.cpp`, `lp_hashtable.cu`, `patch_stash.cu`, `patch_lock.cu`): Uses CUDA memory pool to amortize allocation overhead.
7. **Binary search edge lookup** (`rxmesh.cpp`): `get_edge_id()` uses binary search on sorted `m_sorted_edge_keys` instead of `unordered_map`. Saves 5s construction.
8. **Flat face arrays** (`patcher.cu`, `rxmesh.cpp`): Patcher and topology build use `const uint32_t* flat_fv` instead of `vector<vector<uint32_t>>`. Eliminates millions of tiny heap allocations.
9. **Retained GPU arrays**: `d_edge_key`, `d_ev`, `d_ef_f0`, `d_fv` are kept on device between pipeline stages to avoid re-uploading.

### Key files for GPU pipeline
- `RXMesh/include/rxmesh/gpu_build_topology.cu` — GPU sort-scan edge construction
- `RXMesh/include/rxmesh/gpu_patch_build.cu` — Approach A (thrust ltog), K0a/K0b/K1/K2 kernels
- `RXMesh/include/rxmesh/gpu_patch_build.cuh` — Structs and function declarations
- `src/pipeline.cu` — GPU OBJ parser + GPU readback kernels
- `src/op_remesh.cu` — Remesh wrapper with GPU readback via `extract_mesh_gpu()`

### Edge ID space
All edge IDs in the system are in **GPU edge ID space** (position in sorted `d_edge_key`). There is no CPU vs GPU edge ID mismatch — `m_edges_map` was eliminated and `m_sorted_edge_keys` uses the same ordering. `get_edge_id()` does binary search on `m_sorted_edge_keys`.

### Current timing breakdown (Asian Dragon, 3.6M verts, 2 iterations)
| Step | Time | Where |
|------|------|-------|
| Load OBJ | 0.9s | GPU |
| GPU topology | 0.2s | GPU |
| Patcher Lloyd | 0.5s | GPU |
| Approach A ltog | 0.4s | GPU+CPU |
| CPU topology build | 2.0s | CPU |
| build_device | 1.5s | CPU+GPU |
| GPU remesh kernels | 28s | GPU (irreducible) |
| GPU readback | 2.7s | GPU |

### Remaining CPU bottlenecks to optimize
- **CPU topology build (2s)**: `build_single_patch_topology` — could be replaced by K2 GPU kernel (already written in gpu_patch_build.cu, needs edge ID fix)
- **build_device HT build (1.2s)**: Cuckoo hash table construction on CPU — hardest to GPU-ify
- **Approach A partition (0.3s)**: `stable_partition` on CPU, could move to GPU thrust
- **Prep + downloads (0.5s)**: `flat_verts_to_vv` still builds vector-of-vectors for vertex coordinates

## CUDA Compatibility (12.4+)

`patches/fix_cuda13_compat.py` handles API removals in CUDA 13:
- `cub::Sum`, `cub::Inequality` etc. → compat header at `RXMesh/include/rxmesh/util/cuda13_compat.h`
- `cudaDeviceProp.clockRate` → `cudaDeviceGetAttribute`
- `glm::distance2`, `glm::length2` → `src/glm_compat.h` with inline replacements
- OpenMesh dependency in Delaunay → `#ifdef PYRXMESH_HAS_OPENMESH` guards

All patches are guarded with `#if CUDART_VERSION >= 13000` so they're no-ops on CUDA 12.x.

The patch script must run before build (`pyproject.toml: before-build`). It's idempotent.

## Build

```bash
PATH="/usr/local/cuda/bin:$PATH" pip install --no-build-isolation -e .
```

Requires: CUDA toolkit, scikit-build-core, nanobind, cmake (all pip-installable except CUDA).

Build takes ~12 minutes (8 CUDA translation units with heavy template instantiation).

## CMake Structure

- `RXMesh/` built as static lib with `EXCLUDE_FROM_ALL`, apps/tests/polyscope disabled
- `gpu_build_topology.cu` and `gpu_patch_build.cu` registered in `RXMesh/cmake/RXMeshTarget.cmake`
- `rxmesh_pipeline` — static CUDA lib with `CUDA_SEPARABLE_COMPILATION ON` and `CUDA_RESOLVE_DEVICE_SYMBOLS ON`
- `_pyrxmesh` — nanobind module, linked to pipeline, with `CUDA_RESOLVE_DEVICE_SYMBOLS OFF` to prevent duplicate device link

## Adding New Operations

1. If the op uses RXMeshStatic (no topology changes): add kernel + wrapper to `pipeline.cu`, add to `pipeline.h`, `_pyrxmesh.cpp`, `core.py`
2. If the op uses RXMeshDynamic: create `op_newop.cu` with its own `static struct arg {} Arg;`, include the app header, add to `CMakeLists.txt`
3. If the app header uses `glm::distance2`: include `glm_compat.h` before the app header
4. If the app header uses OpenMesh: stub it out or `#ifdef` guard it

## Tests

46 tests covering all 15 functions. Run with:
```bash
/usr/bin/python3.10 -m pytest tests/ -v
```

## Benchmarks

Multi-mesh remesh benchmark:
```bash
python docs/bench_remesh.py
```

Full demo with all meshes (bunny, armadillo, happy buddha, Asian Dragon):
```bash
python docs/demo_pyrxmesh.py
```

Run single section:
```bash
PYRXMESH_DEMO_ONLY=armadillo_remesh python docs/demo_pyrxmesh.py
```
