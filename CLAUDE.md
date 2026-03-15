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

- **OBJ round-trip for dynamic ops**: Dynamic mesh operations (topology changes) use temp OBJ files for input/output because extracting face-vertex data from RXMeshDynamic after cavity operations is complex. The overhead is negligible compared to the actual GPU work.

- **Float precision**: RXMesh uses float32 internally. We accept float64 from Python (numpy default), cast to float32 for GPU, cast back to float64 for output.

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

Note: needs Python 3.10 (matches the build target). The system may have multiple Python versions.
