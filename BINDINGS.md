# Binding Coverage

## Mapped

| Function | Description |
|----------|-------------|
| `init` | Initialize CUDA device |
| `mesh_info` | Mesh statistics (vertex/edge/face counts, manifold, closed, valence, components) |
| `load_obj` | Load triangle mesh from OBJ file |
| `vertex_normals` | GPU-accelerated area-weighted vertex normals (Max 1999) |
| `smooth` | GPU-accelerated Laplacian smoothing |

## Not Mapped

| Capability | Notes |
|------------|-------|
| ARAP deformation | Complex app, needs iterative solver + boundary conditions |
| Geodesic distances | Requires heat method solver chain |
| Remeshing | Dynamic topology via RXMeshDynamic (edge flip/collapse/split) |
| Parameterization | Tutte embedding + sparse solvers |
| Sparse/Dense matrices | SparseMatrix, DenseMatrix with cuSolver/cuSparse/cuBLAS |
| Linear solvers | Cholesky, CG, PCG, QR, LU, GMG |
| Automatic differentiation | DiffScalarProblem, DiffVectorProblem + optimizers |
| Mean curvature flow | Implicit solver required |
| Edge/face attributes | Only vertex attributes exposed currently |
| Custom CUDA kernels | Would need CuPy/Numba integration |
| VTK export | Available in RXMesh via `export_obj` / VTK utils |
| Multi-mesh | RXMeshStatic constructor accepts multiple OBJ files |
| Patch visualization | Polyscope integration |
