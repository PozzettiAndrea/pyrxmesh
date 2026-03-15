# pyrxmesh

Python bindings for [RXMesh](https://github.com/owensgroup/RXMesh) — GPU-accelerated triangle mesh processing.

<div align="center">
<a href="https://pozzettiandrea.github.io/pyrxmesh/">
<img src="https://pozzettiandrea.github.io/pyrxmesh/preview.png" alt="Geodesic Distance Demo" width="800">
</a>
<br>
<em><a href="https://pozzettiandrea.github.io/pyrxmesh/">View all demos</a></em>
</div>

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.4+
- CMake 3.24+
- Python 3.10+

## Installation

```bash
pip install .
```

## API

### Analysis
```python
import pyrxmesh
import numpy as np

pyrxmesh.init()
v, f = pyrxmesh.load_obj("mesh.obj")

info = pyrxmesh.mesh_info(v, f)
normals = pyrxmesh.vertex_normals(v, f)
curvature = pyrxmesh.gaussian_curvature(v, f)
distances = pyrxmesh.geodesic(v, f, seeds=np.array([0], dtype=np.int32))
```

### Smoothing
```python
v_smooth, f_smooth = pyrxmesh.smooth(v, f, iterations=50, lambda_=0.5)
v_filtered, f_filtered = pyrxmesh.filter(v, f, iterations=5)
v_mcf, f_mcf = pyrxmesh.mcf(v, f, time_step=1.0)
```

### Parameterization (requires mesh with boundaries)
```python
uv_scp = pyrxmesh.scp(v, f, iterations=32)          # spectral conformal
uv_opt = pyrxmesh.param(v, f, newton_iterations=100) # Tutte + Dirichlet
```

### Decimation (requires closed manifold)
```python
v_dec, f_dec = pyrxmesh.qslim(v, f, target_ratio=0.25)  # quadric collapse
v_dec, f_dec = pyrxmesh.sec(v, f, target_ratio=0.25)     # shortest-edge collapse
```

### Remeshing
```python
v_re, f_re = pyrxmesh.remesh(v, f, relative_len=1.0, iterations=3)
v_del, f_del = pyrxmesh.delaunay(v, f)
```

## Credits

RXMesh by [Ahmed H. Mahmoud](https://github.com/Ahdhn) and the [Owens Research Group](https://github.com/owensgroup) at UC Davis.

BSD-2-Clause License.
