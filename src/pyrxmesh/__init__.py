"""pyrxmesh -- Python bindings for RXMesh (GPU-accelerated mesh processing)."""

from pyrxmesh._pyrxmesh import Mesh
from pyrxmesh.core import (
    init,
    mesh_info,
    load_obj,
    vertex_normals,
    smooth,
    gaussian_curvature,
    filter,
    mcf,
    geodesic,
    scp,
    param,
    qslim,
    remesh,
    sec,
    delaunay,
)

__version__ = "0.1.0"
__all__ = [
    "Mesh",
    "init",
    "mesh_info",
    "load_obj",
    "vertex_normals",
    "smooth",
    "gaussian_curvature",
    "filter",
    "mcf",
    "geodesic",
    "scp",
    "param",
    "qslim",
    "remesh",
    "sec",
    "delaunay",
    "__version__",
]
