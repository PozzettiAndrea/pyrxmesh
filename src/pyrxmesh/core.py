"""High-level Python wrappers for RXMesh GPU mesh operations."""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from pyrxmesh._pyrxmesh import (
    init as _init,
    mesh_info as _mesh_info,
    load_obj as _load_obj,
    vertex_normals as _vertex_normals,
    smooth as _smooth,
    gaussian_curvature as _gaussian_curvature,
    filter as _filter,
    mcf as _mcf,
    geodesic as _geodesic,
    scp as _scp,
    param as _param,
    qslim as _qslim,
    remesh as _remesh,
    sec as _sec,
    delaunay as _delaunay,
)


def _validate_mesh(vertices, faces):
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")
    return v, f


@dataclass
class MeshInfo:
    """Mesh topology statistics."""
    num_vertices: int
    num_edges: int
    num_faces: int
    is_edge_manifold: bool
    is_closed: bool
    max_valence: int
    num_components: int


def init(device_id: int = 0) -> None:
    """Initialize CUDA device."""
    _init(device_id)


def mesh_info(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
) -> MeshInfo:
    """Get mesh topology statistics."""
    v, f = _validate_mesh(vertices, faces)
    return MeshInfo(*_mesh_info(v, f))


def load_obj(path: str) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Load triangle mesh from OBJ file."""
    return _load_obj(path)


def vertex_normals(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compute area-weighted vertex normals on GPU (Max 1999)."""
    v, f = _validate_mesh(vertices, faces)
    return _vertex_normals(v, f)


def smooth(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    iterations: int = 10,
    lambda_: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Laplacian mesh smoothing on GPU."""
    v, f = _validate_mesh(vertices, faces)
    return _smooth(v, f, iterations, lambda_)


def gaussian_curvature(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compute discrete Gaussian curvature per vertex (Meyer et al. 2003).

    Returns per-vertex scalar curvature values (K = angle_defect / mixed_area).
    """
    v, f = _validate_mesh(vertices, faces)
    return _gaussian_curvature(v, f)


def filter(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    iterations: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Bilateral mesh denoising on GPU (Fleishman et al. 2003)."""
    v, f = _validate_mesh(vertices, faces)
    return _filter(v, f, iterations)


def mcf(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    time_step: float = 10.0,
    use_uniform_laplace: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Mean Curvature Flow smoothing via Cholesky solver (Desbrun et al. 1999)."""
    v, f = _validate_mesh(vertices, faces)
    return _mcf(v, f, time_step, use_uniform_laplace)


def geodesic(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    seeds: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compute geodesic distances from seed vertices on GPU.

    Parameters
    ----------
    seeds : array-like, shape (K,)
        Indices of seed vertices (distance = 0).

    Returns
    -------
    distances : ndarray, shape (N,), float64
        Geodesic distance from nearest seed for each vertex.
    """
    v, f = _validate_mesh(vertices, faces)
    s = np.ascontiguousarray(seeds, dtype=np.int32)
    if s.ndim != 1:
        raise ValueError(f"seeds must be 1D, got {s.shape}")
    return _geodesic(v, f, s)


def scp(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    iterations: int = 32,
) -> NDArray[np.float64]:
    """Spectral Conformal Parameterization (UV via power method).

    Requires a mesh with boundaries (not closed).

    Returns
    -------
    uv : ndarray, shape (N, 2), float64
        Normalized UV coordinates per vertex.
    """
    v, f = _validate_mesh(vertices, faces)
    return _scp(v, f, iterations)


def param(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    newton_iterations: int = 100,
) -> NDArray[np.float64]:
    """UV Parameterization via Tutte embedding + symmetric Dirichlet energy.

    Uses Newton optimization to minimize distortion. Requires a mesh
    with boundaries (not closed).

    Returns
    -------
    uv : ndarray, shape (N, 2), float64
        Optimized UV coordinates per vertex.
    """
    v, f = _validate_mesh(vertices, faces)
    return _param(v, f, newton_iterations)


def qslim(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    target_ratio: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """QSlim mesh decimation via edge collapse.

    Parameters
    ----------
    target_ratio : float
        Fraction of vertices to keep (0.1 = keep 10%).
    """
    v, f = _validate_mesh(vertices, faces)
    return _qslim(v, f, target_ratio)


def remesh(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    relative_len: float = 1.0,
    iterations: int = 3,
    smooth_iterations: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Isotropic remeshing (split/collapse/flip/smooth).

    Parameters
    ----------
    relative_len : float
        Target edge length as ratio of input average edge length.
    """
    v, f = _validate_mesh(vertices, faces)
    return _remesh(v, f, relative_len, iterations, smooth_iterations)


def sec(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    target_ratio: float = 0.5,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Shortest-edge-collapse decimation.

    Parameters
    ----------
    target_ratio : float
        Fraction of vertices to keep (0.1 = keep 10%).
    """
    v, f = _validate_mesh(vertices, faces)
    return _sec(v, f, target_ratio)


def delaunay(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Delaunay edge flipping (maximize minimum angles)."""
    v, f = _validate_mesh(vertices, faces)
    return _delaunay(v, f)
