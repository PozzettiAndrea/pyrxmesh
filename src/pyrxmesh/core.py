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
    edge_split as _edge_split,
    edge_collapse as _edge_collapse,
    edge_flip as _edge_flip,
    patch_info as _patch_info,
    detect_features as _detect_features,
    expected_edge_length as _expected_edge_length,
    feature_remesh as _feature_remesh,
    quadwild_preprocess as _quadwild_preprocess,
    vcg_remesh as _vcg_remesh,
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
    verbose: bool = False,
) -> NDArray[np.float64]:
    """Compute area-weighted vertex normals on GPU (Max 1999)."""
    v, f = _validate_mesh(vertices, faces)
    return _vertex_normals(v, f, verbose)


def smooth(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    iterations: int = 10,
    lambda_: float = 0.5,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Laplacian mesh smoothing on GPU."""
    v, f = _validate_mesh(vertices, faces)
    return _smooth(v, f, iterations, lambda_, verbose)


def gaussian_curvature(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    verbose: bool = False,
) -> NDArray[np.float64]:
    """Compute discrete Gaussian curvature per vertex (Meyer et al. 2003).

    Returns per-vertex scalar curvature values (K = angle_defect / mixed_area).
    """
    v, f = _validate_mesh(vertices, faces)
    return _gaussian_curvature(v, f, verbose)


def filter(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    iterations: int = 5,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Bilateral mesh denoising on GPU (Fleishman et al. 2003)."""
    v, f = _validate_mesh(vertices, faces)
    return _filter(v, f, iterations, verbose)


def mcf(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    time_step: float = 10.0,
    use_uniform_laplace: bool = True,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Mean Curvature Flow smoothing via Cholesky solver (Desbrun et al. 1999)."""
    v, f = _validate_mesh(vertices, faces)
    return _mcf(v, f, time_step, use_uniform_laplace, verbose)


def geodesic(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    seeds: NDArray[np.int32],
    verbose: bool = False,
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
    return _geodesic(v, f, s, verbose)


def scp(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    iterations: int = 32,
    verbose: bool = False,
) -> NDArray[np.float64]:
    """Spectral Conformal Parameterization (UV via power method).

    Requires a mesh with boundaries (not closed).

    Returns
    -------
    uv : ndarray, shape (N, 2), float64
        Normalized UV coordinates per vertex.
    """
    v, f = _validate_mesh(vertices, faces)
    return _scp(v, f, iterations, verbose)


def param(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    newton_iterations: int = 100,
    verbose: bool = False,
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
    return _param(v, f, newton_iterations, verbose)


def qslim(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    target_ratio: float = 0.5,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """QSlim mesh decimation via edge collapse.

    Parameters
    ----------
    target_ratio : float
        Fraction of vertices to keep (0.1 = keep 10%).
    """
    v, f = _validate_mesh(vertices, faces)
    return _qslim(v, f, target_ratio, verbose)


def remesh(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    relative_len: float = 1.0,
    iterations: int = 3,
    smooth_iterations: int = 5,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Isotropic remeshing (split/collapse/flip/smooth).

    Parameters
    ----------
    relative_len : float
        Target edge length as ratio of input average edge length.
    """
    v, f = _validate_mesh(vertices, faces)
    return _remesh(v, f, relative_len, iterations, smooth_iterations, verbose)


def sec(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    target_ratio: float = 0.5,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Shortest-edge-collapse decimation.

    Parameters
    ----------
    target_ratio : float
        Fraction of vertices to keep (0.1 = keep 10%).
    """
    v, f = _validate_mesh(vertices, faces)
    return _sec(v, f, target_ratio, verbose)


def delaunay(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Delaunay edge flipping (maximize minimum angles)."""
    v, f = _validate_mesh(vertices, faces)
    return _delaunay(v, f, verbose)


# ── GPU feature detection ───────────────────────────────────────────────────


@dataclass
class FeatureData:
    """Per-element feature detection results."""
    edge_is_feature: NDArray[np.int32]
    vertex_is_feature: NDArray[np.int32]
    vertex_is_boundary: NDArray[np.int32]
    num_feature_edges: int


def detect_features(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    crease_angle_deg: float = 35.0,
    erode_dilate_steps: int = 4,
    verbose: bool = False,
) -> FeatureData:
    """Detect feature edges on GPU via dihedral angle threshold.

    Marks edges where the angle between adjacent face normals exceeds
    crease_angle_deg, plus boundary and non-manifold edges.

    Returns a FeatureData with:
    - edge_is_feature: (E,) int — 1 if feature edge, 0 otherwise
    - vertex_is_feature: (N,) int — 1 if vertex touches a feature edge
    - vertex_is_boundary: (N,) int — 1 if vertex is on mesh boundary
    - num_feature_edges: int
    """
    v, f = _validate_mesh(vertices, faces)
    ef, vf, vb, nfe = _detect_features(v, f, crease_angle_deg, erode_dilate_steps, verbose)
    return FeatureData(ef, vf, vb, nfe)


@dataclass
class EdgeLengthData:
    """GPU-computed mesh statistics and target edge length."""
    area: float
    volume: float
    sphericity: float
    target_edge_length: float
    avg_edge_length: float


def expected_edge_length(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    min_faces: int = 10000,
    verbose: bool = False,
) -> EdgeLengthData:
    """Compute QuadWild's ExpectedEdgeL on GPU.

    Uses GPU reductions for mesh area and signed volume, then computes
    sphericity-based target edge length matching QuadWild's formula.
    """
    v, f = _validate_mesh(vertices, faces)
    a, vol, sph, tel, ael = _expected_edge_length(v, f, min_faces, verbose)
    return EdgeLengthData(a, vol, sph, tel, ael)


# ── VCG CPU remeshing ───────────────────────────────────────────────────────


def vcg_remesh(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    target_edge_length: float = 0.0,
    target_faces: int = 10000,
    iterations: int = 3,
    adaptive: bool = True,
    project: bool = True,
    crease_angle_deg: float = 35.0,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """CPU isotropic remeshing via VCG (same algorithm as QuadWild).

    Two-pass remeshing: first non-adaptive, then adaptive with surface
    projection. Matches QuadWild's AutoRemesher::RemeshAdapt().

    Parameters
    ----------
    target_edge_length : float
        Target edge length. 0 = auto from sqrt(area * 2.309 / target_faces).
    target_faces : int
        Target face count for auto edge length (default: 10000).
    iterations : int
        Split+collapse+flip+smooth iterations per pass.
    adaptive : bool
        Run second adaptive pass like QuadWild (default: True).
    project : bool
        Project vertices to original surface (default: True).
    crease_angle_deg : float
        Feature edge angle threshold in degrees (default: 35).
    """
    v, f = _validate_mesh(vertices, faces)
    return _vcg_remesh(v, f, target_edge_length, target_faces, iterations,
                       adaptive, project, crease_angle_deg, verbose)


def feature_remesh(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    relative_len: float = 1.0,
    iterations: int = 15,
    smooth_iterations: int = 5,
    crease_angle_deg: float = 35.0,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Feature-aware GPU isotropic remeshing.

    Like remesh() but detects feature edges (dihedral angle > crease_angle_deg)
    and skips them during split/collapse/flip operations, preserving sharp edges.
    """
    v, f = _validate_mesh(vertices, faces)
    return _feature_remesh(v, f, relative_len, iterations, smooth_iterations,
                           crease_angle_deg, verbose)


# ── QuadWild preprocessing ──────────────────────────────────────────────────


def quadwild_preprocess(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    target_edge_length: float = 0.0,
    target_faces: int = 10000,
    num_iterations: int = 15,
    num_smooth_iters: int = 5,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """GPU isotropic remeshing for QuadWild preprocessing.

    Auto-computes target edge length using QuadWild's ExpectedEdgeL formula
    (sphericity-based: accounts for mesh volume/surface ratio).

    Parameters
    ----------
    target_edge_length : float
        Target edge length. 0 = auto via ExpectedEdgeL (sphericity correction).
    target_faces : int
        MinFaces parameter for ExpectedEdgeL (default: 10000).
    num_iterations : int
        Outer split+collapse+flip+smooth loops.
    num_smooth_iters : int
        Inner smoothing sub-iterations per outer loop.
    """
    v, f = _validate_mesh(vertices, faces)
    return _quadwild_preprocess(v, f, target_edge_length, target_faces,
                                num_iterations, num_smooth_iters, verbose)


# ── Patch visualization ─────────────────────────────────────────────────────


@dataclass
class PatchData:
    """Per-element patch decomposition data for visualization."""
    vertex_patch_ids: NDArray[np.int32]
    face_patch_ids: NDArray[np.int32]
    vertex_is_ribbon: NDArray[np.int32]
    face_is_ribbon: NDArray[np.int32]
    num_patches: int


def patch_info(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
) -> PatchData:
    """Get per-vertex and per-face patch IDs and ribbon masks.

    Returns a PatchData with:
    - vertex_patch_ids: (N,) int — owning patch index per vertex
    - face_patch_ids: (M,) int — owning patch index per face
    - vertex_is_ribbon: (N,) int — 1 if vertex is in ribbon (shared), 0 if owned
    - face_is_ribbon: (M,) int — 1 if face is in ribbon, 0 if owned
    - num_patches: int — total number of patches
    """
    v, f = _validate_mesh(vertices, faces)
    vp, fp, vr, fr, np_ = _patch_info(v, f)
    return PatchData(vp, fp, vr, fr, np_)


# ── QuadWild full pipeline (calls vendored binary) ──────────────────────────


def quadwild_pipeline(
    input_obj: str,
    output_dir: str = "/tmp/pyrxmesh_quadwild",
    steps: int = 4,
    setup_file: str = None,
) -> dict:
    """Run the full QuadWild pipeline on an OBJ file.

    Calls the vendored quadwild binary (must be built first with build_quadwild.sh).

    Parameters
    ----------
    input_obj : str
        Path to input triangle mesh OBJ file.
    output_dir : str
        Directory for output files.
    steps : int
        How many steps to run (1=remesh, 2=+field, 3=+trace, 4=+quad).
    setup_file : str
        Path to QuadWild setup.txt. None = use default.

    Returns
    -------
    dict with keys: 'remeshed', 'traced', 'output_dir', 'returncode', 'stdout'
    """
    import subprocess
    import os
    import shutil

    # Find quadwild binary
    pyrxmesh_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    qw_bin = os.path.join(pyrxmesh_root, "quadwild", "build", "Build", "bin", "quadwild")
    if not os.path.exists(qw_bin):
        # Try the source tree location
        qw_bin = os.path.join(pyrxmesh_root, "..", "quadwild-bimdf-cuda", "build_ninja", "Build", "bin", "quadwild")
    if not os.path.exists(qw_bin):
        raise RuntimeError(
            f"QuadWild binary not found. Build it with: ./build_quadwild.sh\n"
            f"Searched: {qw_bin}"
        )

    os.makedirs(output_dir, exist_ok=True)

    # Copy input mesh to output dir
    base = os.path.splitext(os.path.basename(input_obj))[0]
    work_obj = os.path.join(output_dir, f"{base}.obj")
    shutil.copy(input_obj, work_obj)

    # Write default setup if none provided
    if setup_file is None:
        setup_file = os.path.join(output_dir, "setup.txt")
        with open(setup_file, "w") as f:
            f.write("do_remesh 1\nsharp_feature_thr 35\nalpha 0.01\nscaleFact 1\n")

    # Run quadwild
    cmd = [qw_bin, work_obj, str(steps), setup_file]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                            cwd=output_dir)

    out = {
        "output_dir": output_dir,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "remeshed": os.path.join(output_dir, f"{base}_rem.obj"),
        "traced": os.path.join(output_dir, f"{base}_rem_p0.obj"),
        "quad": os.path.join(output_dir, f"{base}_rem_quadrangulation.obj"),
        "quad_smooth": os.path.join(output_dir, f"{base}_rem_quadrangulation_smooth.obj"),
        "steps": [],  # parsed [STEP] markers
    }

    # Parse [STEP] and [CKPT] markers from stdout
    for line in result.stdout.splitlines():
        if "[STEP]" in line or "[CKPT]" in line:
            out["steps"].append(line.strip())

    # Collect checkpoint OBJ files
    out["checkpoints"] = {}
    for name in ["step_1_0_input", "step_1_1a_features_raw", "step_1_1b_features_eroded",
                 "step_1_2_remeshed", "step_1_3_cleaned", "step_1_4_refined", "step_1_5_field"]:
        obj_path = os.path.join(output_dir, f"{name}.obj")
        sharp_path = os.path.join(output_dir, f"{name}.sharp")
        rosy_path = os.path.join(output_dir, f"{name}.rosy")
        if os.path.exists(obj_path):
            out["checkpoints"][name] = {
                "obj": obj_path,
                "sharp": sharp_path if os.path.exists(sharp_path) else None,
                "rosy": rosy_path if os.path.exists(rosy_path) else None,
            }

    # Check which outputs exist
    for key in ["remeshed", "traced", "quad", "quad_smooth"]:
        if not os.path.exists(out[key]):
            out[key] = None

    return out


# ── Standalone edge operations ──────────────────────────────────────────────


def edge_split(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    relative_len: float = 1.0,
    iterations: int = 1,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Split long edges on GPU.

    Splits edges longer than (4/3) * relative_len * avg_edge_length.
    This is the split sub-operation from isotropic remeshing.

    Parameters
    ----------
    relative_len : float
        Target edge length as ratio of input average edge length.
    """
    v, f = _validate_mesh(vertices, faces)
    return _edge_split(v, f, relative_len, iterations, verbose)


def edge_collapse(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    relative_len: float = 1.0,
    iterations: int = 1,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Collapse short edges on GPU.

    Collapses edges shorter than (4/5) * relative_len * avg_edge_length.
    This is the collapse sub-operation from isotropic remeshing.

    Parameters
    ----------
    relative_len : float
        Target edge length as ratio of input average edge length.
    """
    v, f = _validate_mesh(vertices, faces)
    return _edge_collapse(v, f, relative_len, iterations, verbose)


def edge_flip(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    iterations: int = 1,
    verbose: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Flip edges to equalize vertex valences on GPU.

    Flips edges when it reduces the deviation from target valence (6).
    This is the flip sub-operation from isotropic remeshing.
    """
    v, f = _validate_mesh(vertices, faces)
    return _edge_flip(v, f, iterations, verbose)
