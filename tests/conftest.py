"""Test fixtures for pyrxmesh."""

import numpy as np
import pytest
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"
RXMESH_INPUT = Path(__file__).parent.parent / "RXMesh" / "input"


@pytest.fixture
def cube():
    """Closed cube mesh: 8 vertices, 12 triangular faces."""
    vertices = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],
    ], dtype=np.float64)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)

    return vertices, faces


@pytest.fixture
def icosphere():
    """Closed icosahedron (12 verts, 20 faces) on unit sphere."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=np.float64)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms

    faces = np.array([
        [0,11,5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ], dtype=np.int32)

    return vertices, faces


@pytest.fixture
def bunnyhead():
    """Open mesh with boundaries (from RXMesh inputs). Needed for SCP/Param."""
    obj_path = RXMESH_INPUT / "bunnyhead.obj"
    if not obj_path.exists():
        pytest.skip("bunnyhead.obj not found in RXMesh/input/")
    import pyrxmesh
    return pyrxmesh.load_obj(str(obj_path))


@pytest.fixture
def dragon():
    """Closed mesh (from RXMesh inputs). Needed for QSlim/SEC."""
    obj_path = RXMESH_INPUT / "dragon.obj"
    if not obj_path.exists():
        pytest.skip("dragon.obj not found in RXMesh/input/")
    import pyrxmesh
    return pyrxmesh.load_obj(str(obj_path))
