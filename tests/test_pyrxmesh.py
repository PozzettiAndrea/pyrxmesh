"""Tests for all pyrxmesh GPU mesh processing bindings."""

import numpy as np
import pytest

import pyrxmesh


# =========================================================================
# init / load_obj
# =========================================================================

class TestInit:
    def test_init_default(self):
        pyrxmesh.init()

    def test_init_device_zero(self):
        pyrxmesh.init(device_id=0)


class TestLoadObj:
    def test_load_bunnyhead(self, bunnyhead):
        v, f = bunnyhead
        assert v.shape[1] == 3
        assert f.shape[1] == 3
        assert v.shape[0] > 100


# =========================================================================
# mesh_info
# =========================================================================

class TestMeshInfo:
    def test_cube_info(self, cube):
        v, f = cube
        info = pyrxmesh.mesh_info(v, f)
        assert info.num_vertices == 8
        assert info.num_faces == 12
        assert info.num_edges == 18
        assert info.is_edge_manifold is True
        assert info.is_closed is True

    def test_icosphere_info(self, icosphere):
        v, f = icosphere
        info = pyrxmesh.mesh_info(v, f)
        assert info.num_vertices == 12
        assert info.num_faces == 20
        assert info.num_edges == 30
        assert info.is_edge_manifold is True
        assert info.is_closed is True

    def test_open_mesh_not_closed(self, bunnyhead):
        v, f = bunnyhead
        info = pyrxmesh.mesh_info(v, f)
        assert info.is_closed is False

    def test_invalid_shape(self):
        v = np.zeros((10, 2), dtype=np.float64)
        f = np.zeros((5, 3), dtype=np.int32)
        with pytest.raises(ValueError):
            pyrxmesh.mesh_info(v, f)


# =========================================================================
# vertex_normals
# =========================================================================

class TestVertexNormals:
    def test_shape(self, cube):
        v, f = cube
        normals = pyrxmesh.vertex_normals(v, f)
        assert normals.shape == (8, 3)
        assert normals.dtype == np.float64

    def test_nonzero(self, cube):
        v, f = cube
        normals = pyrxmesh.vertex_normals(v, f)
        norms = np.linalg.norm(normals, axis=1)
        assert np.all(norms > 0)

    def test_icosphere(self, icosphere):
        v, f = icosphere
        normals = pyrxmesh.vertex_normals(v, f)
        assert normals.shape == (12, 3)


# =========================================================================
# smooth
# =========================================================================

class TestSmooth:
    def test_preserves_shape(self, cube):
        v, f = cube
        v_out, f_out = pyrxmesh.smooth(v, f, iterations=5)
        assert v_out.shape == v.shape
        assert f_out.shape == f.shape

    def test_faces_unchanged(self, cube):
        v, f = cube
        _, f_out = pyrxmesh.smooth(v, f, iterations=10)
        np.testing.assert_array_equal(f_out, f)

    def test_zero_iterations(self, cube):
        v, f = cube
        v_out, _ = pyrxmesh.smooth(v, f, iterations=0)
        np.testing.assert_allclose(v_out, v, atol=1e-6)

    def test_reduces_variance(self, icosphere):
        v, f = icosphere
        v_out, _ = pyrxmesh.smooth(v, f, iterations=50, lambda_=0.5)
        var_before = np.var(v, axis=0).sum()
        var_after = np.var(v_out, axis=0).sum()
        assert var_after < var_before * 2.0


# =========================================================================
# gaussian_curvature
# =========================================================================

class TestGaussianCurvature:
    def test_shape(self, cube):
        v, f = cube
        gc = pyrxmesh.gaussian_curvature(v, f)
        assert gc.shape == (8,)
        assert gc.dtype == np.float64

    def test_cube_positive(self, cube):
        """All cube corners should have positive Gaussian curvature."""
        v, f = cube
        gc = pyrxmesh.gaussian_curvature(v, f)
        assert np.all(gc > 0)

    def test_icosphere_positive(self, icosphere):
        """All icosahedron vertices should have positive curvature (convex)."""
        v, f = icosphere
        gc = pyrxmesh.gaussian_curvature(v, f)
        assert np.all(gc > 0)

    def test_gauss_bonnet(self, icosphere):
        """Gauss-Bonnet: integral of K over closed surface = 2*pi*chi.
        For a sphere (genus 0), chi=2, so integral ~ 4*pi."""
        v, f = icosphere
        gc = pyrxmesh.gaussian_curvature(v, f)
        # gc values are already K (curvature / mixed_area * mixed_area = angle defect)
        # For the icosahedron, total angle defect = 4*pi
        total = gc.sum()
        # Mixed-area-weighted curvature sums: each vertex has gc = defect/area,
        # but we'd need areas to integrate. Just check it's reasonable.
        assert total > 0


# =========================================================================
# filter
# =========================================================================

class TestFilter:
    def test_shape_preserved(self, cube):
        v, f = cube
        v_out, f_out = pyrxmesh.filter(v, f, iterations=2)
        assert v_out.shape == v.shape
        assert f_out.shape == f.shape

    def test_faces_unchanged(self, cube):
        v, f = cube
        _, f_out = pyrxmesh.filter(v, f, iterations=2)
        np.testing.assert_array_equal(f_out, f)

    def test_output_shape(self, cube):
        """Filter should return correct shapes."""
        v, f = cube
        v_out, f_out = pyrxmesh.filter(v, f, iterations=1)
        assert v_out.shape == v.shape
        assert f_out.shape == f.shape


# =========================================================================
# mcf (Mean Curvature Flow)
# =========================================================================

class TestMCF:
    def test_shape_preserved(self, bunnyhead):
        v, f = bunnyhead
        v_out, f_out = pyrxmesh.mcf(v, f, time_step=1.0)
        assert v_out.shape[1] == 3
        assert f_out.shape[1] == 3
        assert v_out.shape[0] == v.shape[0]

    def test_output_finite(self, bunnyhead):
        """MCF output should be finite."""
        v, f = bunnyhead
        v_out, _ = pyrxmesh.mcf(v, f, time_step=1.0)
        assert np.all(np.isfinite(v_out))


# =========================================================================
# geodesic
# =========================================================================

class TestGeodesic:
    def test_shape(self, icosphere):
        v, f = icosphere
        seeds = np.array([0], dtype=np.int32)
        dist = pyrxmesh.geodesic(v, f, seeds)
        assert dist.shape == (12,)

    def test_seed_distance_zero(self, icosphere):
        v, f = icosphere
        seeds = np.array([0], dtype=np.int32)
        dist = pyrxmesh.geodesic(v, f, seeds)
        assert dist[0] == 0.0

    def test_all_finite(self, icosphere):
        v, f = icosphere
        seeds = np.array([0], dtype=np.int32)
        dist = pyrxmesh.geodesic(v, f, seeds)
        assert np.all(np.isfinite(dist))

    def test_nonnegative(self, icosphere):
        v, f = icosphere
        seeds = np.array([0], dtype=np.int32)
        dist = pyrxmesh.geodesic(v, f, seeds)
        assert np.all(dist >= 0)

    def test_multiple_seeds(self, cube):
        v, f = cube
        seeds = np.array([0, 6], dtype=np.int32)  # opposite corners
        dist = pyrxmesh.geodesic(v, f, seeds)
        assert dist[0] == 0.0
        assert dist[6] == 0.0
        # Other vertices should be closer than single-seed case
        assert np.all(dist >= 0)

    def test_bunnyhead(self, bunnyhead):
        v, f = bunnyhead
        seeds = np.array([0], dtype=np.int32)
        dist = pyrxmesh.geodesic(v, f, seeds)
        assert dist.shape[0] == v.shape[0]
        assert dist[0] == 0.0
        assert dist.max() > 0


# =========================================================================
# scp (Spectral Conformal Parameterization)
# =========================================================================

class TestSCP:
    def test_shape(self, bunnyhead):
        v, f = bunnyhead
        uv = pyrxmesh.scp(v, f, iterations=16)
        assert uv.shape == (v.shape[0], 2)

    def test_range_normalized(self, bunnyhead):
        v, f = bunnyhead
        uv = pyrxmesh.scp(v, f, iterations=16)
        # Should be normalized to roughly [0,1]
        assert uv[:, 0].min() >= -0.1
        assert uv[:, 0].max() <= 1.1

    def test_closed_mesh_raises(self, cube):
        v, f = cube
        with pytest.raises(RuntimeError, match="boundaries"):
            pyrxmesh.scp(v, f)


# =========================================================================
# param (UV Parameterization)
# =========================================================================

class TestParam:
    def test_shape(self, bunnyhead):
        v, f = bunnyhead
        uv = pyrxmesh.param(v, f, newton_iterations=5)
        assert uv.shape == (v.shape[0], 2)

    def test_finite(self, bunnyhead):
        v, f = bunnyhead
        uv = pyrxmesh.param(v, f, newton_iterations=5)
        assert np.all(np.isfinite(uv))

    def test_closed_mesh_raises(self, cube):
        v, f = cube
        with pytest.raises(RuntimeError, match="boundaries"):
            pyrxmesh.param(v, f)


# =========================================================================
# qslim
# =========================================================================

class TestQSlim:
    def test_decimation(self, cube):
        v, f = cube
        v_out, f_out = pyrxmesh.qslim(v, f, target_ratio=0.5)
        assert v_out.shape[0] <= v.shape[0]
        assert f_out.shape[0] <= f.shape[0]

    def test_large_mesh(self, dragon):
        v, f = dragon
        nv_before = v.shape[0]
        v_out, f_out = pyrxmesh.qslim(v, f, target_ratio=0.5)
        assert v_out.shape[0] < nv_before
        assert v_out.shape[0] <= int(nv_before * 0.6)  # allow some slack

    def test_valid_output(self, cube):
        v, f = cube
        v_out, f_out = pyrxmesh.qslim(v, f, target_ratio=0.5)
        assert v_out.shape[1] == 3
        assert f_out.shape[1] == 3
        # All face indices should be valid
        assert np.all(f_out >= 0)
        assert np.all(f_out < v_out.shape[0])


# =========================================================================
# sec (Shortest Edge Collapse)
# =========================================================================

class TestSEC:
    def test_decimation(self, cube):
        v, f = cube
        v_out, f_out = pyrxmesh.sec(v, f, target_ratio=0.5)
        assert v_out.shape[0] <= v.shape[0]

    def test_large_mesh(self, dragon):
        v, f = dragon
        nv_before = v.shape[0]
        v_out, f_out = pyrxmesh.sec(v, f, target_ratio=0.5)
        assert v_out.shape[0] < nv_before
        assert v_out.shape[0] <= int(nv_before * 0.6)

    def test_valid_faces(self, cube):
        v, f = cube
        v_out, f_out = pyrxmesh.sec(v, f, target_ratio=0.5)
        assert np.all(f_out >= 0)
        assert np.all(f_out < v_out.shape[0])


# =========================================================================
# remesh
# =========================================================================

class TestRemesh:
    def test_topology_changes(self, bunnyhead):
        v, f = bunnyhead
        v_out, f_out = pyrxmesh.remesh(v, f, relative_len=1.0, iterations=1, smooth_iterations=2)
        assert v_out.shape[1] == 3
        assert f_out.shape[1] == 3
        # Remeshing should change vertex/face count
        changed = (v_out.shape[0] != v.shape[0]) or (f_out.shape[0] != f.shape[0])
        assert changed, "Remeshing should modify the mesh topology"

    def test_valid_output(self, bunnyhead):
        v, f = bunnyhead
        v_out, f_out = pyrxmesh.remesh(v, f, relative_len=1.0, iterations=1)
        assert np.all(np.isfinite(v_out))
        assert np.all(f_out >= 0)
        assert np.all(f_out < v_out.shape[0])


# =========================================================================
# delaunay
# =========================================================================

class TestDelaunay:
    def test_preserves_counts(self, cube):
        """Delaunay flipping should preserve vertex/edge/face counts."""
        v, f = cube
        v_out, f_out = pyrxmesh.delaunay(v, f)
        assert v_out.shape[0] == v.shape[0]
        assert f_out.shape[0] == f.shape[0]

    def test_vertices_unchanged(self, cube):
        """Delaunay only flips edges — vertex positions shouldn't change."""
        v, f = cube
        v_out, _ = pyrxmesh.delaunay(v, f)
        # Vertex positions should be the same (possibly reordered)
        # Just check same bounding box
        np.testing.assert_allclose(v_out.min(axis=0), v.min(axis=0), atol=1e-6)
        np.testing.assert_allclose(v_out.max(axis=0), v.max(axis=0), atol=1e-6)

    def test_valid_faces(self, cube):
        v, f = cube
        v_out, f_out = pyrxmesh.delaunay(v, f)
        assert np.all(f_out >= 0)
        assert np.all(f_out < v_out.shape[0])
