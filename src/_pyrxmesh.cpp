// Python bindings for RXMesh GPU mesh processing via nanobind.
//
// This file ONLY includes nanobind and the pipeline wrapper header.
// All RXMesh/CUDA headers are isolated in pipeline.cu.

#include <cstring>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "array_support.h"
#include "pipeline.h"

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void validate_mesh(
    const NDArray<const double, 2>& vertices,
    const NDArray<const int, 2>& faces)
{
    if (vertices.shape(1) != 3)
        throw std::runtime_error("vertices must have shape (N, 3)");
    if (faces.shape(1) != 3)
        throw std::runtime_error("faces must have shape (M, 3)");
    if (vertices.shape(0) == 0 || faces.shape(0) == 0)
        throw std::runtime_error("Input mesh is empty");
}

static nb::tuple mesh_result_to_tuple(const MeshResult& result) {
    NDArray<double, 2> verts_arr = MakeNDArray<double, 2>({result.num_vertices, 3});
    std::memcpy(verts_arr.data(), result.vertices.data(),
        result.num_vertices * 3 * sizeof(double));

    NDArray<int, 2> faces_arr = MakeNDArray<int, 2>({result.num_faces, 3});
    std::memcpy(faces_arr.data(), result.faces.data(),
        result.num_faces * 3 * sizeof(int));

    return nb::make_tuple(verts_arr, faces_arr);
}

// ---------------------------------------------------------------------------
// Binding functions
// ---------------------------------------------------------------------------

static void py_init(int device_id) { pipeline_init(device_id); }

static nb::tuple py_mesh_info(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces)
{
    validate_mesh(vertices, faces);
    MeshInfo info = pipeline_mesh_info(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)));
    return nb::make_tuple(
        info.num_vertices, info.num_edges, info.num_faces,
        info.is_edge_manifold, info.is_closed,
        info.max_valence, info.num_components);
}

static nb::tuple py_load_obj(const std::string& path) {
    return mesh_result_to_tuple(pipeline_load_obj(path));
}

static nb::object py_vertex_normals(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces)
{
    validate_mesh(vertices, faces);
    AttributeResult result = pipeline_vertex_normals(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)));

    NDArray<double, 2> arr = MakeNDArray<double, 2>({result.num_elements, result.num_cols});
    std::memcpy(arr.data(), result.data.data(),
        result.num_elements * result.num_cols * sizeof(double));
    return nb::cast(arr);
}

static nb::tuple py_smooth(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int iterations, double lambda)
{
    validate_mesh(vertices, faces);
    if (iterations < 0) throw std::runtime_error("iterations must be non-negative");
    return mesh_result_to_tuple(pipeline_smooth(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        iterations, lambda));
}

static nb::object py_gaussian_curvature(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces)
{
    validate_mesh(vertices, faces);
    AttributeResult result = pipeline_gaussian_curvature(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)));

    NDArray<double, 1> arr = MakeNDArray<double, 1>({result.num_elements});
    std::memcpy(arr.data(), result.data.data(), result.num_elements * sizeof(double));
    return nb::cast(arr);
}

static nb::tuple py_filter(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int iterations)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_filter(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        iterations));
}

static nb::tuple py_mcf(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double time_step, bool use_uniform_laplace)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_mcf(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        time_step, use_uniform_laplace));
}

static nb::tuple py_qslim(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double target_ratio)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_qslim(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        target_ratio));
}

static nb::tuple py_remesh(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double relative_len, int iterations, int smooth_iterations)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_remesh(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        relative_len, iterations, smooth_iterations));
}

static nb::tuple py_sec(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double target_ratio)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_sec(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        target_ratio));
}

static nb::object py_geodesic(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    const NDArray<const int, 1> seeds)
{
    validate_mesh(vertices, faces);
    if (seeds.shape(0) == 0)
        throw std::runtime_error("At least one seed vertex required");

    AttributeResult result = pipeline_geodesic(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        seeds.data(), static_cast<int>(seeds.shape(0)));

    NDArray<double, 1> arr = MakeNDArray<double, 1>({result.num_elements});
    std::memcpy(arr.data(), result.data.data(), result.num_elements * sizeof(double));
    return nb::cast(arr);
}

static nb::object py_scp(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int iterations)
{
    validate_mesh(vertices, faces);
    AttributeResult result = pipeline_scp(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        iterations);

    NDArray<double, 2> arr = MakeNDArray<double, 2>({result.num_elements, 2});
    std::memcpy(arr.data(), result.data.data(), result.num_elements * 2 * sizeof(double));
    return nb::cast(arr);
}

static nb::object py_param(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int newton_iterations)
{
    validate_mesh(vertices, faces);
    AttributeResult result = pipeline_param(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        newton_iterations);

    NDArray<double, 2> arr = MakeNDArray<double, 2>({result.num_elements, 2});
    std::memcpy(arr.data(), result.data.data(), result.num_elements * 2 * sizeof(double));
    return nb::cast(arr);
}

static nb::tuple py_delaunay(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_delaunay(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0))));
}

// ---------------------------------------------------------------------------
// Persistent Mesh class
// ---------------------------------------------------------------------------

class PyMesh {
    MeshHandle m_handle;
public:
    PyMesh(const NDArray<const double, 2> vertices,
           const NDArray<const int, 2> faces) {
        validate_mesh(vertices, faces);
        m_handle = mesh_create(
            vertices.data(), static_cast<int>(vertices.shape(0)),
            faces.data(), static_cast<int>(faces.shape(0)));
    }

    PyMesh(const std::string& path) {
        m_handle = mesh_create_from_obj(path);
    }

    ~PyMesh() {
        if (m_handle) mesh_destroy(m_handle);
    }

    // Non-copyable
    PyMesh(const PyMesh&) = delete;
    PyMesh& operator=(const PyMesh&) = delete;

    nb::tuple info() {
        MeshInfo i = mesh_get_info(m_handle);
        return nb::make_tuple(i.num_vertices, i.num_edges, i.num_faces,
            i.is_edge_manifold, i.is_closed, i.max_valence, i.num_components);
    }

    int num_vertices() { return mesh_get_num_vertices(m_handle); }
    int num_faces() { return mesh_get_num_faces(m_handle); }

    nb::object vertices() {
        int nv = mesh_get_num_vertices(m_handle);
        NDArray<double, 2> arr = MakeNDArray<double, 2>({nv, 3});
        mesh_get_vertices(m_handle, arr.data());
        return nb::cast(arr);
    }

    nb::object vertex_normals() {
        auto r = mesh_vertex_normals(m_handle);
        NDArray<double, 2> arr = MakeNDArray<double, 2>({r.num_elements, 3});
        std::memcpy(arr.data(), r.data.data(), r.num_elements * 3 * sizeof(double));
        return nb::cast(arr);
    }

    nb::object gaussian_curvature() {
        auto r = mesh_gaussian_curvature(m_handle);
        NDArray<double, 1> arr = MakeNDArray<double, 1>({r.num_elements});
        std::memcpy(arr.data(), r.data.data(), r.num_elements * sizeof(double));
        return nb::cast(arr);
    }

    nb::object geodesic(const NDArray<const int, 1> seeds) {
        auto r = mesh_geodesic(m_handle, seeds.data(), static_cast<int>(seeds.shape(0)));
        NDArray<double, 1> arr = MakeNDArray<double, 1>({r.num_elements});
        std::memcpy(arr.data(), r.data.data(), r.num_elements * sizeof(double));
        return nb::cast(arr);
    }

    nb::tuple smooth(int iterations, double lambda) {
        auto r = mesh_smooth(m_handle, iterations, lambda);
        return mesh_result_to_tuple(r);
    }

    nb::tuple filter(int iterations) {
        auto r = mesh_filter(m_handle, iterations);
        return mesh_result_to_tuple(r);
    }
};

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

NB_MODULE(_pyrxmesh, m) {
    m.doc() = "Python bindings for RXMesh (GPU-accelerated triangle mesh processing)";

    m.def("init", &py_init, "Initialize CUDA device.",
        nb::arg("device_id") = 0);

    m.def("mesh_info", &py_mesh_info, "Get mesh topology statistics.",
        nb::arg("vertices"), nb::arg("faces"));

    m.def("load_obj", &py_load_obj, "Load triangle mesh from OBJ file.",
        nb::arg("path"));

    m.def("vertex_normals", &py_vertex_normals,
        "Compute area-weighted vertex normals on GPU.",
        nb::arg("vertices"), nb::arg("faces"));

    m.def("smooth", &py_smooth, "Laplacian mesh smoothing on GPU.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("iterations") = 10, nb::arg("lambda_") = 0.5);

    m.def("gaussian_curvature", &py_gaussian_curvature,
        "Compute discrete Gaussian curvature per vertex (Meyer et al. 2003).",
        nb::arg("vertices"), nb::arg("faces"));

    m.def("filter", &py_filter,
        "Bilateral mesh denoising on GPU (Fleishman et al. 2003).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("iterations") = 5);

    m.def("mcf", &py_mcf,
        "Mean Curvature Flow smoothing via Cholesky solver.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("time_step") = 10.0, nb::arg("use_uniform_laplace") = true);

    m.def("qslim", &py_qslim,
        "QSlim mesh decimation (edge collapse).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("target_ratio") = 0.5);

    m.def("remesh", &py_remesh,
        "Isotropic remeshing (split/collapse/flip/smooth).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("relative_len") = 1.0,
        nb::arg("iterations") = 3, nb::arg("smooth_iterations") = 5);

    m.def("sec", &py_sec,
        "Shortest-edge-collapse decimation.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("target_ratio") = 0.5);

    m.def("geodesic", &py_geodesic,
        "Compute geodesic distances from seed vertices on GPU.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("seeds"));

    m.def("scp", &py_scp,
        "Spectral Conformal Parameterization (UV coords via power method).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("iterations") = 32);

    m.def("param", &py_param,
        "UV Parameterization via Tutte + symmetric Dirichlet energy.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("newton_iterations") = 100);

    m.def("delaunay", &py_delaunay,
        "Delaunay edge flipping (maximize minimum angles).",
        nb::arg("vertices"), nb::arg("faces"));

    // ── Persistent Mesh class ──
    nb::class_<PyMesh>(m, "Mesh",
        "Persistent GPU-resident mesh. Construct once, run many operations fast.")
        .def(nb::init<const NDArray<const double, 2>, const NDArray<const int, 2>>(),
            nb::arg("vertices"), nb::arg("faces"))
        .def(nb::init<const std::string&>(), nb::arg("path"))
        .def("info", &PyMesh::info, "Mesh topology statistics.")
        .def_prop_ro("num_vertices", &PyMesh::num_vertices)
        .def_prop_ro("num_faces", &PyMesh::num_faces)
        .def("get_vertices", &PyMesh::vertices, "Get vertex positions as (N,3) array.")
        .def("vertex_normals", &PyMesh::vertex_normals, "Area-weighted vertex normals.")
        .def("gaussian_curvature", &PyMesh::gaussian_curvature, "Discrete Gaussian curvature.")
        .def("geodesic", &PyMesh::geodesic, "Geodesic distances from seeds.",
            nb::arg("seeds"))
        .def("smooth", &PyMesh::smooth, "Laplacian smoothing (modifies internal coords).",
            nb::arg("iterations") = 10, nb::arg("lambda_") = 0.5)
        .def("filter", &PyMesh::filter, "Bilateral denoising (modifies internal coords).",
            nb::arg("iterations") = 5);
}
