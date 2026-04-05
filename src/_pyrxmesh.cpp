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
#include "penner/penner_types.h"

// Forward declarations — defined in op_penner.cu
PennerResult pipeline_penner_conformal(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    const PennerConformalParams& params);

PennerFullResult pipeline_penner_with_targets(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    const double* target_cone_angles,
    const PennerConformalParams& params);

// NOTE: penner_real (direct library linkage) disabled due to ABI issues.
// The pipeline uses subprocess calls to parameterize_aligned instead.

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
    const NDArray<const int, 2> faces,
    bool verbose)
{
    validate_mesh(vertices, faces);
    AttributeResult result = pipeline_vertex_normals(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)), verbose);

    NDArray<double, 2> arr = MakeNDArray<double, 2>({result.num_elements, result.num_cols});
    std::memcpy(arr.data(), result.data.data(),
        result.num_elements * result.num_cols * sizeof(double));
    return nb::cast(arr);
}

static nb::tuple py_smooth(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int iterations, double lambda, bool verbose)
{
    validate_mesh(vertices, faces);
    if (iterations < 0) throw std::runtime_error("iterations must be non-negative");
    return mesh_result_to_tuple(pipeline_smooth(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        iterations, lambda, verbose));
}

static nb::object py_gaussian_curvature(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    bool verbose)
{
    validate_mesh(vertices, faces);
    AttributeResult result = pipeline_gaussian_curvature(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)), verbose);

    NDArray<double, 1> arr = MakeNDArray<double, 1>({result.num_elements});
    std::memcpy(arr.data(), result.data.data(), result.num_elements * sizeof(double));
    return nb::cast(arr);
}

static nb::tuple py_filter(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int iterations, bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_filter(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        iterations, verbose));
}

static nb::tuple py_mcf(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double time_step, bool use_uniform_laplace, bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_mcf(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        time_step, use_uniform_laplace, verbose));
}

static nb::tuple py_qslim(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double target_ratio, bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_qslim(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        target_ratio, verbose));
}

static nb::tuple py_remesh(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double relative_len, int iterations, int smooth_iterations, bool verbose,
    float capacity_factor)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_remesh(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        relative_len, iterations, smooth_iterations, verbose,
        capacity_factor));
}

static nb::tuple py_sec(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double target_ratio, bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_sec(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        target_ratio, verbose));
}

static nb::object py_geodesic(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    const NDArray<const int, 1> seeds,
    bool verbose)
{
    validate_mesh(vertices, faces);
    if (seeds.shape(0) == 0)
        throw std::runtime_error("At least one seed vertex required");

    AttributeResult result = pipeline_geodesic(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        seeds.data(), static_cast<int>(seeds.shape(0)), verbose);

    NDArray<double, 1> arr = MakeNDArray<double, 1>({result.num_elements});
    std::memcpy(arr.data(), result.data.data(), result.num_elements * sizeof(double));
    return nb::cast(arr);
}

static nb::object py_scp(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int iterations, bool verbose)
{
    validate_mesh(vertices, faces);
    AttributeResult result = pipeline_scp(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        iterations, verbose);

    NDArray<double, 2> arr = MakeNDArray<double, 2>({result.num_elements, 2});
    std::memcpy(arr.data(), result.data.data(), result.num_elements * 2 * sizeof(double));
    return nb::cast(arr);
}

static nb::object py_param(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int newton_iterations, bool verbose)
{
    validate_mesh(vertices, faces);
    AttributeResult result = pipeline_param(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        newton_iterations, verbose);

    NDArray<double, 2> arr = MakeNDArray<double, 2>({result.num_elements, 2});
    std::memcpy(arr.data(), result.data.data(), result.num_elements * 2 * sizeof(double));
    return nb::cast(arr);
}

static nb::tuple py_delaunay(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_delaunay(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)), verbose));
}

// --- GPU feature detection ---

static nb::tuple py_detect_features(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    float crease_angle_deg, int erode_dilate_steps, bool verbose)
{
    validate_mesh(vertices, faces);
    FeatureResult fr = pipeline_detect_features(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        crease_angle_deg, erode_dilate_steps, verbose);

    int ne = fr.num_edges;
    int nv = static_cast<int>(vertices.shape(0));

    NDArray<int, 1> ef = MakeNDArray<int, 1>({ne});
    std::memcpy(ef.data(), fr.edge_is_feature.data(), ne * sizeof(int));

    NDArray<int, 1> vf = MakeNDArray<int, 1>({nv});
    std::memcpy(vf.data(), fr.vertex_is_feature.data(), nv * sizeof(int));

    NDArray<int, 1> vb = MakeNDArray<int, 1>({nv});
    std::memcpy(vb.data(), fr.vertex_is_boundary.data(), nv * sizeof(int));

    return nb::make_tuple(ef, vf, vb, fr.num_feature_edges);
}

// --- GPU ExpectedEdgeL ---

static nb::tuple py_expected_edge_length(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int min_faces, bool verbose)
{
    validate_mesh(vertices, faces);
    EdgeLengthResult r = pipeline_expected_edge_length(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        min_faces, verbose);
    return nb::make_tuple(r.area, r.volume, r.sphericity,
                          r.target_edge_length, r.avg_edge_length);
}

// --- VCG CPU remesh ---

static nb::tuple py_vcg_remesh(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    float target_edge_length, int target_faces,
    int iterations, bool adaptive, bool project,
    float crease_angle_deg, bool verbose)
{
    validate_mesh(vertices, faces);
    VcgRemeshParams p;
    p.target_edge_length = target_edge_length;
    p.target_faces       = target_faces;
    p.iterations         = iterations;
    p.adaptive           = adaptive;
    p.project            = project;
    p.crease_angle_deg   = crease_angle_deg;

    VcgRemeshResult r = vcg_remesh(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        p, verbose);

    NDArray<double, 2> verts_arr = MakeNDArray<double, 2>({r.num_vertices, 3});
    std::memcpy(verts_arr.data(), r.vertices.data(), r.num_vertices * 3 * sizeof(double));

    NDArray<int, 2> faces_arr = MakeNDArray<int, 2>({r.num_faces, 3});
    std::memcpy(faces_arr.data(), r.faces.data(), r.num_faces * 3 * sizeof(int));

    return nb::make_tuple(verts_arr, faces_arr);
}

// --- VCG remesh with checkpoints ---

static nb::tuple mesh_result_to_np(const VcgRemeshResult& r) {
    NDArray<double, 2> v = MakeNDArray<double, 2>({r.num_vertices, 3});
    std::memcpy(v.data(), r.vertices.data(), r.num_vertices * 3 * sizeof(double));
    NDArray<int, 2> f = MakeNDArray<int, 2>({r.num_faces, 3});
    std::memcpy(f.data(), r.faces.data(), r.num_faces * 3 * sizeof(int));
    return nb::make_tuple(v, f);
}

static nb::dict py_vcg_remesh_checkpoints(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    float target_edge_length, int target_faces,
    int iterations, bool adaptive, bool project,
    float crease_angle_deg, bool verbose)
{
    validate_mesh(vertices, faces);
    VcgRemeshParams p;
    p.target_edge_length = target_edge_length;
    p.target_faces       = target_faces;
    p.iterations         = iterations;
    p.adaptive           = adaptive;
    p.project            = project;
    p.crease_angle_deg   = crease_angle_deg;

    VcgRemeshCheckpoints ck = vcg_remesh_with_checkpoints(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        p, verbose);

    nb::dict d;
    d["after_pass1"] = mesh_result_to_np(ck.after_pass1);
    d["after_micro_collapse"] = mesh_result_to_np(ck.after_micro_collapse);
    if (ck.has_pass2)
        d["after_pass2"] = mesh_result_to_np(ck.after_pass2);
    d["after_cleanup"] = mesh_result_to_np(ck.after_cleanup);
    d["after_refine"] = mesh_result_to_np(ck.after_refine);
    return d;
}

// --- Standalone micro-edge collapse ---

static nb::tuple py_vcg_micro_collapse(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    float quality_thr, int max_iter, bool verbose)
{
    validate_mesh(vertices, faces);
    VcgRemeshResult r = vcg_micro_collapse(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        quality_thr, max_iter, verbose);

    NDArray<double, 2> v = MakeNDArray<double, 2>({r.num_vertices, 3});
    std::memcpy(v.data(), r.vertices.data(), r.num_vertices * 3 * sizeof(double));
    NDArray<int, 2> f = MakeNDArray<int, 2>({r.num_faces, 3});
    std::memcpy(f.data(), r.faces.data(), r.num_faces * 3 * sizeof(int));
    return nb::make_tuple(v, f);
}

static nb::tuple py_vcg_remesh_adaptive(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    float target_edge_length, int target_faces, int iterations,
    float crease_angle_deg, bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_np(vcg_remesh_adaptive(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        target_edge_length, target_faces, iterations,
        crease_angle_deg, verbose));
}

static nb::tuple py_vcg_clean_mesh(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_np(vcg_clean_mesh(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        verbose));
}

static nb::tuple py_vcg_refine_if_needed(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    float crease_angle_deg, bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_np(vcg_refine_if_needed(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        crease_angle_deg, verbose));
}

static nb::dict py_quadwild_remesh(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double relative_len, int isotropic_iterations, int adaptive_iterations,
    int smooth_iterations, float crease_angle_deg, float micro_quality_thr,
    bool verbose)
{
    validate_mesh(vertices, faces);
    auto r = pipeline_quadwild_remesh(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        relative_len, isotropic_iterations, adaptive_iterations,
        smooth_iterations, crease_angle_deg, micro_quality_thr, verbose);

    nb::dict d;
    d["after_isotropic"] = mesh_result_to_tuple(r.after_isotropic);
    d["after_micro"] = mesh_result_to_tuple(r.after_micro);
    d["final"] = mesh_result_to_tuple(r.final_mesh);
    return d;
}

// --- Feature-aware GPU remesh ---

static nb::tuple py_feature_remesh(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double relative_len, int iterations, int smooth_iterations,
    float crease_angle_deg, float flip_normal_thr, bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_feature_remesh(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        relative_len, iterations, smooth_iterations,
        crease_angle_deg, flip_normal_thr, verbose));
}

// --- QuadWild preprocessing ---

static nb::tuple py_quadwild_preprocess(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    float target_edge_length, int target_faces,
    int num_iterations, int num_smooth_iters, bool verbose)
{
    validate_mesh(vertices, faces);
    int nv = static_cast<int>(vertices.shape(0));
    int nf = static_cast<int>(faces.shape(0));

    // Convert double vertices to float
    std::vector<float> vf(nv * 3);
    for (int i = 0; i < nv * 3; ++i)
        vf[i] = static_cast<float>(vertices.data()[i]);

    QuadwildParams qp;
    qp.target_edge_length = target_edge_length;
    qp.target_faces       = target_faces;
    qp.num_iterations     = num_iterations;
    qp.num_smooth_iters   = num_smooth_iters;

    return mesh_result_to_tuple(pipeline_quadwild_preprocess(
        vf.data(), nv, faces.data(), nf, qp, verbose));
}

// --- Patch visualization ---

static nb::tuple py_patch_info(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces)
{
    validate_mesh(vertices, faces);
    PatchResult pr = pipeline_patch_info(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)));

    int nv = static_cast<int>(vertices.shape(0));
    int nf = static_cast<int>(faces.shape(0));

    NDArray<int, 1> v_patch = MakeNDArray<int, 1>({nv});
    std::memcpy(v_patch.data(), pr.vertex_patch_ids.data(), nv * sizeof(int));

    NDArray<int, 1> f_patch = MakeNDArray<int, 1>({nf});
    std::memcpy(f_patch.data(), pr.face_patch_ids.data(), nf * sizeof(int));

    NDArray<int, 1> v_ribbon = MakeNDArray<int, 1>({nv});
    std::memcpy(v_ribbon.data(), pr.vertex_is_ribbon.data(), nv * sizeof(int));

    NDArray<int, 1> f_ribbon = MakeNDArray<int, 1>({nf});
    std::memcpy(f_ribbon.data(), pr.face_is_ribbon.data(), nf * sizeof(int));

    return nb::make_tuple(v_patch, f_patch, v_ribbon, f_ribbon, pr.num_patches);
}

// --- Standalone edge operations ---

static nb::tuple py_edge_split(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double relative_len, int iterations, bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_edge_split(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        relative_len, iterations, verbose));
}

static nb::tuple py_edge_collapse(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    double relative_len, int iterations, bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_edge_collapse(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        relative_len, iterations, verbose));
}

static nb::tuple py_edge_flip(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int iterations, bool verbose)
{
    validate_mesh(vertices, faces);
    return mesh_result_to_tuple(pipeline_edge_flip(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        iterations, verbose));
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
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("verbose") = false);

    m.def("smooth", &py_smooth, "Laplacian mesh smoothing on GPU.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("iterations") = 10, nb::arg("lambda_") = 0.5,
        nb::arg("verbose") = false);

    m.def("gaussian_curvature", &py_gaussian_curvature,
        "Compute discrete Gaussian curvature per vertex (Meyer et al. 2003).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("verbose") = false);

    m.def("filter", &py_filter,
        "Bilateral mesh denoising on GPU (Fleishman et al. 2003).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("iterations") = 5,
        nb::arg("verbose") = false);

    m.def("mcf", &py_mcf,
        "Mean Curvature Flow smoothing via Cholesky solver.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("time_step") = 10.0, nb::arg("use_uniform_laplace") = true,
        nb::arg("verbose") = false);

    m.def("qslim", &py_qslim,
        "QSlim mesh decimation (edge collapse).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("target_ratio") = 0.5,
        nb::arg("verbose") = false);

    m.def("remesh", &py_remesh,
        "Isotropic remeshing (split/collapse/flip/smooth).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("relative_len") = 1.0,
        nb::arg("iterations") = 3, nb::arg("smooth_iterations") = 5,
        nb::arg("verbose") = false,
        nb::arg("capacity_factor") = 2.0f);

    m.def("sec", &py_sec,
        "Shortest-edge-collapse decimation.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("target_ratio") = 0.5,
        nb::arg("verbose") = false);

    m.def("geodesic", &py_geodesic,
        "Compute geodesic distances from seed vertices on GPU.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("seeds"),
        nb::arg("verbose") = false);

    m.def("scp", &py_scp,
        "Spectral Conformal Parameterization (UV coords via power method).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("iterations") = 32,
        nb::arg("verbose") = false);

    m.def("param", &py_param,
        "UV Parameterization via Tutte + symmetric Dirichlet energy.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("newton_iterations") = 100,
        nb::arg("verbose") = false);

    m.def("delaunay", &py_delaunay,
        "Delaunay edge flipping (maximize minimum angles).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("verbose") = false);

    m.def("detect_features", &py_detect_features,
        "Detect feature edges on GPU via dihedral angle threshold.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("crease_angle_deg") = 35.0f,
        nb::arg("erode_dilate_steps") = 4,
        nb::arg("verbose") = false);

    m.def("expected_edge_length", &py_expected_edge_length,
        "Compute QuadWild's ExpectedEdgeL on GPU (sphericity-based target).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("min_faces") = 10000,
        nb::arg("verbose") = false);

    m.def("vcg_remesh", &py_vcg_remesh,
        "CPU isotropic remeshing via VCG (same as QuadWild's AutoRemesher).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("target_edge_length") = 0.0f,
        nb::arg("target_faces") = 10000,
        nb::arg("iterations") = 3,
        nb::arg("adaptive") = true,
        nb::arg("project") = true,
        nb::arg("crease_angle_deg") = 35.0f,
        nb::arg("verbose") = false);

    m.def("vcg_remesh_checkpoints", &py_vcg_remesh_checkpoints,
        "CPU remeshing with intermediate checkpoints at each pipeline stage.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("target_edge_length") = 0.0f,
        nb::arg("target_faces") = 10000,
        nb::arg("iterations") = 3,
        nb::arg("adaptive") = true,
        nb::arg("project") = true,
        nb::arg("crease_angle_deg") = 35.0f,
        nb::arg("verbose") = false);

    m.def("vcg_micro_collapse", &py_vcg_micro_collapse,
        "Collapse micro-edges in degenerate triangles (CPU).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("quality_thr") = 0.01f,
        nb::arg("max_iter") = 2,
        nb::arg("verbose") = false);

    m.def("vcg_remesh_adaptive", &py_vcg_remesh_adaptive,
        "Adaptive-only remesh pass (pass 2 only, no isotropic pass 1).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("target_edge_length") = 0.0f,
        nb::arg("target_faces") = 10000,
        nb::arg("iterations") = 15,
        nb::arg("crease_angle_deg") = 35.0f,
        nb::arg("verbose") = false);

    m.def("vcg_clean_mesh", &py_vcg_clean_mesh,
        "SolveGeometricArtifacts: remove zero-area faces, non-manifold, small components (CPU).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("verbose") = false);

    m.def("vcg_refine_if_needed", &py_vcg_refine_if_needed,
        "RefineIfNeeded: split faces with 3 sharp edges at centroid (CPU).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("crease_angle_deg") = 35.0f,
        nb::arg("verbose") = false);

    m.def("quadwild_remesh", &py_quadwild_remesh,
        "Full QuadWild GPU pipeline: isotropic → micro-collapse → re-detect → adaptive.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("relative_len") = 1.0,
        nb::arg("isotropic_iterations") = 15,
        nb::arg("adaptive_iterations") = 0,
        nb::arg("smooth_iterations") = 5,
        nb::arg("crease_angle_deg") = 35.0f,
        nb::arg("micro_quality_thr") = 0.01f,
        nb::arg("verbose") = false);

    m.def("feature_remesh", &py_feature_remesh,
        "Feature-aware GPU remeshing (split/collapse/flip/smooth/project).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("relative_len") = 1.0,
        nb::arg("iterations") = 15,
        nb::arg("smooth_iterations") = 5,
        nb::arg("crease_angle_deg") = 35.0f,
        nb::arg("flip_normal_thr") = 0.996f,
        nb::arg("verbose") = false);

    m.def("quadwild_preprocess", &py_quadwild_preprocess,
        "GPU isotropic remeshing for QuadWild preprocessing.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("target_edge_length") = 0.0f,
        nb::arg("target_faces") = 10000,
        nb::arg("num_iterations") = 15,
        nb::arg("num_smooth_iters") = 5,
        nb::arg("verbose") = false);

    m.def("patch_info", &py_patch_info,
        "Get per-vertex/face patch IDs and ribbon masks for visualization.",
        nb::arg("vertices"), nb::arg("faces"));

    // ── Standalone edge operations ──
    m.def("edge_split", &py_edge_split,
        "Split long edges (from isotropic remeshing sub-op).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("relative_len") = 1.0, nb::arg("iterations") = 1,
        nb::arg("verbose") = false);

    m.def("edge_collapse", &py_edge_collapse,
        "Collapse short edges (from isotropic remeshing sub-op).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("relative_len") = 1.0, nb::arg("iterations") = 1,
        nb::arg("verbose") = false);

    m.def("edge_flip", &py_edge_flip,
        "Flip edges to equalize vertex valences (target = 6).",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("iterations") = 1,
        nb::arg("verbose") = false);

    // ── Penner conformal optimization ──
    m.def("penner_conformal",
        [](const NDArray<const double, 2> vertices,
           const NDArray<const int, 2> faces,
           double error_eps, int max_iterations,
           double min_angle_deg, bool verbose) -> nb::dict {
            validate_mesh(vertices, faces);
            PennerConformalParams params;
            params.error_eps = error_eps;
            params.max_iterations = max_iterations;
            params.min_angle_deg = min_angle_deg;
            params.verbose = verbose;
            auto r = pipeline_penner_conformal(
                vertices.data(), static_cast<int>(vertices.shape(0)),
                faces.data(), static_cast<int>(faces.shape(0)),
                params);
            nb::dict d;
            d["newton_iterations"] = r.newton_iterations;
            d["final_error"] = r.final_error;
            d["total_time_ms"] = r.total_time_ms;
            d["num_vertices"] = r.num_vertices;
            d["num_edges"] = r.num_edges;
            // Log-lengths as numpy array
            NDArray<double, 1> ll = MakeNDArray<double, 1>({(size_t)r.num_edges});
            std::memcpy(ll.data(), r.log_lengths.data(), r.num_edges * sizeof(double));
            d["log_lengths"] = ll;
            return d;
        },
        "Penner conformal metric optimization (GPU). Returns optimized log edge lengths.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("error_eps") = 1e-10,
        nb::arg("max_iterations") = 100,
        nb::arg("min_angle_deg") = 25.0,
        nb::arg("verbose") = false);

    // ── Penner with external target cone angles ──
    m.def("penner_with_targets",
        [](const NDArray<const double, 2> vertices,
           const NDArray<const int, 2> faces,
           const NDArray<const double, 1> target_cone_angles,
           double error_eps, int max_iterations,
           double min_angle_deg, bool verbose,
           const std::string& debug_dir) -> nb::dict {
            validate_mesh(vertices, faces);
            int nV = static_cast<int>(vertices.shape(0));
            if ((int)target_cone_angles.shape(0) != nV)
                throw std::runtime_error("target_cone_angles size must match vertex count");
            PennerConformalParams params;
            params.error_eps = error_eps;
            params.max_iterations = max_iterations;
            params.min_angle_deg = min_angle_deg;
            params.verbose = verbose;
            params.debug_dir = debug_dir;
            auto r = pipeline_penner_with_targets(
                vertices.data(), nV,
                faces.data(), static_cast<int>(faces.shape(0)),
                target_cone_angles.data(), params);
            nb::dict d;
            d["newton_iterations"] = r.newton_iterations;
            d["final_error"] = r.final_error;
            d["total_time_ms"] = r.total_time_ms;
            // Export halfedge mesh connectivity for layout
            int nHE = r.mesh.num_halfedges;
            int nF = r.mesh.num_faces;
            NDArray<double, 1> he_len = MakeNDArray<double, 1>({(size_t)nHE});
            std::memcpy(he_len.data(), r.he_length.data(), nHE * sizeof(double));
            d["he_length"] = he_len;
            NDArray<double, 1> scale = MakeNDArray<double, 1>({(size_t)nV});
            std::memcpy(scale.data(), r.scale_factors.data(), nV * sizeof(double));
            d["scale_factors"] = scale;
            // Mesh connectivity
            NDArray<int, 1> he_next = MakeNDArray<int, 1>({(size_t)nHE});
            std::memcpy(he_next.data(), r.mesh.next.data(), nHE * sizeof(int));
            d["he_next"] = he_next;
            NDArray<int, 1> he_to = MakeNDArray<int, 1>({(size_t)nHE});
            std::memcpy(he_to.data(), r.mesh.to.data(), nHE * sizeof(int));
            d["he_to"] = he_to;
            NDArray<int, 1> he_twin = MakeNDArray<int, 1>({(size_t)nHE});
            std::memcpy(he_twin.data(), r.mesh.twin.data(), nHE * sizeof(int));
            d["he_twin"] = he_twin;
            NDArray<int, 1> fhe = MakeNDArray<int, 1>({(size_t)nF});
            std::memcpy(fhe.data(), r.mesh.fhe.data(), nF * sizeof(int));
            d["fhe"] = fhe;
            d["num_vertices"] = r.mesh.num_vertices;
            d["num_halfedges"] = nHE;
            d["num_faces"] = nF;
            return d;
        },
        "Penner optimization with external target cone angles.\n"
        "Returns optimized halfedge mesh + edge lengths for layout.",
        nb::arg("vertices"), nb::arg("faces"),
        nb::arg("target_cone_angles"),
        nb::arg("error_eps") = 1e-10,
        nb::arg("max_iterations") = 100,
        nb::arg("min_angle_deg") = 25.0,
        nb::arg("verbose") = false,
        nb::arg("debug_dir") = "");

    // NOTE: penner_real binding removed — ABI issues with linked Penner library.
    // Use subprocess calls to parameterize_aligned binary instead.

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
