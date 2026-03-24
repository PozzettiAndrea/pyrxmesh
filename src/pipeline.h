// Pure C++ wrapper for the RXMesh pipeline.
// This header MUST NOT include any Python/nanobind/CUDA headers to avoid
// compiler conflicts. Only standard C++ types.

#ifndef PYRXMESH_PIPELINE_H
#define PYRXMESH_PIPELINE_H

#include <cstdint>
#include <string>
#include <vector>

// --- Mesh info result ---
struct MeshInfo {
    uint32_t num_vertices;
    uint32_t num_edges;
    uint32_t num_faces;
    bool     is_edge_manifold;
    bool     is_closed;
    uint32_t max_valence;
    uint32_t num_components;
};

// --- Mesh result (vertices + faces as flat arrays) ---
struct MeshResult {
    std::vector<double> vertices;  // flat [x0,y0,z0, x1,y1,z1, ...]
    std::vector<int>    faces;     // flat [v0,v1,v2, ...]
    int num_vertices;
    int num_faces;
};

// --- Attribute result (per-element data) ---
struct AttributeResult {
    std::vector<double> data;  // flat row-major
    int num_elements;
    int num_cols;
};

// Initialize CUDA device. Call before any other pipeline function.
void pipeline_init(int device_id = 0);

// Get mesh statistics from vertex/face arrays.
MeshInfo pipeline_mesh_info(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces);

// Load an OBJ file and return vertices + faces.
MeshResult pipeline_load_obj(const std::string& path);

// Compute area-weighted vertex normals on GPU.
AttributeResult pipeline_vertex_normals(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    bool verbose = false);

// Laplacian smoothing on GPU.
MeshResult pipeline_smooth(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int iterations,
    double lambda,
    bool verbose = false);

// Compute discrete Gaussian curvature per vertex (Meyer et al. 2003).
// Returns per-vertex scalar curvature values.
AttributeResult pipeline_gaussian_curvature(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    bool verbose = false);

// Bilateral mesh filtering/denoising (Fleishman et al. 2003).
MeshResult pipeline_filter(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int iterations,
    bool verbose = false);

// Mean Curvature Flow via Cholesky solver (Desbrun et al. 1999).
MeshResult pipeline_mcf(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double time_step,
    bool use_uniform_laplace,
    bool verbose = false);

// QSlim mesh decimation (Garland & Heckbert).
// target_ratio: fraction of vertices to keep (e.g. 0.1 = keep 10%).
MeshResult pipeline_qslim(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double target_ratio,
    bool verbose = false);

// Isotropic remeshing (split/collapse/flip/smooth).
// relative_len: target edge length as ratio of input average edge length.
MeshResult pipeline_remesh(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double relative_len,
    int iterations,
    int smooth_iterations,
    bool verbose = false);

// Shortest-edge-collapse decimation (batch histogram method).
// target_ratio: fraction of vertices to keep.
MeshResult pipeline_sec(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double target_ratio,
    bool verbose = false);

// Delaunay edge flipping (maximize minimum angles).
MeshResult pipeline_delaunay(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    bool verbose = false);

// Compute geodesic distances from seed vertices on GPU.
// Returns per-vertex scalar distance values.
AttributeResult pipeline_geodesic(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    const int* seed_vertices, int num_seeds,
    bool verbose = false);

// Spectral Conformal Parameterization (power method).
// Returns per-vertex UV coordinates (N, 2).
// Requires mesh with boundaries (not closed).
AttributeResult pipeline_scp(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int iterations,
    bool verbose = false);

// UV Parameterization via Tutte embedding + symmetric Dirichlet energy.
// Returns per-vertex UV coordinates (N, 2).
// Requires mesh with boundaries (not closed).
AttributeResult pipeline_param(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int newton_iterations,
    bool verbose = false);

// =========================================================================
// Standalone edge operations (from Remesh sub-ops)
// =========================================================================

// Split long edges. relative_len controls threshold relative to avg edge length.
MeshResult pipeline_edge_split(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double relative_len,
    int iterations,
    bool verbose = false);

// Collapse short edges. relative_len controls threshold relative to avg edge length.
MeshResult pipeline_edge_collapse(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double relative_len,
    int iterations,
    bool verbose = false);

// Flip edges to equalize vertex valences (target valence = 6).
MeshResult pipeline_edge_flip(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int iterations,
    bool verbose = false);

// =========================================================================
// GPU feature detection — dihedral angle crease edges + boundary
// =========================================================================

struct FeatureResult {
    std::vector<int> edge_is_feature;    // num_edges — 1 if feature, 0 otherwise
    std::vector<int> vertex_is_feature;  // num_vertices — 1 if on a feature edge
    std::vector<int> vertex_is_boundary; // num_vertices — 1 if on mesh boundary
    int num_edges;
    int num_feature_edges;
};

// Detect feature edges on GPU using dihedral angle threshold.
// Marks edges where the angle between adjacent face normals exceeds
// crease_angle_deg, plus boundary and non-manifold edges.
// erode_dilate_steps: QuadWild uses 4. 0 = skip erode/dilate.
FeatureResult pipeline_detect_features(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    float crease_angle_deg = 35.0f,
    int erode_dilate_steps = 4,
    bool verbose = false);

// Compute QuadWild's ExpectedEdgeL on GPU (sphericity-based target edge length).
// Uses GPU reductions for mesh area and signed volume.
struct EdgeLengthResult {
    double area;
    double volume;
    double sphericity;
    double target_edge_length;
    double avg_edge_length;
};

EdgeLengthResult pipeline_expected_edge_length(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int min_faces = 10000,
    bool verbose = false);

// =========================================================================
// Feature-aware GPU remeshing — skips feature edges during split/collapse/flip
// =========================================================================

MeshResult pipeline_feature_remesh(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double relative_len,
    int iterations,
    int smooth_iterations,
    float crease_angle_deg = 35.0f,
    bool verbose = false);

// =========================================================================
// QuadWild preprocessing — GPU remesh with auto target edge length
// =========================================================================

struct QuadwildParams {
    float target_edge_length;   // 0 = auto via ExpectedEdgeL (sphericity-based)
    int   target_faces;         // MinFaces for ExpectedEdgeL (default: 10000)
    int   num_iterations;       // outer split+collapse+flip+smooth loops (default: 15, matching QuadWild)
    int   num_smooth_iters;     // inner smoothing sub-iterations (default: 5)
};

// Remesh a mesh for QuadWild preprocessing.
// Accepts float input (QuadWild's native format), returns MeshResult (double).
MeshResult pipeline_quadwild_preprocess(
    const float* vertices, int num_vertices,
    const int* faces, int num_faces,
    const QuadwildParams& params,
    bool verbose = false);

// =========================================================================
// CPU remeshing via VCG (same algorithm as QuadWild's AutoRemesher)
// =========================================================================

// Forward declaration — implemented in vcg_remesh.cpp (no CUDA)
#include "vcg_remesh.h"

// =========================================================================
// Patch visualization — per-element patch IDs and ribbon masks
// =========================================================================

struct PatchResult {
    std::vector<int> vertex_patch_ids;   // num_vertices — owning patch per vertex
    std::vector<int> face_patch_ids;     // num_faces — owning patch per face
    std::vector<int> vertex_is_ribbon;   // num_vertices — 1 if ribbon (not owned), 0 if owned
    std::vector<int> face_is_ribbon;     // num_faces — 1 if ribbon, 0 if owned
    int num_patches;
};

PatchResult pipeline_patch_info(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces);

// =========================================================================
// Persistent mesh handle — construct once, run many operations fast
// =========================================================================

// Opaque handle to a GPU-resident mesh (RXMeshStatic + coordinates).
typedef struct MeshHandleImpl* MeshHandle;

// Create a persistent mesh on GPU. Returns opaque handle.
MeshHandle mesh_create(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces);

// Create from OBJ file.
MeshHandle mesh_create_from_obj(const std::string& path);

// Destroy and free GPU resources.
void mesh_destroy(MeshHandle h);

// --- Queries on persistent mesh ---
MeshInfo          mesh_get_info(MeshHandle h);
AttributeResult   mesh_vertex_normals(MeshHandle h);
AttributeResult   mesh_gaussian_curvature(MeshHandle h);
AttributeResult   mesh_geodesic(MeshHandle h, const int* seeds, int num_seeds);
MeshResult        mesh_smooth(MeshHandle h, int iterations, double lambda);
MeshResult        mesh_filter(MeshHandle h, int iterations);

// Get current vertex positions (e.g. after smooth modifies them in-place).
void mesh_get_vertices(MeshHandle h, double* out_vertices);
int  mesh_get_num_vertices(MeshHandle h);
int  mesh_get_num_faces(MeshHandle h);

#endif // PYRXMESH_PIPELINE_H
