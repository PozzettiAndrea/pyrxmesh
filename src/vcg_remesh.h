// CPU isotropic remeshing via VCG (same algorithm as QuadWild's AutoRemesher).
// Pure C++ — no CUDA.

#ifndef PYRXMESH_VCG_REMESH_H
#define PYRXMESH_VCG_REMESH_H

#include <vector>

struct VcgRemeshParams {
    float target_edge_length;   // 0 = auto from mesh area + target_faces
    int   target_faces;         // used when target_edge_length == 0 (default: 10000)
    int   iterations;           // IsotropicRemeshing iterations per pass (default: 3)
    bool  adaptive;             // run second adaptive pass like QuadWild (default: true)
    bool  project;              // project vertices to original surface (default: true)
    float crease_angle_deg;     // feature edge angle threshold (default: 35)
};

struct VcgRemeshResult {
    std::vector<double> vertices;   // flat [x0,y0,z0, ...]
    std::vector<int>    faces;      // flat [v0,v1,v2, ...]
    int num_vertices;
    int num_faces;
};

VcgRemeshResult vcg_remesh(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    const VcgRemeshParams& params,
    bool verbose = false);

// Clean a mesh: remove non-manifold edges/vertices, zero-area faces,
// small components, make orientable. Matches QuadWild's SolveGeometricArtifacts.
// Returns cleaned mesh. Input mesh is not modified.
VcgRemeshResult vcg_clean_mesh(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    bool verbose = false);

// Collapse micro-edges in degenerate triangles.
// Matches QuadWild's AutoRemesher::collapseSurvivingMicroEdges exactly:
// finds faces with QualityRadii <= qualityThr, collapses shortest edge.
VcgRemeshResult vcg_micro_collapse(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    float quality_thr = 0.01f,
    int max_iter = 2,
    bool verbose = false);

#endif // PYRXMESH_VCG_REMESH_H
