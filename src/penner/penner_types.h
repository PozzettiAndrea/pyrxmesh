// Penner coordinate optimization — data types
// Implements "Metric Optimization in Penner Coordinates" (Capouellez & Zorin, 2023)
// and "Seamless Parametrization in Penner Coordinates" (2024)
// Pure CUDA + cuSolver — no RXMesh dependency.

#pragma once

#include <vector>
#include <string>
#include <cstdint>

// Halfedge mesh connectivity (host-side, copied to device for kernels)
struct HalfedgeMesh {
    int num_vertices;
    int num_edges;
    int num_halfedges;
    int num_faces;

    // Per-halfedge arrays (size = num_halfedges)
    std::vector<int> next;       // next halfedge in face
    std::vector<int> prev;       // prev halfedge in face (= next[next[h]])
    std::vector<int> twin;       // opposite halfedge (-1 if boundary)
    std::vector<int> to;         // vertex this halfedge points to
    std::vector<int> face;       // face this halfedge belongs to
    std::vector<int> edge;       // edge index (shared with twin)

    // Per-face arrays (size = num_faces)
    std::vector<int> fhe;        // one halfedge per face

    // Per-vertex arrays (size = num_vertices)
    std::vector<int> vhe;        // one outgoing halfedge per vertex
    std::vector<double> Th_hat;  // target cone angles (2π for interior regular)

    // Per-edge arrays (size = num_edges)
    std::vector<int> e2he;       // one halfedge per edge

    // Build from V, F arrays
    void build(const double* vertices, int nv, const int* faces, int nf);
};

// Device-side pointers for GPU kernels (flat arrays on device)
struct HalfedgeMeshDevice {
    int num_vertices;
    int num_edges;
    int num_halfedges;
    int num_faces;

    int* next;
    int* prev;
    int* twin;
    int* to;
    int* face;
    int* edge;
    int* fhe;
    int* e2he;

    double* log_length;   // per-halfedge log(l_e), main optimization variable
    double* he_angle;     // per-halfedge corner angle (opposite)
    double* he_cot;       // per-halfedge cotangent
    double* vertex_angle; // per-vertex angle sum
    double* Th_hat;       // per-vertex target cone angle
};

struct PennerResult {
    std::vector<double> log_lengths;   // per-edge optimized log(l_e)
    std::vector<double> vertex_angles; // per-vertex realized cone angles
    int num_vertices;
    int num_edges;
    int newton_iterations;
    double final_error;
    double total_time_ms;
};

struct PennerConformalParams {
    double error_eps = 1e-10;          // convergence threshold
    int max_iterations = 100;          // max Newton iterations
    double min_angle_deg = 25.0;       // min angle for metric interpolation
    double line_search_alpha = 0.5;    // backtracking parameter
    bool verbose = false;
    std::string debug_dir = "";        // if set, dump per-step data here
};

// Extended result including the halfedge mesh for layout
struct PennerFullResult {
    HalfedgeMesh mesh;                // intrinsic mesh after Ptolemy flips
    std::vector<double> he_length;    // edge lengths after optimization
    std::vector<double> scale_factors;// per-vertex conformal scale factors
    int newton_iterations;
    double final_error;
    double total_time_ms;
};
