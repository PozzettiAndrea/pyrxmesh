#pragma once

#include <cstdint>
#include "rxmesh/gpu_build_topology.cuh"

namespace rxmesh {

// Forward declarations
class RXMesh;
struct PatchInfo;

// Result of GPU patch construction — device pointers to bulk-allocated arrays.
struct GpuPatchBuildResult {
    uint32_t num_patches;
    uint32_t num_vertices;
    uint32_t num_edges;
    uint32_t num_faces;

    // Per-element patch ownership (device arrays)
    uint32_t* d_edge_patch;    // [num_edges]
    uint32_t* d_vertex_patch;  // [num_vertices]

    // Per-patch element counts
    uint16_t max_vertices_per_patch;
    uint16_t max_edges_per_patch;
    uint16_t max_faces_per_patch;

    // Whether construction succeeded
    bool success;
};

// Build all per-patch data structures on GPU.
// Takes GPU-resident topology + partition results, produces PatchInfo array.
// Replaces: assign_patch, extract_ribbons, build_single_patch_ltog,
//           build_single_patch_topology, build_device.
GpuPatchBuildResult gpu_build_patches(
    // GPU topology (from gpu_build_topology, kept on device)
    const uint32_t* d_fv,         // [num_faces * 3]
    const uint64_t* d_edge_key,   // [num_edges] packed (min<<32|max)
    const uint32_t* d_ev,         // [num_edges * 2]
    const uint32_t* d_ef_f0,      // [num_edges]
    const uint32_t* d_ef_f1,      // [num_edges]
    const uint32_t* d_ff_offset,  // [num_faces + 1]
    const uint32_t* d_ff_values,  // CSR values
    uint32_t num_vertices,
    uint32_t num_edges,
    uint32_t num_faces,
    // Patcher results (on device)
    const uint32_t* d_face_patch, // [num_faces]
    uint32_t num_patches,
    // Capacities
    float capacity_factor,
    float lp_load_factor,
    // Output
    PatchInfo* d_patches_info,    // [max_num_patches] device array
    PatchInfo* h_patches_info);   // [max_num_patches] host mirror

// Test wrapper: run K1 (build_ltog) and return results for validation.
struct K1Result {
    std::vector<uint32_t> ltog_f;  // all patches concatenated
    std::vector<uint32_t> ltog_e;
    std::vector<uint32_t> ltog_v;
    std::vector<uint16_t> num_elements_f;
    std::vector<uint16_t> num_elements_e;
    std::vector<uint16_t> num_elements_v;
    std::vector<uint16_t> num_owned_f;
    std::vector<uint16_t> num_owned_e;
    std::vector<uint16_t> num_owned_v;
    uint32_t max_f_per_patch;  // stride
    uint32_t max_e_per_patch;
    uint32_t max_v_per_patch;
};

K1Result gpu_test_k1(
    const uint32_t* d_fv,
    const uint64_t* d_edge_key,
    uint32_t num_edges_global,
    const uint32_t* d_patches_val,
    const uint32_t* d_patches_offset,
    const uint32_t* d_ribbon_val,
    const uint32_t* d_ribbon_offset,  // [P+1]
    const uint32_t* d_face_patch,
    const uint32_t* d_edge_patch,
    const uint32_t* d_vertex_patch,
    uint32_t num_patches,
    uint32_t max_f, uint32_t max_e, uint32_t max_v);

// Test wrapper: run K1+K2 together and return ltog + topology results
struct K1K2Result {
    // K1 outputs
    std::vector<uint32_t> ltog_f, ltog_e, ltog_v;  // concatenated
    std::vector<uint16_t> num_elements_f, num_elements_e, num_elements_v;
    std::vector<uint16_t> num_owned_f, num_owned_e, num_owned_v;
    uint32_t max_f, max_e, max_v;  // strides

    // K2 outputs: local topology per patch (concatenated, stride = max_cap * 2 or * 3)
    std::vector<uint16_t> ev_local;  // [P * ev_stride]
    std::vector<uint16_t> fe_local;  // [P * fe_stride]
    uint32_t ev_stride, fe_stride;
};

// Run K0a only: assign edge/vertex patches on GPU
void gpu_run_k0a(
    const uint32_t* d_face_patch,
    const uint32_t* d_ev,
    const uint32_t* d_ef_f0,
    uint32_t* d_edge_patch,    // output [E]
    uint32_t* d_vertex_patch,  // output [V], must be pre-initialized to INVALID32
    uint32_t num_edges);

K1K2Result gpu_run_k1k2(
    const uint32_t* d_fv,
    const uint64_t* d_edge_key,
    const uint32_t* d_ev_global,  // [E*2] on device
    uint32_t num_edges_global,
    const uint32_t* d_patches_val,
    const uint32_t* d_patches_offset,
    const uint32_t* d_ribbon_val,
    const uint32_t* d_ribbon_offset,
    const uint32_t* d_face_patch,
    const uint32_t* d_edge_patch,
    const uint32_t* d_vertex_patch,
    uint32_t num_patches,
    uint32_t max_f, uint32_t max_e, uint32_t max_v,
    uint32_t edge_capacity, uint32_t face_capacity);

}  // namespace rxmesh
