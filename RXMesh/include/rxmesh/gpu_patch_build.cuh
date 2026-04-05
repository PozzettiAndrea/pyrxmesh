#pragma once

#include <cstdint>
#include "rxmesh/gpu_build_topology.cuh"
#include "rxmesh/lp_hashtable.h"
#include "rxmesh/patch_stash.h"

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
    std::vector<uint16_t> ev_local;  // [P * ev_stride] host copy
    std::vector<uint16_t> fe_local;  // [P * fe_stride] host copy
    uint32_t ev_stride, fe_stride;

    // K2 device-retained topology (for D2D copy in build_device)
    uint16_t* d_ev_local = nullptr;  // [P * ev_stride] on device
    uint16_t* d_fe_local = nullptr;  // [P * fe_stride] on device

    void free_device_topo() {
        if (d_ev_local) { cudaFree(d_ev_local); d_ev_local = nullptr; }
        if (d_fe_local) { cudaFree(d_fe_local); d_fe_local = nullptr; }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Approach A: Global thrust sort-based ltog construction
// No per-block kernels, no shared memory limits, no hand-rolled sort.
// Uses thrust::sort_by_key + thrust::unique to build per-patch ltog arrays.
// ═══════════════════════════════════════════════════════════════════════════

struct ThrustLtogResult {
    // Per-patch data (host, ready to store in m_h_patches_ltog_*)
    // Flat concatenated arrays + per-patch offsets/counts
    std::vector<uint32_t> ltog_f;  // all faces, sorted by patch then by global ID
    std::vector<uint32_t> ltog_e;  // all edges
    std::vector<uint32_t> ltog_v;  // all vertices
    std::vector<uint32_t> f_offset; // [P+1] where each patch starts in ltog_f
    std::vector<uint32_t> e_offset; // [P+1]
    std::vector<uint32_t> v_offset; // [P+1]
    std::vector<uint16_t> num_owned_f; // [P]
    std::vector<uint16_t> num_owned_e; // [P]
    std::vector<uint16_t> num_owned_v; // [P]

    // Device-resident arrays retained for K2 topology kernel
    uint32_t* d_ltog_f = nullptr;
    uint32_t* d_ltog_e = nullptr;
    uint32_t* d_ltog_v = nullptr;
    uint32_t* d_f_offset = nullptr;   // [P+1]
    uint32_t* d_e_offset = nullptr;
    uint32_t* d_v_offset = nullptr;
    uint16_t* d_num_elements_f = nullptr;  // [P]
    uint16_t* d_num_elements_e = nullptr;
    uint16_t* d_num_elements_v = nullptr;
    uint16_t* d_num_owned_f = nullptr;
    uint16_t* d_num_owned_e = nullptr;
    uint16_t* d_num_owned_v = nullptr;
    bool device_arrays_valid = false;

    void free_device() {
        if (!device_arrays_valid) return;
        cudaFree(d_ltog_f); cudaFree(d_ltog_e); cudaFree(d_ltog_v);
        cudaFree(d_f_offset); cudaFree(d_e_offset); cudaFree(d_v_offset);
        cudaFree(d_num_elements_f); cudaFree(d_num_elements_e);
        cudaFree(d_num_elements_v);
        cudaFree(d_num_owned_f); cudaFree(d_num_owned_e);
        cudaFree(d_num_owned_v);
        device_arrays_valid = false;
    }
};

// GPU two-ring graph coloring via Jones-Plassmann
void gpu_patch_coloring(
    const uint32_t* d_stash,   // [P * stash_size] flat neighbor IDs
    uint32_t num_patches,
    uint32_t stash_size,
    uint32_t* h_colors,        // [P] output on host
    uint32_t& num_colors);

// GPU create_handles: build handle arrays directly on device
void gpu_create_handles(
    const uint32_t* d_vertex_prefix,  // [P+1]
    const uint32_t* d_edge_prefix,
    const uint32_t* d_face_prefix,
    const uint16_t* d_num_owned_v,    // [P] (in ThrustLtogResult or uploaded)
    const uint16_t* d_num_owned_e,
    const uint16_t* d_num_owned_f,
    uint32_t num_patches,
    void* d_v_handles,  // VertexHandle*
    void* d_e_handles,  // EdgeHandle*
    void* d_f_handles); // FaceHandle*

// GPU stash-only build (for populate_patch_stash before coloring)
void gpu_build_stash(
    const ThrustLtogResult& thr,
    const uint32_t* d_face_patch,
    const uint32_t* d_edge_patch,
    const uint32_t* d_vertex_patch,
    uint32_t num_patches,
    uint8_t* d_stash_bulk,
    size_t stash_bytes_per);

// GPU ribbon extraction — vertex-centric via thrust sort.
// Replaces CPU Patcher::extract_ribbons (~1s → ~25ms).
void gpu_extract_ribbons(
    const uint32_t* d_face_patch,
    const uint32_t* d_fv,
    uint32_t num_faces,
    uint32_t num_vertices,
    uint32_t num_patches,
    uint32_t** out_d_ribbon_val,
    uint32_t** out_d_ribbon_offset,
    std::vector<uint32_t>& h_ribbon_val,
    std::vector<uint32_t>& h_ribbon_offset);

// Build bitmasks + stash + hash tables entirely on GPU.
// Fills pre-allocated bulk device arrays directly — no host round-trip.
void gpu_build_device_data(
    const ThrustLtogResult& thr,
    const uint32_t* d_face_patch,
    const uint32_t* d_edge_patch,
    const uint32_t* d_vertex_patch,
    uint32_t num_patches,
    uint16_t v_cap, uint16_t e_cap, uint16_t f_cap,
    uint8_t* d_mask_av_bulk, uint8_t* d_mask_ae_bulk, uint8_t* d_mask_af_bulk,
    uint8_t* d_mask_ov_bulk, uint8_t* d_mask_oe_bulk, uint8_t* d_mask_of_bulk,
    size_t mask_v_bytes, size_t mask_e_bytes, size_t mask_f_bytes,
    uint8_t* d_counts_bulk, size_t counts_bytes,
    uint8_t* d_stash_bulk, size_t stash_bytes_per,
    uint8_t* d_ht_v_bulk, uint8_t* d_ht_e_bulk, uint8_t* d_ht_f_bulk,
    size_t ht_v_bytes, size_t ht_e_bytes, size_t ht_f_bytes,
    uint8_t* d_ht_stash_v_bulk, uint8_t* d_ht_stash_e_bulk,
    uint8_t* d_ht_stash_f_bulk, size_t ht_stash_bytes,
    LPHashTable ht_template_v, LPHashTable ht_template_e, LPHashTable ht_template_f);

ThrustLtogResult gpu_thrust_build_ltog(
    const uint32_t* d_fv,              // [F*3] on device
    const uint64_t* d_edge_key,        // [E] unique sorted packed keys on device
    uint32_t num_faces,
    uint32_t num_edges,
    uint32_t num_vertices,
    // Patcher results (on device)
    const uint32_t* d_face_patch,      // [F]
    const uint32_t* d_edge_patch,      // [E] in GPU edge ID space
    const uint32_t* d_vertex_patch,    // [V]
    // Patch face lists (from Patcher, on device)
    const uint32_t* d_patches_val,     // owned face IDs sorted by patch
    const uint32_t* d_patches_offset,  // [P] cumulative offsets
    const uint32_t* d_ribbon_val,      // ribbon face IDs per patch
    const uint32_t* d_ribbon_offset,   // [P+1] prefix offsets
    uint32_t num_patches);

// Launch K2 topology kernel using retained device arrays from Approach A.
// Returns ev_local/fe_local in K1K2Result, frees thr's device arrays.
K1K2Result gpu_launch_k2(
    ThrustLtogResult& thr,
    const uint32_t* d_fv,
    const uint64_t* d_edge_key,
    const uint32_t* d_ev_global,
    uint32_t num_edges_global,
    const uint32_t* d_patches_val,
    const uint32_t* d_patches_offset,
    const uint32_t* d_ribbon_val,
    const uint32_t* d_ribbon_offset,
    const uint32_t* d_face_patch,
    const uint32_t* d_edge_patch,
    const uint32_t* d_vertex_patch,
    uint32_t num_patches,
    uint32_t edge_capacity,
    uint32_t face_capacity);

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
