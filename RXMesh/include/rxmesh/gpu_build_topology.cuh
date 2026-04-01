#pragma once

#include <cstdint>
#include <vector>

namespace rxmesh {

// GPU sort-scan edge construction result.
// All arrays are flat and contiguous.
struct GpuTopoResult {
    uint32_t num_vertices;
    uint32_t num_edges;
    std::vector<uint32_t> ev_flat;     // [num_edges * 2] edge-vertex
    std::vector<uint32_t> ef_f0;       // [num_edges] first face per edge
    std::vector<uint32_t> ef_f1;       // [num_edges] second face (UINT32_MAX if boundary)
    std::vector<uint32_t> ff_offset;   // [num_faces + 1] CSR offsets
    std::vector<uint32_t> ff_values;   // face-face adjacency values

    // Retained device arrays (for downstream GPU kernels)
    uint64_t* d_edge_key = nullptr;    // [num_edges] sorted packed keys
    uint32_t* d_ev = nullptr;          // [num_edges * 2] on device

    void free_device() {
        if (d_edge_key) { cudaFree(d_edge_key); d_edge_key = nullptr; }
        if (d_ev) { cudaFree(d_ev); d_ev = nullptr; }
    }
};

// Build mesh topology entirely on GPU using thrust sort-scan.
// Takes flat face array (3 ints per face), returns edge/adjacency data.
GpuTopoResult gpu_build_topology(const uint32_t* faces, uint32_t num_faces);

}  // namespace rxmesh
