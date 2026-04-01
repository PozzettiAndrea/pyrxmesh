// GPU sort-scan mesh topology construction.
// Replaces unordered_map-based edge building with parallel sort on GPU.
// Reference: Possemiers & Lee 2015, CMU Parallel Loop Subdivision project.

#include "rxmesh/gpu_build_topology.cuh"
#include "rxmesh/util/macros.h"
#include <chrono>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

namespace rxmesh {

// Each face produces 3 directed edges. Pack (min_v, max_v) into a uint64 key.
__global__ static void expand_edges_kernel(
    const uint32_t* faces, uint32_t num_faces,
    uint64_t* edge_key, uint32_t* edge_face)
{
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    uint32_t v0 = faces[f * 3 + 0];
    uint32_t v1 = faces[f * 3 + 1];
    uint32_t v2 = faces[f * 3 + 2];

    auto pack = [](uint32_t a, uint32_t b) -> uint64_t {
        uint32_t lo = min(a, b), hi = max(a, b);
        return (uint64_t(lo) << 32) | uint64_t(hi);
    };

    edge_key[f * 3 + 0]  = pack(v0, v1);
    edge_face[f * 3 + 0] = f;
    edge_key[f * 3 + 1]  = pack(v1, v2);
    edge_face[f * 3 + 1] = f;
    edge_key[f * 3 + 2]  = pack(v2, v0);
    edge_face[f * 3 + 2] = f;
}

// Mark where key changes → new unique edge
__global__ static void mark_unique_kernel(
    const uint64_t* edge_key, uint32_t n, uint32_t* is_new)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    is_new[i] = (i == 0 || edge_key[i] != edge_key[i-1]) ? 1 : 0;
}

// Single pass: each first occurrence writes ev, ef_f0, and checks next for ef_f1.
__global__ static void extract_edges_kernel(
    const uint64_t* edge_key, const uint32_t* edge_face,
    const uint32_t* edge_id, uint32_t n,
    uint32_t* ev_flat, uint32_t* ef_f0, uint32_t* ef_f1,
    uint32_t* ff_count)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    bool is_first = (i == 0 || edge_key[i] != edge_key[i-1]);
    if (!is_first) return;

    uint32_t eid = edge_id[i];
    ev_flat[eid * 2 + 0] = static_cast<uint32_t>(edge_key[i] >> 32);
    ev_flat[eid * 2 + 1] = static_cast<uint32_t>(edge_key[i] & 0xFFFFFFFF);
    ef_f0[eid] = edge_face[i];

    if (i + 1 < n && edge_key[i + 1] == edge_key[i]) {
        ef_f1[eid] = edge_face[i + 1];
        atomicAdd(&ff_count[edge_face[i]], 1u);
        atomicAdd(&ff_count[edge_face[i + 1]], 1u);
    }
}

// Build ff_values from ef data
__global__ static void build_ff_kernel(
    const uint32_t* ef_f0, const uint32_t* ef_f1,
    uint32_t num_edges,
    const uint32_t* ff_offset, uint32_t* ff_cursor, uint32_t* ff_values)
{
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    if (ef_f1[e] == UINT32_MAX) return;

    uint32_t f0 = ef_f0[e], f1 = ef_f1[e];
    uint32_t slot0 = ff_offset[f0] + atomicAdd(&ff_cursor[f0], 1u);
    uint32_t slot1 = ff_offset[f1] + atomicAdd(&ff_cursor[f1], 1u);
    ff_values[slot0] = f1;
    ff_values[slot1] = f0;
}


GpuTopoResult gpu_build_topology(const uint32_t* h_faces, uint32_t num_faces)
{
    uint32_t num_half_edges = num_faces * 3;
    constexpr int BLOCK = 256;

    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t_total = clk::now();

    // Upload faces
    uint32_t* d_faces;
    CUDA_ERROR(cudaMalloc(&d_faces, num_faces * 3 * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_faces, h_faces, num_faces * 3 * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    // Step 1: Expand faces → edge keys + face IDs
    uint64_t* d_edge_key;
    uint32_t* d_edge_face;
    CUDA_ERROR(cudaMalloc(&d_edge_key,  num_half_edges * sizeof(uint64_t)));
    CUDA_ERROR(cudaMalloc(&d_edge_face, num_half_edges * sizeof(uint32_t)));

    int grid = (num_faces + BLOCK - 1) / BLOCK;
    expand_edges_kernel<<<grid, BLOCK>>>(d_faces, num_faces, d_edge_key, d_edge_face);
    CUDA_ERROR(cudaFree(d_faces));

    // Step 2: Sort by packed key
    thrust::sort_by_key(
        thrust::device,
        thrust::device_pointer_cast(d_edge_key),
        thrust::device_pointer_cast(d_edge_key) + num_half_edges,
        thrust::device_pointer_cast(d_edge_face));

    // Step 3: Mark unique + prefix sum → edge IDs
    uint32_t* d_is_new;
    CUDA_ERROR(cudaMalloc(&d_is_new, num_half_edges * sizeof(uint32_t)));
    grid = (num_half_edges + BLOCK - 1) / BLOCK;
    mark_unique_kernel<<<grid, BLOCK>>>(d_edge_key, num_half_edges, d_is_new);

    uint32_t* d_edge_id;
    CUDA_ERROR(cudaMalloc(&d_edge_id, num_half_edges * sizeof(uint32_t)));
    thrust::exclusive_scan(
        thrust::device,
        thrust::device_pointer_cast(d_is_new),
        thrust::device_pointer_cast(d_is_new) + num_half_edges,
        thrust::device_pointer_cast(d_edge_id));

    uint32_t last_id, last_new;
    CUDA_ERROR(cudaMemcpy(&last_id, d_edge_id + num_half_edges - 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(&last_new, d_is_new + num_half_edges - 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t num_edges = last_id + last_new;
    CUDA_ERROR(cudaFree(d_is_new));

    // Step 4: Extract ev, ef, ff_count
    uint32_t *d_ev_flat, *d_ef_f0, *d_ef_f1, *d_ff_count;
    CUDA_ERROR(cudaMalloc(&d_ev_flat, num_edges * 2 * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ef_f0,   num_edges * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ef_f1,   num_edges * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ff_count, num_faces * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_ef_f1, 0xFF, num_edges * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_ff_count, 0, num_faces * sizeof(uint32_t)));

    extract_edges_kernel<<<grid, BLOCK>>>(
        d_edge_key, d_edge_face, d_edge_id, num_half_edges,
        d_ev_flat, d_ef_f0, d_ef_f1, d_ff_count);
    CUDA_ERROR(cudaDeviceSynchronize());

    // Keep d_edge_key for downstream K1 edge lookups (freed via result.free_device())
    // d_edge_face and d_edge_id are no longer needed
    CUDA_ERROR(cudaFree(d_edge_face));
    CUDA_ERROR(cudaFree(d_edge_id));

    // Step 5: Build ff_offset (prefix sum of ff_count)
    uint32_t* d_ff_offset;
    CUDA_ERROR(cudaMalloc(&d_ff_offset, (num_faces + 1) * sizeof(uint32_t)));
    thrust::exclusive_scan(
        thrust::device,
        thrust::device_pointer_cast(d_ff_count),
        thrust::device_pointer_cast(d_ff_count) + num_faces,
        thrust::device_pointer_cast(d_ff_offset));

    uint32_t ff_total;
    {
        uint32_t last_count, last_off;
        CUDA_ERROR(cudaMemcpy(&last_count, d_ff_count + num_faces - 1,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(&last_off, d_ff_offset + num_faces - 1,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        ff_total = last_off + last_count;
    }
    CUDA_ERROR(cudaMemcpy(d_ff_offset + num_faces, &ff_total,
                          sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Step 6: Build ff_values
    uint32_t *d_ff_values, *d_ff_cursor;
    CUDA_ERROR(cudaMalloc(&d_ff_values, ff_total * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ff_cursor, num_faces * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_ff_cursor, 0, num_faces * sizeof(uint32_t)));

    grid = (num_edges + BLOCK - 1) / BLOCK;
    build_ff_kernel<<<grid, BLOCK>>>(
        d_ef_f0, d_ef_f1, num_edges,
        d_ff_offset, d_ff_cursor, d_ff_values);
    CUDA_ERROR(cudaDeviceSynchronize());

    // Copy results to host
    GpuTopoResult result;
    result.num_edges = num_edges;
    result.ev_flat.resize(num_edges * 2);
    result.ef_f0.resize(num_edges);
    result.ef_f1.resize(num_edges);
    result.ff_offset.resize(num_faces + 1);
    result.ff_values.resize(ff_total);

    CUDA_ERROR(cudaMemcpy(result.ev_flat.data(), d_ev_flat,
                          num_edges * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(result.ef_f0.data(), d_ef_f0,
                          num_edges * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(result.ef_f1.data(), d_ef_f1,
                          num_edges * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(result.ff_offset.data(), d_ff_offset,
                          (num_faces + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(result.ff_values.data(), d_ff_values,
                          ff_total * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Find num_vertices from ev
    uint32_t max_v = 0;
    for (uint32_t i = 0; i < num_edges * 2; ++i)
        max_v = std::max(max_v, result.ev_flat[i]);
    result.num_vertices = max_v + 1;

    // Retain device arrays for downstream GPU kernels
    result.d_edge_key = d_edge_key;
    result.d_ev = d_ev_flat;

    CUDA_ERROR(cudaFree(d_ef_f0));
    CUDA_ERROR(cudaFree(d_ef_f1));
    CUDA_ERROR(cudaFree(d_ff_count));
    CUDA_ERROR(cudaFree(d_ff_offset));
    CUDA_ERROR(cudaFree(d_ff_values));
    CUDA_ERROR(cudaFree(d_ff_cursor));

    fprintf(stderr, "[gpu_topo] TOTAL: %.0fms (%u faces -> %u edges, %u verts)\n",
            ms_since(t_total), num_faces, num_edges, result.num_vertices);

    return result;
}

}  // namespace rxmesh
