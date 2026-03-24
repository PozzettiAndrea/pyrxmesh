// SPDX-License-Identifier: MIT
#pragma once

// ============================================================
// GPU Linear BVH for nearest-triangle queries
//
// Replaces the CPU TriangleKdTree for project_to_reference().
// Uses Morton-code-based radix tree construction (Karras 2012).
// ============================================================

#include <cstdint>

namespace pyrxmesh_bvh {

struct GpuBVH
{
    // Triangle data (reference mesh)
    float* d_tri_v0 = nullptr; // [3*nT] triangle vertex 0
    float* d_tri_v1 = nullptr; // [3*nT] triangle vertex 1
    float* d_tri_v2 = nullptr; // [3*nT] triangle vertex 2

    // BVH nodes
    float* d_aabb_min = nullptr; // [3*nNodes] bounding box min
    float* d_aabb_max = nullptr; // [3*nNodes] bounding box max
    int* d_left = nullptr;       // [nInternal] left child index
    int* d_right = nullptr;      // [nInternal] right child index
    int* d_parent = nullptr;     // [nNodes] parent index

    // Morton codes for construction
    uint32_t* d_morton = nullptr; // [nT] Morton codes
    int* d_sorted_idx = nullptr;  // [nT] sorted triangle indices

    int nTriangles = 0;
    int nNodes = 0; // = 2*nTriangles - 1
};

struct NearestResult
{
    float dist;
    int face_idx;
    float nearest_x, nearest_y, nearest_z;
    float bary_u, bary_v, bary_w;
};

// Build BVH from reference mesh triangles (positions interleaved as float[3*nV])
void gpu_bvh_build(GpuBVH& bvh, const float* d_V, const int* d_F, int nF,
                   int nV);

// Free BVH memory
void gpu_bvh_free(GpuBVH& bvh);

// Batched nearest-triangle query: for each query point, find closest triangle
// query_points: [3*nQ] interleaved xyz
// results: [nQ] output
void gpu_bvh_nearest(const GpuBVH& bvh, const float* d_query_points, int nQ,
                     NearestResult* d_results);

} // namespace pyrxmesh_bvh
