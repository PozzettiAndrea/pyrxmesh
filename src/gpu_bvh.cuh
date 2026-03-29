// SPDX-License-Identifier: MIT
#pragma once

// ============================================================
// GPU Linear BVH for nearest-triangle queries
//
// Replaces the CPU TriangleKdTree for project_to_reference().
// Uses Morton-code-based radix tree construction (Karras 2012).
// ============================================================

#include <cstdint>
#include <cuda_runtime.h>
#include <cmath>

namespace pyrxmesh_bvh {

// ── float3 helpers (inline device) ──────────────────────────
__device__ inline float3 bvh_operator_add(float3 a, float3 b)
{ return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }

__device__ inline float3 bvh_operator_sub(float3 a, float3 b)
{ return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }

__device__ inline float3 bvh_operator_mul(float s, float3 a)
{ return make_float3(s*a.x, s*a.y, s*a.z); }

__device__ inline float bvh_dot3(float3 a, float3 b)
{ return a.x*b.x + a.y*b.y + a.z*b.z; }

__device__ inline float3 bvh_cross3(float3 a, float3 b)
{ return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }

__device__ inline float bvh_sqrnorm3(float3 a) { return bvh_dot3(a, a); }
__device__ inline float bvh_norm3(float3 a) { return sqrtf(bvh_sqrnorm3(a)); }

__device__ inline float3 bvh_fminf3(float3 a, float3 b)
{ return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z)); }

__device__ inline float3 bvh_fmaxf3(float3 a, float3 b)
{ return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z)); }

__device__ inline float bvh_dist_point_line_segment(
    float3 p, float3 v0, float3 v1, float3& nearest)
{
    float3 d1 = bvh_operator_sub(p, v0);
    float3 d2 = bvh_operator_sub(v1, v0);
    float3 min_v = v0;
    float t = bvh_sqrnorm3(d2);
    if (t > 1e-30f) {
        t = bvh_dot3(d1, d2) / t;
        if (t > 1.0f) { min_v = v1; d1 = bvh_operator_sub(p, v1); }
        else if (t > 0.0f) { min_v = bvh_operator_add(v0, bvh_operator_mul(t, d2)); d1 = bvh_operator_sub(p, min_v); }
    }
    nearest = min_v;
    return bvh_norm3(d1);
}

__device__ inline float bvh_dist_point_triangle(
    float3 p, float3 v0, float3 v1, float3 v2, float3& nearest)
{
    float3 v0v1 = bvh_operator_sub(v1, v0);
    float3 v0v2 = bvh_operator_sub(v2, v0);
    float3 n = bvh_cross3(v0v1, v0v2);
    float d = bvh_sqrnorm3(n);
    if (fabsf(d) < 1e-30f) {
        float3 q;
        float dist = bvh_dist_point_line_segment(p, v0, v1, nearest);
        float other = bvh_dist_point_line_segment(p, v1, v2, q);
        if (other < dist) { dist = other; nearest = q; }
        other = bvh_dist_point_line_segment(p, v2, v0, q);
        if (other < dist) { dist = other; nearest = q; }
        return dist;
    }
    float inv_d = 1.0f / d;
    float3 v1v2 = bvh_operator_sub(v2, v1);
    float3 v0p = bvh_operator_sub(p, v0);
    float3 t = bvh_cross3(v0p, n);
    float a = bvh_dot3(t, v0v2) * -inv_d;
    float b = bvh_dot3(t, v0v1) * inv_d;
    float s01, s02, s12;
    if (a < 0.0f) {
        s02 = bvh_dot3(v0v2, v0p) / bvh_sqrnorm3(v0v2);
        if (s02 < 0.0f) {
            s01 = bvh_dot3(v0v1, v0p) / bvh_sqrnorm3(v0v1);
            if (s01 <= 0.0f) v0p = v0;
            else if (s01 >= 1.0f) v0p = v1;
            else v0p = bvh_operator_add(v0, bvh_operator_mul(s01, v0v1));
        } else if (s02 > 1.0f) {
            s12 = bvh_dot3(v1v2, bvh_operator_sub(p, v1)) / bvh_sqrnorm3(v1v2);
            if (s12 >= 1.0f) v0p = v2;
            else if (s12 <= 0.0f) v0p = v1;
            else v0p = bvh_operator_add(v1, bvh_operator_mul(s12, v1v2));
        } else v0p = bvh_operator_add(v0, bvh_operator_mul(s02, v0v2));
    } else if (b < 0.0f) {
        s01 = bvh_dot3(v0v1, v0p) / bvh_sqrnorm3(v0v1);
        if (s01 < 0.0f) {
            s02 = bvh_dot3(v0v2, v0p) / bvh_sqrnorm3(v0v2);
            if (s02 <= 0.0f) v0p = v0;
            else if (s02 >= 1.0f) v0p = v2;
            else v0p = bvh_operator_add(v0, bvh_operator_mul(s02, v0v2));
        } else if (s01 > 1.0f) {
            s12 = bvh_dot3(v1v2, bvh_operator_sub(p, v1)) / bvh_sqrnorm3(v1v2);
            if (s12 >= 1.0f) v0p = v2;
            else if (s12 <= 0.0f) v0p = v1;
            else v0p = bvh_operator_add(v1, bvh_operator_mul(s12, v1v2));
        } else v0p = bvh_operator_add(v0, bvh_operator_mul(s01, v0v1));
    } else if (a + b > 1.0f) {
        s12 = bvh_dot3(v1v2, bvh_operator_sub(p, v1)) / bvh_sqrnorm3(v1v2);
        if (s12 >= 1.0f) {
            s02 = bvh_dot3(v0v2, v0p) / bvh_sqrnorm3(v0v2);
            if (s02 <= 0.0f) v0p = v0;
            else if (s02 >= 1.0f) v0p = v2;
            else v0p = bvh_operator_add(v0, bvh_operator_mul(s02, v0v2));
        } else if (s12 <= 0.0f) {
            s01 = bvh_dot3(v0v1, v0p) / bvh_sqrnorm3(v0v1);
            if (s01 <= 0.0f) v0p = v0;
            else if (s01 >= 1.0f) v0p = v1;
            else v0p = bvh_operator_add(v0, bvh_operator_mul(s01, v0v1));
        } else v0p = bvh_operator_add(v1, bvh_operator_mul(s12, v1v2));
    } else {
        float3 proj = bvh_operator_mul(bvh_dot3(n, v0p) * inv_d, n);
        v0p = bvh_operator_sub(p, proj);
    }
    nearest = v0p;
    return bvh_norm3(bvh_operator_sub(v0p, p));
}

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

// Device-side BVH context — passed as kernel argument for device-callable queries.
// Constructed from host-side GpuBVH.
struct GpuBVHDevice
{
    const float* tri_v0;
    const float* tri_v1;
    const float* tri_v2;
    const float* aabb_min;
    const float* aabb_max;
    const int*   left_arr;
    const int*   right_arr;
    const int*   sorted_idx;
    int nInternal;
    int nTriangles;
};

inline GpuBVHDevice gpu_bvh_device_context(const GpuBVH& bvh)
{
    GpuBVHDevice ctx;
    ctx.tri_v0     = bvh.d_tri_v0;
    ctx.tri_v1     = bvh.d_tri_v1;
    ctx.tri_v2     = bvh.d_tri_v2;
    ctx.aabb_min   = bvh.d_aabb_min;
    ctx.aabb_max   = bvh.d_aabb_max;
    ctx.left_arr   = bvh.d_left;
    ctx.right_arr  = bvh.d_right;
    ctx.sorted_idx = bvh.d_sorted_idx;
    ctx.nInternal  = bvh.nTriangles - 1;
    ctx.nTriangles = bvh.nTriangles;
    return ctx;
}

// Device-callable nearest-point query on BVH
__device__ inline NearestResult gpu_bvh_query_point(
    const GpuBVHDevice& bvh, float px, float py, float pz)
{
    float3 qp = make_float3(px, py, pz);
    float best_dist = 1e30f;
    int best_face = -1;
    float3 best_nearest = make_float3(0, 0, 0);

    int stack[64];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int node = stack[--sp];
        if (node >= bvh.nInternal) {
            int leaf_idx = node - bvh.nInternal;
            int fi = bvh.sorted_idx[leaf_idx];
            float3 v0 = make_float3(bvh.tri_v0[3*fi], bvh.tri_v0[3*fi+1], bvh.tri_v0[3*fi+2]);
            float3 v1 = make_float3(bvh.tri_v1[3*fi], bvh.tri_v1[3*fi+1], bvh.tri_v1[3*fi+2]);
            float3 v2 = make_float3(bvh.tri_v2[3*fi], bvh.tri_v2[3*fi+1], bvh.tri_v2[3*fi+2]);
            float3 nearest;
            float dist = bvh_dist_point_triangle(qp, v0, v1, v2, nearest);
            if (dist < best_dist) {
                best_dist = dist;
                best_face = fi;
                best_nearest = nearest;
            }
        } else {
            int lc = bvh.left_arr[node];
            int rc = bvh.right_arr[node];
            float3 lmn = make_float3(bvh.aabb_min[3*lc], bvh.aabb_min[3*lc+1], bvh.aabb_min[3*lc+2]);
            float3 lmx = make_float3(bvh.aabb_max[3*lc], bvh.aabb_max[3*lc+1], bvh.aabb_max[3*lc+2]);
            float3 cl = bvh_fmaxf3(lmn, bvh_fminf3(qp, lmx));
            float dl = bvh_norm3(bvh_operator_sub(cl, qp));
            float3 rmn = make_float3(bvh.aabb_min[3*rc], bvh.aabb_min[3*rc+1], bvh.aabb_min[3*rc+2]);
            float3 rmx = make_float3(bvh.aabb_max[3*rc], bvh.aabb_max[3*rc+1], bvh.aabb_max[3*rc+2]);
            float3 cr = bvh_fmaxf3(rmn, bvh_fminf3(qp, rmx));
            float dr = bvh_norm3(bvh_operator_sub(cr, qp));
            if (dl < dr) {
                if (dr < best_dist && sp < 63) stack[sp++] = rc;
                if (dl < best_dist && sp < 63) stack[sp++] = lc;
            } else {
                if (dl < best_dist && sp < 63) stack[sp++] = lc;
                if (dr < best_dist && sp < 63) stack[sp++] = rc;
            }
        }
    }

    NearestResult r;
    r.dist = best_dist;
    r.face_idx = best_face;
    r.nearest_x = best_nearest.x;
    r.nearest_y = best_nearest.y;
    r.nearest_z = best_nearest.z;
    return r;
}

} // namespace pyrxmesh_bvh
