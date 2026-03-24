// SPDX-License-Identifier: MIT

// ============================================================
// GPU Linear BVH for nearest-triangle queries
//
// Morton-code based radix tree (Karras 2012).
// Used by tangential smoothing for project_to_reference().
// ============================================================

#include "gpu_bvh.cuh"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#define CUDA_CHECK(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            printf("[GPU-BVH] CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err));                                    \
        }                                                                      \
    } while (0)

namespace pyrxmesh_bvh {

// ============================================================
// Device helpers
// ============================================================

__device__ inline float3 make_f3(float x, float y, float z)
{
    return make_float3(x, y, z);
}

__device__ inline float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(float s, float3 a)
{
    return make_float3(s * a.x, s * a.y, s * a.z);
}

__device__ inline float dot3(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 cross3(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ inline float sqrnorm3(float3 a) { return dot3(a, a); }

__device__ inline float norm3(float3 a) { return sqrtf(sqrnorm3(a)); }

__device__ inline float3 fminf3(float3 a, float3 b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ inline float3 fmaxf3(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

// ============================================================
// dist_point_triangle on GPU (port of CPU version)
// ============================================================

__device__ float gpu_dist_point_line_segment(float3 p, float3 v0, float3 v1,
                                             float3& nearest)
{
    float3 d1 = p - v0;
    float3 d2 = v1 - v0;
    float3 min_v = v0;
    float t = sqrnorm3(d2);
    if (t > 1e-30f)
    {
        t = dot3(d1, d2) / t;
        if (t > 1.0f)
        {
            min_v = v1;
            d1 = p - v1;
        }
        else if (t > 0.0f)
        {
            min_v = v0 + t * d2;
            d1 = p - min_v;
        }
    }
    nearest = min_v;
    return norm3(d1);
}

__device__ float gpu_dist_point_triangle(float3 p, float3 v0, float3 v1,
                                         float3 v2, float3& nearest)
{
    float3 v0v1 = v1 - v0;
    float3 v0v2 = v2 - v0;
    float3 n = cross3(v0v1, v0v2);
    float d = sqrnorm3(n);

    // Degenerate triangle
    if (fabsf(d) < 1e-30f)
    {
        float3 q;
        float dist = gpu_dist_point_line_segment(p, v0, v1, nearest);
        float other = gpu_dist_point_line_segment(p, v1, v2, q);
        if (other < dist) { dist = other; nearest = q; }
        other = gpu_dist_point_line_segment(p, v2, v0, q);
        if (other < dist) { dist = other; nearest = q; }
        return dist;
    }

    float inv_d = 1.0f / d;
    float3 v1v2 = v2 - v1;
    float3 v0p = p - v0;
    float3 t = cross3(v0p, n);
    float a = dot3(t, v0v2) * -inv_d;
    float b = dot3(t, v0v1) * inv_d;
    float s01, s02, s12;

    if (a < 0.0f)
    {
        s02 = dot3(v0v2, v0p) / sqrnorm3(v0v2);
        if (s02 < 0.0f)
        {
            s01 = dot3(v0v1, v0p) / sqrnorm3(v0v1);
            if (s01 <= 0.0f) v0p = v0;
            else if (s01 >= 1.0f) v0p = v1;
            else v0p = v0 + s01 * v0v1;
        }
        else if (s02 > 1.0f)
        {
            s12 = dot3(v1v2, p - v1) / sqrnorm3(v1v2);
            if (s12 >= 1.0f) v0p = v2;
            else if (s12 <= 0.0f) v0p = v1;
            else v0p = v1 + s12 * v1v2;
        }
        else
        {
            v0p = v0 + s02 * v0v2;
        }
    }
    else if (b < 0.0f)
    {
        s01 = dot3(v0v1, v0p) / sqrnorm3(v0v1);
        if (s01 < 0.0f)
        {
            s02 = dot3(v0v2, v0p) / sqrnorm3(v0v2);
            if (s02 <= 0.0f) v0p = v0;
            else if (s02 >= 1.0f) v0p = v2;
            else v0p = v0 + s02 * v0v2;
        }
        else if (s01 > 1.0f)
        {
            s12 = dot3(v1v2, p - v1) / sqrnorm3(v1v2);
            if (s12 >= 1.0f) v0p = v2;
            else if (s12 <= 0.0f) v0p = v1;
            else v0p = v1 + s12 * v1v2;
        }
        else
        {
            v0p = v0 + s01 * v0v1;
        }
    }
    else if (a + b > 1.0f)
    {
        s12 = dot3(v1v2, p - v1) / sqrnorm3(v1v2);
        if (s12 >= 1.0f)
        {
            s02 = dot3(v0v2, v0p) / sqrnorm3(v0v2);
            if (s02 <= 0.0f) v0p = v0;
            else if (s02 >= 1.0f) v0p = v2;
            else v0p = v0 + s02 * v0v2;
        }
        else if (s12 <= 0.0f)
        {
            s01 = dot3(v0v1, v0p) / sqrnorm3(v0v1);
            if (s01 <= 0.0f) v0p = v0;
            else if (s01 >= 1.0f) v0p = v1;
            else v0p = v0 + s01 * v0v1;
        }
        else
        {
            v0p = v1 + s12 * v1v2;
        }
    }
    else
    {
        // Interior point
        float3 proj = (dot3(n, v0p) * inv_d) * n;
        v0p = p - proj;
    }

    nearest = v0p;
    return norm3(v0p - p);
}

// Barycentric coordinates (port of CPU version)
__device__ void gpu_barycentric(float3 p, float3 u, float3 v, float3 w,
                                float& b0, float& b1, float& b2)
{
    float3 vu = v - u, wu = w - u, pu = p - u;
    float nx = vu.y * wu.z - vu.z * wu.y;
    float ny = vu.z * wu.x - vu.x * wu.z;
    float nz = vu.x * wu.y - vu.y * wu.x;
    float ax = fabsf(nx), ay = fabsf(ny), az = fabsf(nz);

    b0 = b1 = b2 = 1.0f / 3.0f; // default: barycenter

    if (ax > ay && ax > az)
    {
        if (1.0f + ax != 1.0f)
        {
            b1 = (pu.y * wu.z - pu.z * wu.y) / nx;
            b2 = (vu.y * pu.z - vu.z * pu.y) / nx;
            b0 = 1.0f - b1 - b2;
        }
    }
    else if (ay > az)
    {
        if (1.0f + ay != 1.0f)
        {
            b1 = (pu.z * wu.x - pu.x * wu.z) / ny;
            b2 = (vu.z * pu.x - vu.x * pu.z) / ny;
            b0 = 1.0f - b1 - b2;
        }
    }
    else
    {
        if (1.0f + az != 1.0f)
        {
            b1 = (pu.x * wu.y - pu.y * wu.x) / nz;
            b2 = (vu.x * pu.y - vu.y * pu.x) / nz;
            b0 = 1.0f - b1 - b2;
        }
    }
}

// ============================================================
// Morton codes
// ============================================================

__device__ unsigned int expand_bits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ unsigned int morton3D(float x, float y, float z)
{
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expand_bits((unsigned int)x);
    unsigned int yy = expand_bits((unsigned int)y);
    unsigned int zz = expand_bits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

// ============================================================
// Kernels for BVH construction
// ============================================================

// Extract triangle vertices from V+F arrays and compute centroids + AABB
__global__ void k_extract_triangles(const float* __restrict__ V,
                                    const int* __restrict__ F, int nF,
                                    float* __restrict__ tri_v0,
                                    float* __restrict__ tri_v1,
                                    float* __restrict__ tri_v2,
                                    float3* __restrict__ centroids,
                                    float3* __restrict__ aabb_min_out,
                                    float3* __restrict__ aabb_max_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;

    int i0 = F[3 * i + 0], i1 = F[3 * i + 1], i2 = F[3 * i + 2];
    float3 v0 = make_float3(V[3 * i0], V[3 * i0 + 1], V[3 * i0 + 2]);
    float3 v1 = make_float3(V[3 * i1], V[3 * i1 + 1], V[3 * i1 + 2]);
    float3 v2 = make_float3(V[3 * i2], V[3 * i2 + 1], V[3 * i2 + 2]);

    tri_v0[3 * i] = v0.x; tri_v0[3 * i + 1] = v0.y; tri_v0[3 * i + 2] = v0.z;
    tri_v1[3 * i] = v1.x; tri_v1[3 * i + 1] = v1.y; tri_v1[3 * i + 2] = v1.z;
    tri_v2[3 * i] = v2.x; tri_v2[3 * i + 1] = v2.y; tri_v2[3 * i + 2] = v2.z;

    centroids[i] = make_float3((v0.x + v1.x + v2.x) / 3.0f,
                               (v0.y + v1.y + v2.y) / 3.0f,
                               (v0.z + v1.z + v2.z) / 3.0f);

    float3 mn = fminf3(v0, fminf3(v1, v2));
    float3 mx = fmaxf3(v0, fmaxf3(v1, v2));
    aabb_min_out[i] = mn;
    aabb_max_out[i] = mx;
}

// Compute Morton codes from normalized centroids
__global__ void k_morton_codes(const float3* __restrict__ centroids, int nF,
                               float3 scene_min, float3 scene_extent,
                               unsigned int* __restrict__ morton)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;

    float3 c = centroids[i];
    float nx = (scene_extent.x > 0) ? (c.x - scene_min.x) / scene_extent.x : 0;
    float ny = (scene_extent.y > 0) ? (c.y - scene_min.y) / scene_extent.y : 0;
    float nz = (scene_extent.z > 0) ? (c.z - scene_min.z) / scene_extent.z : 0;
    morton[i] = morton3D(nx, ny, nz);
}

// Build leaf AABBs after sorting
__global__ void k_build_leaf_aabb(const int* __restrict__ sorted_idx, int nF,
                                  const float3* __restrict__ face_aabb_min,
                                  const float3* __restrict__ face_aabb_max,
                                  float* __restrict__ aabb_min,
                                  float* __restrict__ aabb_max, int nInternal)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;

    int leaf = nInternal + i;
    int fi = sorted_idx[i];
    float3 mn = face_aabb_min[fi];
    float3 mx = face_aabb_max[fi];
    aabb_min[3 * leaf] = mn.x; aabb_min[3 * leaf + 1] = mn.y; aabb_min[3 * leaf + 2] = mn.z;
    aabb_max[3 * leaf] = mx.x; aabb_max[3 * leaf + 1] = mx.y; aabb_max[3 * leaf + 2] = mx.z;
}

// Karras 2012 internal node construction
__device__ int delta(const unsigned int* __restrict__ morton, int nF, int i,
                     int j)
{
    if (j < 0 || j >= nF) return -1;
    if (morton[i] == morton[j])
        return 32 + __clz(i ^ j);
    return __clz(morton[i] ^ morton[j]);
}

__global__ void k_build_internal_nodes(const unsigned int* __restrict__ morton,
                                       int nF, int* __restrict__ left,
                                       int* __restrict__ right,
                                       int* __restrict__ parent, int nInternal)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInternal) return;

    // Determine direction
    int d_left = delta(morton, nF, i, i - 1);
    int d_right = delta(morton, nF, i, i + 1);
    int dir = (d_right > d_left) ? 1 : -1;

    // Compute upper bound for range length
    int d_min = delta(morton, nF, i, i - dir);
    int lmax = 2;
    while (delta(morton, nF, i, i + lmax * dir) > d_min)
        lmax *= 2;

    // Binary search for range end
    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2)
    {
        if (delta(morton, nF, i, i + (l + t) * dir) > d_min)
            l += t;
    }
    int j = i + l * dir;

    // Find split position
    int d_node = delta(morton, nF, i, j);
    int s = 0;
    float div = 2.0f;
    int t_val = (int)ceilf((float)(l) / div);
    while (t_val >= 1)
    {
        if (delta(morton, nF, i, i + (s + t_val) * dir) > d_node)
            s += t_val;
        div *= 2.0f;
        t_val = (int)ceilf((float)(l) / div);
    }
    int gamma = i + s * dir + min(dir, 0);

    // Output children
    int left_child = (min(i, j) == gamma) ? nInternal + gamma : gamma;
    int right_child = (max(i, j) == gamma + 1) ? nInternal + gamma + 1
                                                : gamma + 1;

    left[i] = left_child;
    right[i] = right_child;
    parent[left_child] = i;
    parent[right_child] = i;
}

// Bottom-up AABB propagation using atomic flags
__global__ void k_propagate_aabb(int nF, int nInternal,
                                 const int* __restrict__ parent_arr,
                                 const int* __restrict__ left_arr,
                                 const int* __restrict__ right_arr,
                                 float* __restrict__ aabb_min,
                                 float* __restrict__ aabb_max,
                                 int* __restrict__ flags)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nF) return;

    int idx = nInternal + i; // leaf index
    int p = parent_arr[idx];

    while (p >= 0)
    {
        // atomicAdd returns old value; first thread to arrive sees 0 and exits
        if (atomicAdd(&flags[p], 1) == 0)
            return;

        // Second thread arriving merges children
        int lc = left_arr[p];
        int rc = right_arr[p];
        float3 lmn = make_float3(aabb_min[3 * lc], aabb_min[3 * lc + 1], aabb_min[3 * lc + 2]);
        float3 lmx = make_float3(aabb_max[3 * lc], aabb_max[3 * lc + 1], aabb_max[3 * lc + 2]);
        float3 rmn = make_float3(aabb_min[3 * rc], aabb_min[3 * rc + 1], aabb_min[3 * rc + 2]);
        float3 rmx = make_float3(aabb_max[3 * rc], aabb_max[3 * rc + 1], aabb_max[3 * rc + 2]);
        float3 mn = fminf3(lmn, rmn);
        float3 mx = fmaxf3(lmx, rmx);
        aabb_min[3 * p] = mn.x; aabb_min[3 * p + 1] = mn.y; aabb_min[3 * p + 2] = mn.z;
        aabb_max[3 * p] = mx.x; aabb_max[3 * p + 1] = mx.y; aabb_max[3 * p + 2] = mx.z;

        idx = p;
        p = parent_arr[p];
    }
}

// ============================================================
// BVH traversal: nearest triangle query
// ============================================================

__global__ void k_bvh_nearest_query(
    int nQ,
    const float* __restrict__ query_pts, // [3*nQ]
    const float* __restrict__ tri_v0,
    const float* __restrict__ tri_v1,
    const float* __restrict__ tri_v2,
    const float* __restrict__ aabb_min,
    const float* __restrict__ aabb_max,
    const int* __restrict__ left_arr,
    const int* __restrict__ right_arr,
    const int* __restrict__ sorted_idx,
    int nInternal, int nF,
    NearestResult* __restrict__ results)
{
    int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nQ) return;

    float3 qp = make_float3(query_pts[3 * qi], query_pts[3 * qi + 1],
                             query_pts[3 * qi + 2]);

    float best_dist = 1e30f;
    int best_face = -1;
    float3 best_nearest = make_float3(0, 0, 0);

    // Stack-based traversal
    int stack[64];
    int sp = 0;
    stack[sp++] = 0; // root

    while (sp > 0)
    {
        int node = stack[--sp];

        if (node >= nInternal)
        {
            // Leaf node
            int leaf_idx = node - nInternal;
            int fi = sorted_idx[leaf_idx];
            float3 v0 = make_float3(tri_v0[3 * fi], tri_v0[3 * fi + 1], tri_v0[3 * fi + 2]);
            float3 v1 = make_float3(tri_v1[3 * fi], tri_v1[3 * fi + 1], tri_v1[3 * fi + 2]);
            float3 v2 = make_float3(tri_v2[3 * fi], tri_v2[3 * fi + 1], tri_v2[3 * fi + 2]);
            float3 nearest;
            float dist = gpu_dist_point_triangle(qp, v0, v1, v2, nearest);
            if (dist < best_dist)
            {
                best_dist = dist;
                best_face = fi;
                best_nearest = nearest;
            }
        }
        else
        {
            // Internal: check AABB distance
            int lc = left_arr[node];
            int rc = right_arr[node];

            // AABB distance for left child
            float3 lmn = make_float3(aabb_min[3 * lc], aabb_min[3 * lc + 1], aabb_min[3 * lc + 2]);
            float3 lmx = make_float3(aabb_max[3 * lc], aabb_max[3 * lc + 1], aabb_max[3 * lc + 2]);
            float3 clamped_l = fmaxf3(lmn, fminf3(qp, lmx));
            float dl = norm3(clamped_l - qp);

            float3 rmn = make_float3(aabb_min[3 * rc], aabb_min[3 * rc + 1], aabb_min[3 * rc + 2]);
            float3 rmx = make_float3(aabb_max[3 * rc], aabb_max[3 * rc + 1], aabb_max[3 * rc + 2]);
            float3 clamped_r = fmaxf3(rmn, fminf3(qp, rmx));
            float dr = norm3(clamped_r - qp);

            // Push closer child last (so it's popped first)
            if (dl < dr)
            {
                if (dr < best_dist && sp < 63) stack[sp++] = rc;
                if (dl < best_dist && sp < 63) stack[sp++] = lc;
            }
            else
            {
                if (dl < best_dist && sp < 63) stack[sp++] = lc;
                if (dr < best_dist && sp < 63) stack[sp++] = rc;
            }
        }
    }

    results[qi].dist = best_dist;
    results[qi].face_idx = best_face;
    results[qi].nearest_x = best_nearest.x;
    results[qi].nearest_y = best_nearest.y;
    results[qi].nearest_z = best_nearest.z;

    // Compute barycentric coordinates
    if (best_face >= 0)
    {
        float3 v0 = make_float3(tri_v0[3 * best_face], tri_v0[3 * best_face + 1], tri_v0[3 * best_face + 2]);
        float3 v1 = make_float3(tri_v1[3 * best_face], tri_v1[3 * best_face + 1], tri_v1[3 * best_face + 2]);
        float3 v2 = make_float3(tri_v2[3 * best_face], tri_v2[3 * best_face + 1], tri_v2[3 * best_face + 2]);
        gpu_barycentric(best_nearest, v0, v1, v2,
                        results[qi].bary_u, results[qi].bary_v,
                        results[qi].bary_w);
    }
}

// ============================================================
// Host API
// ============================================================

void gpu_bvh_build(GpuBVH& bvh, const float* d_V, const int* d_F, int nF,
                   int nV)
{
    (void)nV;
    gpu_bvh_free(bvh);

    bvh.nTriangles = nF;
    int nInternal = nF - 1;
    int nNodes = 2 * nF - 1;
    bvh.nNodes = nNodes;

    const int BS = 256;
    int gridF = (nF + BS - 1) / BS;

    // Allocate triangle data
    CUDA_CHECK(cudaMalloc(&bvh.d_tri_v0, 3 * nF * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bvh.d_tri_v1, 3 * nF * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bvh.d_tri_v2, 3 * nF * sizeof(float)));

    // Temp: centroids and per-face AABBs
    float3* d_centroids;
    float3* d_face_aabb_min;
    float3* d_face_aabb_max;
    CUDA_CHECK(cudaMalloc(&d_centroids, nF * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_face_aabb_min, nF * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_face_aabb_max, nF * sizeof(float3)));

    // Extract triangles
    k_extract_triangles<<<gridF, BS>>>(d_V, d_F, nF, bvh.d_tri_v0,
                                       bvh.d_tri_v1, bvh.d_tri_v2,
                                       d_centroids, d_face_aabb_min,
                                       d_face_aabb_max);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute scene AABB for Morton normalization
    // BVH is built once (persistent state), so host-side reduction is acceptable
    std::vector<float3> h_mn(nF), h_mx(nF);
    CUDA_CHECK(cudaMemcpy(h_mn.data(), d_face_aabb_min, nF * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mx.data(), d_face_aabb_max, nF * sizeof(float3), cudaMemcpyDeviceToHost));

    float3 scene_min = h_mn[0], scene_max = h_mx[0];
    for (int i = 1; i < nF; ++i)
    {
        scene_min.x = std::min(scene_min.x, h_mn[i].x);
        scene_min.y = std::min(scene_min.y, h_mn[i].y);
        scene_min.z = std::min(scene_min.z, h_mn[i].z);
        scene_max.x = std::max(scene_max.x, h_mx[i].x);
        scene_max.y = std::max(scene_max.y, h_mx[i].y);
        scene_max.z = std::max(scene_max.z, h_mx[i].z);
    }
    float3 scene_extent = {scene_max.x - scene_min.x,
                           scene_max.y - scene_min.y,
                           scene_max.z - scene_min.z};

    // Morton codes
    CUDA_CHECK(cudaMalloc(&bvh.d_morton, nF * sizeof(unsigned int)));
    k_morton_codes<<<gridF, BS>>>(d_centroids, nF, scene_min, scene_extent,
                                  bvh.d_morton);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sort by Morton code
    CUDA_CHECK(cudaMalloc(&bvh.d_sorted_idx, nF * sizeof(int)));
    thrust::device_ptr<int> dp_idx(bvh.d_sorted_idx);
    thrust::sequence(thrust::device, dp_idx, dp_idx + nF);
    thrust::device_ptr<unsigned int> dp_morton(bvh.d_morton);
    thrust::sort_by_key(thrust::device, dp_morton, dp_morton + nF, dp_idx);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate BVH node data
    CUDA_CHECK(cudaMalloc(&bvh.d_aabb_min, 3 * nNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bvh.d_aabb_max, 3 * nNodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&bvh.d_left, nInternal * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bvh.d_right, nInternal * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bvh.d_parent, nNodes * sizeof(int)));

    CUDA_CHECK(cudaMemset(bvh.d_parent, 0xFF, nNodes * sizeof(int))); // -1

    // Build leaf AABBs
    k_build_leaf_aabb<<<gridF, BS>>>(bvh.d_sorted_idx, nF, d_face_aabb_min,
                                     d_face_aabb_max, bvh.d_aabb_min,
                                     bvh.d_aabb_max, nInternal);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Build internal nodes (Karras 2012)
    if (nInternal > 0)
    {
        int gridI = (nInternal + BS - 1) / BS;
        k_build_internal_nodes<<<gridI, BS>>>(bvh.d_morton, nF, bvh.d_left,
                                              bvh.d_right, bvh.d_parent,
                                              nInternal);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Bottom-up AABB propagation
        int* d_flags;
        CUDA_CHECK(cudaMalloc(&d_flags, nInternal * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_flags, 0, nInternal * sizeof(int)));

        k_propagate_aabb<<<gridF, BS>>>(nF, nInternal, bvh.d_parent,
                                        bvh.d_left, bvh.d_right,
                                        bvh.d_aabb_min, bvh.d_aabb_max,
                                        d_flags);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_flags);
    }

    // Cleanup temp
    cudaFree(d_centroids);
    cudaFree(d_face_aabb_min);
    cudaFree(d_face_aabb_max);

    printf("[GPU-BVH] Built: %d triangles, %d nodes\n", nF, nNodes);
}

void gpu_bvh_free(GpuBVH& bvh)
{
    if (bvh.d_tri_v0) cudaFree(bvh.d_tri_v0);
    if (bvh.d_tri_v1) cudaFree(bvh.d_tri_v1);
    if (bvh.d_tri_v2) cudaFree(bvh.d_tri_v2);
    if (bvh.d_aabb_min) cudaFree(bvh.d_aabb_min);
    if (bvh.d_aabb_max) cudaFree(bvh.d_aabb_max);
    if (bvh.d_left) cudaFree(bvh.d_left);
    if (bvh.d_right) cudaFree(bvh.d_right);
    if (bvh.d_parent) cudaFree(bvh.d_parent);
    if (bvh.d_morton) cudaFree(bvh.d_morton);
    if (bvh.d_sorted_idx) cudaFree(bvh.d_sorted_idx);
    bvh = GpuBVH{};
}

void gpu_bvh_nearest(const GpuBVH& bvh, const float* d_query_points, int nQ,
                     NearestResult* d_results)
{
    if (nQ == 0 || bvh.nTriangles == 0) return;

    int nInternal = bvh.nTriangles - 1;
    const int BS = 256;
    int grid = (nQ + BS - 1) / BS;

    k_bvh_nearest_query<<<grid, BS>>>(
        nQ, d_query_points, bvh.d_tri_v0, bvh.d_tri_v1, bvh.d_tri_v2,
        bvh.d_aabb_min, bvh.d_aabb_max, bvh.d_left, bvh.d_right,
        bvh.d_sorted_idx, nInternal, bvh.nTriangles, d_results);
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace pyrxmesh_bvh
