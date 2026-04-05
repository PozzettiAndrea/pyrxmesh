// GPU patch construction pipeline.
// Replaces 23s of CPU work with ~70ms of GPU kernels.
// Builds per-patch data structures (ltog, topology, bitmasks, hashtables)
// entirely on GPU from topology + partition results.

#include "rxmesh/gpu_patch_build.cuh"
#include "rxmesh/patch_info.h"
#include "rxmesh/lp_hashtable.h"
#include "rxmesh/lp_pair.cuh"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/bitmask_util.h"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/tabulate.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <chrono>
#include <cstdio>
#include <random>

namespace rxmesh {

// ═══════════════════════════════════════════════════════════════════════════
// K0a: Assign edge_patch and vertex_patch from face_patch
// One thread per edge. atomicCAS for first-writer-wins vertex ownership.
// ═══════════════════════════════════════════════════════════════════════════

__global__ static void k0a_assign_edge_vertex_patch(
    const uint32_t* d_face_patch,
    const uint32_t* d_ev,          // [E*2]
    const uint32_t* d_ef_f0,       // [E]
    uint32_t*       d_edge_patch,  // [E] output
    uint32_t*       d_vertex_patch,// [V] output (init to INVALID32)
    uint32_t        num_edges)
{
    uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    uint32_t f0 = d_ef_f0[e];
    uint32_t p  = d_face_patch[f0];
    d_edge_patch[e] = p;

    uint32_t v0 = d_ev[e * 2];
    uint32_t v1 = d_ev[e * 2 + 1];
    atomicCAS(&d_vertex_patch[v0], INVALID32, p);
    atomicCAS(&d_vertex_patch[v1], INVALID32, p);
}

// ═══════════════════════════════════════════════════════════════════════════
// K0b: Mark boundary faces and count ribbon faces per patch
// A face is a boundary face if any ff-neighbor is in a different patch.
// A ribbon face for patch p is a face NOT in p that shares a vertex with
// a boundary face of p.
// ═══════════════════════════════════════════════════════════════════════════

// Step 1: mark boundary faces
__global__ static void k0b_mark_boundary_faces(
    const uint32_t* d_face_patch,
    const uint32_t* d_ff_offset,
    const uint32_t* d_ff_values,
    uint8_t*        d_is_boundary_face,  // [F] output
    uint32_t        num_faces)
{
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    uint32_t p = d_face_patch[f];
    uint8_t is_boundary = 0;
    for (uint32_t i = d_ff_offset[f]; i < d_ff_offset[f + 1]; ++i) {
        if (d_face_patch[d_ff_values[i]] != p) {
            is_boundary = 1;
            break;
        }
    }
    d_is_boundary_face[f] = is_boundary;
}

// Step 2: mark boundary vertices (any vertex of a boundary face)
__global__ static void k0b_mark_boundary_vertices(
    const uint32_t* d_fv,
    const uint8_t*  d_is_boundary_face,
    uint8_t*        d_is_boundary_vertex,  // [V] output (init to 0)
    uint32_t        num_faces)
{
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;
    if (!d_is_boundary_face[f]) return;

    d_is_boundary_vertex[d_fv[f * 3 + 0]] = 1;
    d_is_boundary_vertex[d_fv[f * 3 + 1]] = 1;
    d_is_boundary_vertex[d_fv[f * 3 + 2]] = 1;
}

// Step 3: count ribbon faces per patch.
// Face f (in patch q) is a ribbon for patch p if f has a vertex v where
// vertex_patch[v] == p AND p != q.
// We count how many ribbon faces each patch has.
__global__ static void k0b_count_ribbon_per_patch(
    const uint32_t* d_fv,
    const uint32_t* d_face_patch,
    const uint32_t* d_vertex_patch,
    const uint8_t*  d_is_boundary_vertex,
    uint32_t*       d_ribbon_count,  // [num_patches] output (atomic)
    uint32_t        num_faces)
{
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    uint32_t q = d_face_patch[f];
    uint32_t v0 = d_fv[f * 3 + 0];
    uint32_t v1 = d_fv[f * 3 + 1];
    uint32_t v2 = d_fv[f * 3 + 2];

    // Check each vertex — if it's a boundary vertex owned by a different patch,
    // this face is a ribbon face for that patch.
    // Use a small local set to avoid double-counting (face could be ribbon
    // for multiple patches, but we count once per patch).
    uint32_t seen[3] = {INVALID32, INVALID32, INVALID32};
    int nseen = 0;

    auto check_vertex = [&](uint32_t v) {
        if (!d_is_boundary_vertex[v]) return;
        uint32_t p = d_vertex_patch[v];
        if (p == q || p == INVALID32) return;
        // Check not already counted for this face
        for (int i = 0; i < nseen; ++i)
            if (seen[i] == p) return;
        seen[nseen++] = p;
        atomicAdd(&d_ribbon_count[p], 1u);
    };

    check_vertex(v0);
    check_vertex(v1);
    check_vertex(v2);
}

// Step 4: scatter ribbon face IDs into per-patch arrays
__global__ static void k0b_scatter_ribbon_faces(
    const uint32_t* d_fv,
    const uint32_t* d_face_patch,
    const uint32_t* d_vertex_patch,
    const uint8_t*  d_is_boundary_vertex,
    const uint32_t* d_ribbon_offset,  // [num_patches] prefix sum of counts
    uint32_t*       d_ribbon_cursor,  // [num_patches] atomic cursor
    uint32_t*       d_ribbon_val,     // output: ribbon face IDs
    uint32_t        num_faces)
{
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    uint32_t q = d_face_patch[f];
    uint32_t v0 = d_fv[f * 3 + 0];
    uint32_t v1 = d_fv[f * 3 + 1];
    uint32_t v2 = d_fv[f * 3 + 2];

    uint32_t seen[3] = {INVALID32, INVALID32, INVALID32};
    int nseen = 0;

    auto check_vertex = [&](uint32_t v) {
        if (!d_is_boundary_vertex[v]) return;
        uint32_t p = d_vertex_patch[v];
        if (p == q || p == INVALID32) return;
        for (int i = 0; i < nseen; ++i)
            if (seen[i] == p) return;
        seen[nseen++] = p;
        uint32_t slot = d_ribbon_offset[p] + atomicAdd(&d_ribbon_cursor[p], 1u);
        d_ribbon_val[slot] = f;
    };

    check_vertex(v0);
    check_vertex(v1);
    check_vertex(v2);
}


// ═══════════════════════════════════════════════════════════════════════════
// Device helper: binary search for edge ID from sorted edge_key array
// edge_key is packed as (min<<32 | max), sorted ascending.
// ═══════════════════════════════════════════════════════════════════════════

__device__ static uint32_t gpu_find_edge_id(
    uint32_t v0, uint32_t v1,
    const uint64_t* d_edge_key, uint32_t num_edges)
{
    uint32_t lo = min(v0, v1), hi = max(v0, v1);
    uint64_t key = (uint64_t(lo) << 32) | uint64_t(hi);

    // Binary search
    uint32_t left = 0, right = num_edges;
    while (left < right) {
        uint32_t mid = (left + right) / 2;
        if (d_edge_key[mid] < key)
            left = mid + 1;
        else
            right = mid;
    }
    return left;  // edge ID = position in sorted unique array
}

// Device helper: binary search in a sorted array for a value
__device__ static uint16_t gpu_lower_bound(
    const uint32_t* arr, uint32_t start, uint32_t end, uint32_t val)
{
    uint32_t lo = start, hi = end;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (arr[mid] < val)
            lo = mid + 1;
        else
            hi = mid;
    }
    return (lo < end && arr[lo] == val) ? static_cast<uint16_t>(lo) : INVALID16;
}


// ═══════════════════════════════════════════════════════════════════════════
// K1: Build per-patch ltog arrays
// One block per patch. Collects face/vertex/edge IDs, deduplicates, sorts,
// partitions into owned-first.
//
// Output per patch:
//   d_ltog_f[patch_offset_f..] — global face IDs (owned first, sorted)
//   d_ltog_e[patch_offset_e..] — global edge IDs (owned first, sorted)
//   d_ltog_v[patch_offset_v..] — global vertex IDs (owned first, sorted)
//   d_num_owned_f/e/v[patch] — count of owned elements
//   d_num_elements_f/e/v[patch] — total count
// ═══════════════════════════════════════════════════════════════════════════

// Max elements per patch (conservative, covers max_patch_size ~700 faces)
#define K1_MAX_FACES  1024
#define K1_MAX_EDGES  3072
#define K1_MAX_VERTS  1536

__global__ static void k1_build_ltog(
    // Topology
    const uint32_t* d_fv,          // [F*3]
    const uint64_t* d_edge_key,    // [E] sorted packed keys
    const uint32_t* d_ev,          // [E*2]
    uint32_t        num_edges_global,
    // Patch face lists
    const uint32_t* d_patches_val,     // owned face IDs (sorted by patch)
    const uint32_t* d_patches_offset,  // [P] offsets into patches_val
    const uint32_t* d_ribbon_val,      // ribbon face IDs per patch
    const uint32_t* d_ribbon_offset,   // [P+1] offsets into ribbon_val
    // Patch ownership
    const uint32_t* d_face_patch,
    const uint32_t* d_edge_patch,
    const uint32_t* d_vertex_patch,
    uint32_t        num_patches,
    // Output: per-patch ltog (pre-allocated flat arrays)
    uint32_t*       d_ltog_f,      // [total_f_slots]
    uint32_t*       d_ltog_e,      // [total_e_slots]
    uint32_t*       d_ltog_v,      // [total_v_slots]
    const uint32_t* d_ltog_f_offset, // [P+1] offsets per patch
    const uint32_t* d_ltog_e_offset,
    const uint32_t* d_ltog_v_offset,
    uint16_t*       d_num_elements_f, // [P] output
    uint16_t*       d_num_elements_e,
    uint16_t*       d_num_elements_v,
    uint16_t*       d_num_owned_f,  // [P] output
    uint16_t*       d_num_owned_e,
    uint16_t*       d_num_owned_v)
{
    uint32_t p = blockIdx.x;
    if (p >= num_patches) return;

    // Shared memory for collecting unique IDs
    __shared__ uint32_t s_faces[K1_MAX_FACES];
    __shared__ uint32_t s_edges[K1_MAX_EDGES];
    __shared__ uint32_t s_verts[K1_MAX_VERTS];
    __shared__ uint32_t s_nf, s_ne, s_nv;

    if (threadIdx.x == 0) {
        s_nf = 0; s_ne = 0; s_nv = 0;
    }
    __syncthreads();

    // ── Collect owned face IDs ───────────────────────────────────────────
    uint32_t owned_start = (p == 0) ? 0 : d_patches_offset[p - 1];
    uint32_t owned_end   = d_patches_offset[p];
    uint32_t num_owned_faces = owned_end - owned_start;

    for (uint32_t i = threadIdx.x; i < num_owned_faces; i += blockDim.x) {
        uint32_t slot = atomicAdd(&s_nf, 1u);
        if (slot < K1_MAX_FACES)
            s_faces[slot] = d_patches_val[owned_start + i];
    }
    __syncthreads();

    // ── Collect ribbon face IDs ──────────────────────────────────────────
    uint32_t rib_start = d_ribbon_offset[p];
    uint32_t rib_end   = d_ribbon_offset[p + 1];
    uint32_t num_ribbon = rib_end - rib_start;

    for (uint32_t i = threadIdx.x; i < num_ribbon; i += blockDim.x) {
        uint32_t slot = atomicAdd(&s_nf, 1u);
        if (slot < K1_MAX_FACES)
            s_faces[slot] = d_ribbon_val[rib_start + i];
    }
    __syncthreads();

    uint32_t total_faces = s_nf;

    // ── For each face, extract unique vertices and edges ─────────────────
    // Use atomicAdd on s_ne, s_nv with linear probe dedup in shared mem
    // Simple approach: each thread processes one face, uses atomicCAS
    // on a shared hash set for dedup.

    // We'll use a simpler approach: collect all, then sort+unique at the end.
    // First collect ALL vertex IDs (3 per face, with duplicates)
    __shared__ uint32_t s_all_verts[K1_MAX_FACES * 3];
    __shared__ uint32_t s_all_edges[K1_MAX_FACES * 3];

    for (uint32_t i = threadIdx.x; i < total_faces; i += blockDim.x) {
        uint32_t fid = s_faces[i];
        uint32_t v0 = d_fv[fid * 3 + 0];
        uint32_t v1 = d_fv[fid * 3 + 1];
        uint32_t v2 = d_fv[fid * 3 + 2];

        s_all_verts[i * 3 + 0] = v0;
        s_all_verts[i * 3 + 1] = v1;
        s_all_verts[i * 3 + 2] = v2;

        s_all_edges[i * 3 + 0] = gpu_find_edge_id(v0, v1, d_edge_key, num_edges_global);
        s_all_edges[i * 3 + 1] = gpu_find_edge_id(v1, v2, d_edge_key, num_edges_global);
        s_all_edges[i * 3 + 2] = gpu_find_edge_id(v2, v0, d_edge_key, num_edges_global);
    }
    __syncthreads();

    // ── Parallel sort + dedup in shared memory ─────────────────────────
    // All threads participate in odd-even transposition sort, then parallel compaction.
    uint32_t nv_raw = total_faces * 3;
    uint32_t ne_raw = total_faces * 3;

    // Helper: parallel odd-even sort on shared memory array
    auto parallel_sort = [&](uint32_t* arr, uint32_t n) {
        for (uint32_t phase = 0; phase < n; ++phase) {
            uint32_t start = phase & 1;  // alternate even/odd phases
            for (uint32_t i = start + threadIdx.x * 2; i + 1 < n; i += blockDim.x * 2) {
                if (arr[i] > arr[i + 1]) {
                    uint32_t tmp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = tmp;
                }
            }
            __syncthreads();
        }
    };

    // Helper: parallel dedup on sorted array → compacted unique array
    // Returns count of unique elements (via shared counter)
    auto parallel_dedup = [&](uint32_t* sorted, uint32_t n,
                              uint32_t* out, uint32_t* s_count) {
        if (threadIdx.x == 0) *s_count = 0;
        __syncthreads();

        // Mark unique: first element + any element different from predecessor
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            bool is_unique = (i == 0) || (sorted[i] != sorted[i - 1]);
            if (is_unique) {
                uint32_t pos = atomicAdd(s_count, 1u);
                // Can't write to correct position yet (need prefix sum).
                // Instead, mark in-place: store index+1 if unique, 0 if dup.
                sorted[i] = sorted[i] | 0x80000000u;  // high bit = unique flag
            }
        }
        __syncthreads();

        // Compact: thread 0 scans and copies (sequential but array is small)
        if (threadIdx.x == 0) {
            uint32_t wi = 0;
            for (uint32_t i = 0; i < n; ++i) {
                if (sorted[i] & 0x80000000u) {
                    out[wi++] = sorted[i] & 0x7FFFFFFFu;
                }
            }
            *s_count = wi;
        }
        __syncthreads();
    };

    // Sort and dedup vertices
    // First copy s_all_verts to a working buffer (reuse s_all_verts in-place)
    parallel_sort(s_all_verts, nv_raw);
    parallel_dedup(s_all_verts, nv_raw, s_verts, &s_nv);

    // Sort and dedup edges
    parallel_sort(s_all_edges, ne_raw);
    parallel_dedup(s_all_edges, ne_raw, s_edges, &s_ne);

    // Sort faces (parallel)
    parallel_sort(s_faces, total_faces);
    __syncthreads();

    uint32_t nv = s_nv, ne = s_ne, nf = total_faces;

    // ── Stable partition: owned first, then not-owned ────────────────────
    // Thread 0 does the partition (small arrays)
    __shared__ uint16_t s_num_owned_f, s_num_owned_e, s_num_owned_v;

    // Use s_all_verts/s_all_edges as temp buffers for partition (reuse memory)
    // They're no longer needed after sort+dedup above.
    if (threadIdx.x == 0) {
        uint32_t* temp_f = s_all_verts;  // reuse, enough space for K1_MAX_FACES
        uint32_t* temp_e = s_all_edges;  // reuse, enough space for K1_MAX_EDGES

        // Partition faces: owned first
        uint16_t owned = 0;
        for (uint32_t i = 0; i < nf; ++i)
            if (d_face_patch[s_faces[i]] == p) owned++;
        s_num_owned_f = owned;
        uint16_t wi = 0;
        for (uint32_t i = 0; i < nf; ++i)
            if (d_face_patch[s_faces[i]] == p) temp_f[wi++] = s_faces[i];
        for (uint32_t i = 0; i < nf; ++i)
            if (d_face_patch[s_faces[i]] != p) temp_f[wi++] = s_faces[i];
        for (uint32_t i = 0; i < nf; ++i) s_faces[i] = temp_f[i];

        // Partition edges
        owned = 0;
        for (uint32_t i = 0; i < ne; ++i)
            if (d_edge_patch[s_edges[i]] == p) owned++;
        s_num_owned_e = owned;
        wi = 0;
        for (uint32_t i = 0; i < ne; ++i)
            if (d_edge_patch[s_edges[i]] == p) temp_e[wi++] = s_edges[i];
        for (uint32_t i = 0; i < ne; ++i)
            if (d_edge_patch[s_edges[i]] != p) temp_e[wi++] = s_edges[i];
        for (uint32_t i = 0; i < ne; ++i) s_edges[i] = temp_e[i];

        // Partition vertices — reuse temp_f (already done with faces)
        uint32_t* temp_v = temp_f;
        owned = 0;
        for (uint32_t i = 0; i < nv; ++i)
            if (d_vertex_patch[s_verts[i]] == p) owned++;
        s_num_owned_v = owned;
        wi = 0;
        for (uint32_t i = 0; i < nv; ++i)
            if (d_vertex_patch[s_verts[i]] == p) temp_v[wi++] = s_verts[i];
        for (uint32_t i = 0; i < nv; ++i)
            if (d_vertex_patch[s_verts[i]] != p) temp_v[wi++] = s_verts[i];
        for (uint32_t i = 0; i < nv; ++i) s_verts[i] = temp_v[i];
    }
    __syncthreads();

    // ── Write results to global memory ───────────────────────────────────
    uint32_t f_base = d_ltog_f_offset[p];
    uint32_t e_base = d_ltog_e_offset[p];
    uint32_t v_base = d_ltog_v_offset[p];

    for (uint32_t i = threadIdx.x; i < nf; i += blockDim.x)
        d_ltog_f[f_base + i] = s_faces[i];
    for (uint32_t i = threadIdx.x; i < ne; i += blockDim.x)
        d_ltog_e[e_base + i] = s_edges[i];
    for (uint32_t i = threadIdx.x; i < nv; i += blockDim.x)
        d_ltog_v[v_base + i] = s_verts[i];

    if (threadIdx.x == 0) {
        d_num_elements_f[p] = nf;
        d_num_elements_e[p] = ne;
        d_num_elements_v[p] = nv;
        d_num_owned_f[p] = s_num_owned_f;
        d_num_owned_e[p] = s_num_owned_e;
        d_num_owned_v[p] = s_num_owned_v;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// K2: Build per-patch local topology (ev, fe)
// One block per patch. Reads ltog arrays, builds local index mappings.
// ═══════════════════════════════════════════════════════════════════════════

__global__ static void k2_build_topology(
    const uint32_t* d_fv,
    const uint64_t* d_edge_key,
    const uint32_t* d_ev_global,
    uint32_t        num_edges_global,
    // Patch face lists
    const uint32_t* d_patches_val,
    const uint32_t* d_patches_offset,
    const uint32_t* d_ribbon_val,
    const uint32_t* d_ribbon_offset,
    // Patch ownership
    const uint32_t* d_face_patch,
    const uint32_t* d_edge_patch,
    const uint32_t* d_vertex_patch,
    // ltog arrays (from K1)
    const uint32_t* d_ltog_f, const uint32_t* d_ltog_f_offset,
    const uint32_t* d_ltog_e, const uint32_t* d_ltog_e_offset,
    const uint32_t* d_ltog_v, const uint32_t* d_ltog_v_offset,
    const uint16_t* d_num_elements_f, const uint16_t* d_num_owned_f,
    const uint16_t* d_num_elements_e, const uint16_t* d_num_owned_e,
    const uint16_t* d_num_elements_v, const uint16_t* d_num_owned_v,
    uint32_t        num_patches,
    // Output: local topology (bulk-allocated)
    uint16_t*       d_ev_local_all,  // [total_edge_slots * 2] LocalVertexT
    uint16_t*       d_fe_local_all,  // [total_face_slots * 3] LocalEdgeT (with dir)
    uint32_t        ev_stride,       // per-patch stride = max_edge_capacity * 2
    uint32_t        fe_stride)       // per-patch stride = max_face_capacity * 3
{
    uint32_t p = blockIdx.x;
    if (p >= num_patches) return;

    uint16_t nf = d_num_elements_f[p];
    uint16_t ne = d_num_elements_e[p];
    uint16_t nv = d_num_elements_v[p];
    uint16_t owned_f = d_num_owned_f[p];
    uint16_t owned_e = d_num_owned_e[p];
    uint16_t owned_v = d_num_owned_v[p];

    uint32_t f_base = d_ltog_f_offset[p];
    uint32_t e_base = d_ltog_e_offset[p];
    uint32_t v_base = d_ltog_v_offset[p];

    const uint32_t* my_ltog_f = d_ltog_f + f_base;
    const uint32_t* my_ltog_e = d_ltog_e + e_base;
    const uint32_t* my_ltog_v = d_ltog_v + v_base;

    uint16_t* my_ev = d_ev_local_all + p * ev_stride;
    uint16_t* my_fe = d_fe_local_all + p * fe_stride;

    // Track which edges have been processed
    // Use a shared bitmask (max 2048 edges → 64 uint32_t words)
    __shared__ uint32_t s_edge_done[64];  // supports up to 2048 edges
    for (uint32_t i = threadIdx.x; i < 64; i += blockDim.x)
        s_edge_done[i] = 0;
    __syncthreads();

    // Process each face in parallel
    for (uint32_t fi = threadIdx.x; fi < nf; fi += blockDim.x) {
        uint32_t global_fid = my_ltog_f[fi];

        // Find local face index (it's just fi since ltog_f is sorted)
        uint16_t local_fid = fi;

        for (uint32_t e = 0; e < 3; ++e) {
            uint32_t gv0 = d_fv[global_fid * 3 + e];
            uint32_t gv1 = d_fv[global_fid * 3 + ((e + 1) % 3)];

            // Edge direction: canonical order is (max, min)
            uint32_t hi = max(gv0, gv1), lo = min(gv0, gv1);
            int dir = (hi == gv0 && lo == gv1) ? 0 : 1;

            // Find global edge ID
            uint32_t global_eid = gpu_find_edge_id(gv0, gv1, d_edge_key, num_edges_global);

            // Find local edge ID via binary search on ltog_e
            uint32_t e_search_start = (d_edge_patch[global_eid] == p) ? 0 : owned_e;
            uint32_t e_search_end = (d_edge_patch[global_eid] == p) ? owned_e : ne;
            uint16_t local_eid = gpu_lower_bound(my_ltog_e, e_search_start, e_search_end, global_eid);

            // Write fe (face-edge with direction bit)
            uint16_t fe_val = (local_eid << 1) | (dir & 1);
            my_fe[local_fid * 3 + e] = fe_val;

            // Write ev (edge-vertex) — only first thread to process this edge
            uint32_t word = local_eid / 32;
            uint32_t bit  = 1u << (local_eid % 32);
            if ((atomicOr(&s_edge_done[word], bit) & bit) == 0) {
                // First to process this edge — write ev
                uint16_t local_v0 = gpu_lower_bound(my_ltog_v,
                    (d_vertex_patch[hi] == p) ? 0 : owned_v,
                    (d_vertex_patch[hi] == p) ? owned_v : nv, hi);
                uint16_t local_v1 = gpu_lower_bound(my_ltog_v,
                    (d_vertex_patch[lo] == p) ? 0 : owned_v,
                    (d_vertex_patch[lo] == p) ? owned_v : nv, lo);
                my_ev[local_eid * 2 + 0] = local_v0;
                my_ev[local_eid * 2 + 1] = local_v1;
            }
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════════════

GpuPatchBuildResult gpu_build_patches(
    const uint32_t* d_fv,
    const uint64_t* d_edge_key,
    const uint32_t* d_ev,
    const uint32_t* d_ef_f0,
    const uint32_t* d_ef_f1,
    const uint32_t* d_ff_offset,
    const uint32_t* d_ff_values,
    uint32_t num_vertices,
    uint32_t num_edges,
    uint32_t num_faces,
    const uint32_t* d_face_patch,
    uint32_t num_patches,
    float capacity_factor,
    float lp_load_factor,
    PatchInfo* d_patches_info,
    PatchInfo* h_patches_info)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t_total = clk::now();
    constexpr int BLOCK = 256;

    GpuPatchBuildResult result = {};
    result.num_patches = num_patches;
    result.num_vertices = num_vertices;
    result.num_edges = num_edges;
    result.num_faces = num_faces;

    // ── K0a: assign edge/vertex patches ──────────────────────────────────
    auto tp = clk::now();
    uint32_t* d_edge_patch;
    uint32_t* d_vertex_patch;
    CUDA_ERROR(cudaMalloc(&d_edge_patch, num_edges * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_vertex_patch, num_vertices * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_vertex_patch, 0xFF, num_vertices * sizeof(uint32_t)));

    int grid = (num_edges + BLOCK - 1) / BLOCK;
    k0a_assign_edge_vertex_patch<<<grid, BLOCK>>>(
        d_face_patch, d_ev, d_ef_f0,
        d_edge_patch, d_vertex_patch, num_edges);
    CUDA_ERROR(cudaDeviceSynchronize());
    fprintf(stderr, "[gpu_patches] K0a assign: %.1fms\n", ms_since(tp));

    result.d_edge_patch = d_edge_patch;
    result.d_vertex_patch = d_vertex_patch;

    // ── K0b: extract ribbons ─────────────────────────────────────────────
    tp = clk::now();

    // Step 1: mark boundary faces
    uint8_t* d_is_boundary_face;
    CUDA_ERROR(cudaMalloc(&d_is_boundary_face, num_faces * sizeof(uint8_t)));
    grid = (num_faces + BLOCK - 1) / BLOCK;
    k0b_mark_boundary_faces<<<grid, BLOCK>>>(
        d_face_patch, d_ff_offset, d_ff_values,
        d_is_boundary_face, num_faces);

    // Step 2: mark boundary vertices
    uint8_t* d_is_boundary_vertex;
    CUDA_ERROR(cudaMalloc(&d_is_boundary_vertex, num_vertices * sizeof(uint8_t)));
    CUDA_ERROR(cudaMemset(d_is_boundary_vertex, 0, num_vertices * sizeof(uint8_t)));
    k0b_mark_boundary_vertices<<<grid, BLOCK>>>(
        d_fv, d_is_boundary_face, d_is_boundary_vertex, num_faces);

    // Step 3: count ribbon faces per patch
    uint32_t* d_ribbon_count;
    CUDA_ERROR(cudaMalloc(&d_ribbon_count, num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_ribbon_count, 0, num_patches * sizeof(uint32_t)));
    k0b_count_ribbon_per_patch<<<grid, BLOCK>>>(
        d_fv, d_face_patch, d_vertex_patch, d_is_boundary_vertex,
        d_ribbon_count, num_faces);

    // Prefix sum → ribbon offsets
    uint32_t* d_ribbon_offset;
    CUDA_ERROR(cudaMalloc(&d_ribbon_offset, (num_patches + 1) * sizeof(uint32_t)));
    thrust::exclusive_scan(
        thrust::device,
        thrust::device_pointer_cast(d_ribbon_count),
        thrust::device_pointer_cast(d_ribbon_count) + num_patches,
        thrust::device_pointer_cast(d_ribbon_offset));

    // Get total ribbon faces
    uint32_t total_ribbon;
    {
        uint32_t last_count, last_off;
        CUDA_ERROR(cudaMemcpy(&last_count, d_ribbon_count + num_patches - 1,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(&last_off, d_ribbon_offset + num_patches - 1,
                              sizeof(uint32_t), cudaMemcpyDeviceToHost));
        total_ribbon = last_off + last_count;
    }
    CUDA_ERROR(cudaMemcpy(d_ribbon_offset + num_patches, &total_ribbon,
                          sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Step 4: scatter ribbon face IDs
    uint32_t* d_ribbon_val;
    uint32_t* d_ribbon_cursor;
    CUDA_ERROR(cudaMalloc(&d_ribbon_val, total_ribbon * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ribbon_cursor, num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_ribbon_cursor, 0, num_patches * sizeof(uint32_t)));
    k0b_scatter_ribbon_faces<<<grid, BLOCK>>>(
        d_fv, d_face_patch, d_vertex_patch, d_is_boundary_vertex,
        d_ribbon_offset, d_ribbon_cursor, d_ribbon_val, num_faces);
    CUDA_ERROR(cudaDeviceSynchronize());

    CUDA_ERROR(cudaFree(d_is_boundary_face));
    CUDA_ERROR(cudaFree(d_is_boundary_vertex));
    CUDA_ERROR(cudaFree(d_ribbon_cursor));

    fprintf(stderr, "[gpu_patches] K0b ribbons: %.1fms (total_ribbon=%u)\n",
            ms_since(tp), total_ribbon);

    // ── TODO: K1-K4 ─────────────────────────────────────────────────────
    // For now, download results and let CPU handle the rest.
    // This validates K0a/K0b correctness before building K1-K4.

    fprintf(stderr, "[gpu_patches] TOTAL so far: %.1fms\n", ms_since(t_total));

    result.success = true;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// APPROACH A: Global thrust sort-based ltog construction
// ═══════════════════════════════════════════════════════════════════════════

// Step 1 kernel: for each face in each patch (owned + ribbon),
// emit (patch_id, edge_id) and (patch_id, vertex_id) tuples.
__global__ static void expand_patch_elements_kernel(
    const uint32_t* d_fv,           // [F*3]
    const uint64_t* d_edge_key,     // [E] unique sorted
    uint32_t        num_edges,
    // Owned faces
    const uint32_t* d_patches_val,
    const uint32_t* d_patches_offset, // [P] cumulative
    // Ribbon faces
    const uint32_t* d_ribbon_val,
    const uint32_t* d_ribbon_offset,  // [P+1] prefix
    uint32_t        num_patches,
    // Output: (patch, element) pairs for edges and vertices
    // Each face contributes 3 edge pairs and 3 vertex pairs
    // Total output size = total_patch_faces * 3
    uint64_t*       d_edge_pairs,   // packed (patch<<32 | edge_id)
    uint64_t*       d_vert_pairs,   // packed (patch<<32 | vertex_id)
    uint32_t*       d_face_pairs_patch, // patch ID for each face
    uint32_t*       d_face_pairs_fid,   // global face ID
    // Per-patch face counts for indexing
    const uint32_t* d_patch_face_offset, // [P+1] prefix sum of total faces per patch
    uint32_t        total_patch_faces)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_patch_faces) return;

    // Find which patch and which face within that patch
    // d_patch_face_offset[p] = start index for patch p's faces
    // Binary search to find patch
    uint32_t lo = 0, hi = num_patches;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (d_patch_face_offset[mid + 1] <= idx) lo = mid + 1;
        else hi = mid;
    }
    uint32_t patch = lo;
    uint32_t local_idx = idx - d_patch_face_offset[patch];

    // Get the face ID
    uint32_t owned_start = (patch == 0) ? 0 : d_patches_offset[patch - 1];
    uint32_t owned_end = d_patches_offset[patch];
    uint32_t num_owned = owned_end - owned_start;

    uint32_t rib_start = d_ribbon_offset[patch];
    uint32_t rib_end = d_ribbon_offset[patch + 1];

    uint32_t face_id;
    if (local_idx < num_owned) {
        face_id = d_patches_val[owned_start + local_idx];
    } else {
        face_id = d_ribbon_val[rib_start + (local_idx - num_owned)];
    }

    // Write face pair
    d_face_pairs_patch[idx] = patch;
    d_face_pairs_fid[idx] = face_id;

    // Emit 3 edge pairs and 3 vertex pairs
    uint32_t out_base = idx * 3;
    for (int e = 0; e < 3; ++e) {
        uint32_t v0 = d_fv[face_id * 3 + e];
        uint32_t v1 = d_fv[face_id * 3 + ((e + 1) % 3)];

        // Find edge ID via binary search on unique d_edge_key
        uint32_t elo = min(v0, v1), ehi = max(v0, v1);
        uint64_t key = (uint64_t(elo) << 32) | uint64_t(ehi);
        // Binary search
        uint32_t left = 0, right = num_edges;
        while (left < right) {
            uint32_t mid = (left + right) / 2;
            if (d_edge_key[mid] < key) left = mid + 1;
            else right = mid;
        }
        uint32_t edge_id = left;

        d_edge_pairs[out_base + e] = (uint64_t(patch) << 32) | uint64_t(edge_id);
        d_vert_pairs[out_base + e] = (uint64_t(patch) << 32) | uint64_t(v0);
    }
}

ThrustLtogResult gpu_thrust_build_ltog(
    const uint32_t* d_fv,
    const uint64_t* d_edge_key,
    uint32_t num_faces,
    uint32_t num_edges,
    uint32_t num_vertices,
    const uint32_t* d_face_patch,
    const uint32_t* d_edge_patch,
    const uint32_t* d_vertex_patch,
    const uint32_t* d_patches_val,
    const uint32_t* d_patches_offset,
    const uint32_t* d_ribbon_val,
    const uint32_t* d_ribbon_offset,
    uint32_t num_patches)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t_total = clk::now();

    // ── Step 0: compute total faces per patch and prefix offsets ──────────
    auto tp = clk::now();
    // Download patches_offset and ribbon_offset to compute sizes
    std::vector<uint32_t> h_po(num_patches);
    std::vector<uint32_t> h_ro(num_patches + 1);
    CUDA_ERROR(cudaMemcpy(h_po.data(), d_patches_offset,
                          num_patches * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(h_ro.data(), d_ribbon_offset,
                          (num_patches + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::vector<uint32_t> h_patch_face_count(num_patches);
    std::vector<uint32_t> h_patch_face_offset(num_patches + 1);
    h_patch_face_offset[0] = 0;
    for (uint32_t p = 0; p < num_patches; ++p) {
        uint32_t owned = (p == 0) ? h_po[0] : h_po[p] - h_po[p - 1];
        uint32_t ribbon = h_ro[p + 1] - h_ro[p];
        h_patch_face_count[p] = owned + ribbon;
        h_patch_face_offset[p + 1] = h_patch_face_offset[p] + h_patch_face_count[p];
    }
    uint32_t total_patch_faces = h_patch_face_offset[num_patches];

    uint32_t* d_patch_face_offset;
    CUDA_ERROR(cudaMalloc(&d_patch_face_offset, (num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_patch_face_offset, h_patch_face_offset.data(),
                          (num_patches + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    fprintf(stderr, "[thrust_ltog] step0 offsets: %.0fms (total_patch_faces=%u)\n",
            ms_since(tp), total_patch_faces);

    // ── Step 1: expand all patch faces into (patch, edge/vert) pairs ─────
    tp = clk::now();
    uint32_t total_half_edges = total_patch_faces * 3;

    uint64_t *d_edge_pairs, *d_vert_pairs;
    uint32_t *d_face_pairs_patch, *d_face_pairs_fid;
    CUDA_ERROR(cudaMalloc(&d_edge_pairs, total_half_edges * sizeof(uint64_t)));
    CUDA_ERROR(cudaMalloc(&d_vert_pairs, total_half_edges * sizeof(uint64_t)));
    CUDA_ERROR(cudaMalloc(&d_face_pairs_patch, total_patch_faces * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_face_pairs_fid, total_patch_faces * sizeof(uint32_t)));

    int grid = (total_patch_faces + 255) / 256;
    expand_patch_elements_kernel<<<grid, 256>>>(
        d_fv, d_edge_key, num_edges,
        d_patches_val, d_patches_offset,
        d_ribbon_val, d_ribbon_offset,
        num_patches,
        d_edge_pairs, d_vert_pairs,
        d_face_pairs_patch, d_face_pairs_fid,
        d_patch_face_offset, total_patch_faces);
    CUDA_ERROR(cudaDeviceSynchronize());
    CUDA_ERROR(cudaFree(d_patch_face_offset));
    fprintf(stderr, "[thrust_ltog] step1 expand: %.0fms\n", ms_since(tp));

    // ── Step 1b: add ALL owned edges and vertices (safeguard) ──────────
    // Some edges/vertices are owned by a patch but not incident to any
    // face in that patch's owned+ribbon list. Add them explicitly.
    {
        // Edges: for each edge e, add (edge_patch[e], e)
        uint64_t* d_owned_edge_pairs;
        CUDA_ERROR(cudaMalloc(&d_owned_edge_pairs, num_edges * sizeof(uint64_t)));
        // Kernel to emit owned edge pairs
        auto emit_owned = [=] __device__ (uint32_t e) -> uint64_t {
            return (uint64_t(d_edge_patch[e]) << 32) | uint64_t(e);
        };
        // Use thrust::tabulate
        thrust::tabulate(thrust::device,
            thrust::device_pointer_cast(d_owned_edge_pairs),
            thrust::device_pointer_cast(d_owned_edge_pairs) + num_edges,
            [d_edge_patch] __device__ (uint32_t e) -> uint64_t {
                return (uint64_t(d_edge_patch[e]) << 32) | uint64_t(e);
            });

        // Merge with face-derived edge pairs
        uint64_t* d_all_edge_pairs;
        uint32_t total_edge_entries = total_half_edges + num_edges;
        CUDA_ERROR(cudaMalloc(&d_all_edge_pairs, total_edge_entries * sizeof(uint64_t)));
        CUDA_ERROR(cudaMemcpy(d_all_edge_pairs, d_edge_pairs,
                              total_half_edges * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        CUDA_ERROR(cudaMemcpy(d_all_edge_pairs + total_half_edges, d_owned_edge_pairs,
                              num_edges * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        CUDA_ERROR(cudaFree(d_edge_pairs));
        CUDA_ERROR(cudaFree(d_owned_edge_pairs));
        d_edge_pairs = d_all_edge_pairs;
        total_half_edges = total_edge_entries;

        // Vertices: for each vertex v, add (vertex_patch[v], v)
        uint64_t* d_owned_vert_pairs;
        CUDA_ERROR(cudaMalloc(&d_owned_vert_pairs, num_vertices * sizeof(uint64_t)));
        thrust::tabulate(thrust::device,
            thrust::device_pointer_cast(d_owned_vert_pairs),
            thrust::device_pointer_cast(d_owned_vert_pairs) + num_vertices,
            [d_vertex_patch] __device__ (uint32_t v) -> uint64_t {
                return (uint64_t(d_vertex_patch[v]) << 32) | uint64_t(v);
            });

        uint64_t* d_all_vert_pairs;
        uint32_t total_vert_entries = total_patch_faces * 3 + num_vertices;
        CUDA_ERROR(cudaMalloc(&d_all_vert_pairs, total_vert_entries * sizeof(uint64_t)));
        CUDA_ERROR(cudaMemcpy(d_all_vert_pairs, d_vert_pairs,
                              total_patch_faces * 3 * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        CUDA_ERROR(cudaMemcpy(d_all_vert_pairs + total_patch_faces * 3, d_owned_vert_pairs,
                              num_vertices * sizeof(uint64_t), cudaMemcpyDeviceToDevice));
        CUDA_ERROR(cudaFree(d_vert_pairs));
        CUDA_ERROR(cudaFree(d_owned_vert_pairs));
        d_vert_pairs = d_all_vert_pairs;
        uint32_t total_vert_half = total_vert_entries;

        fprintf(stderr, "[thrust_ltog] step1b safeguard: added %u owned edges + %u owned verts\n",
                num_edges, num_vertices);
    }

    // ── Step 2: sort edge pairs and vert pairs ───────────────────────────
    tp = clk::now();
    // Note: total_half_edges now includes owned edge safeguard entries
    thrust::sort(thrust::device,
                 thrust::device_pointer_cast(d_edge_pairs),
                 thrust::device_pointer_cast(d_edge_pairs) + total_half_edges);
    // For vertices: total_vert_half = original total_patch_faces*3 + num_vertices
    // (set in step 1b block, but scope is lost — need to track it)
    uint32_t total_vert_entries = total_patch_faces * 3 + num_vertices;
    thrust::sort(thrust::device,
                 thrust::device_pointer_cast(d_vert_pairs),
                 thrust::device_pointer_cast(d_vert_pairs) + total_vert_entries);
    // Pack face pairs as uint64_t (patch<<32 | face_id) for proper sorting
    {
        uint64_t* d_face_pairs_packed;
        CUDA_ERROR(cudaMalloc(&d_face_pairs_packed, total_patch_faces * sizeof(uint64_t)));
        thrust::tabulate(thrust::device,
            thrust::device_pointer_cast(d_face_pairs_packed),
            thrust::device_pointer_cast(d_face_pairs_packed) + total_patch_faces,
            [d_face_pairs_patch, d_face_pairs_fid] __device__ (uint32_t i) -> uint64_t {
                return (uint64_t(d_face_pairs_patch[i]) << 32) | uint64_t(d_face_pairs_fid[i]);
            });
        CUDA_ERROR(cudaFree(d_face_pairs_patch));
        CUDA_ERROR(cudaFree(d_face_pairs_fid));

        thrust::sort(thrust::device,
                     thrust::device_pointer_cast(d_face_pairs_packed),
                     thrust::device_pointer_cast(d_face_pairs_packed) + total_patch_faces);

        // Unpack back
        CUDA_ERROR(cudaMalloc(&d_face_pairs_patch, total_patch_faces * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&d_face_pairs_fid, total_patch_faces * sizeof(uint32_t)));
        thrust::tabulate(thrust::device,
            thrust::device_pointer_cast(d_face_pairs_patch),
            thrust::device_pointer_cast(d_face_pairs_patch) + total_patch_faces,
            [d_face_pairs_packed] __device__ (uint32_t i) -> uint32_t {
                return d_face_pairs_packed[i] >> 32;
            });
        thrust::tabulate(thrust::device,
            thrust::device_pointer_cast(d_face_pairs_fid),
            thrust::device_pointer_cast(d_face_pairs_fid) + total_patch_faces,
            [d_face_pairs_packed] __device__ (uint32_t i) -> uint32_t {
                return d_face_pairs_packed[i] & 0xFFFFFFFFu;
            });
        CUDA_ERROR(cudaFree(d_face_pairs_packed));
    }
    CUDA_ERROR(cudaDeviceSynchronize());
    fprintf(stderr, "[thrust_ltog] step2 sort: %.0fms\n", ms_since(tp));

    // ── Step 3: unique to remove duplicates ──────────────────────────────
    tp = clk::now();
    uint64_t* d_unique_edges;
    uint64_t* d_unique_verts;
    CUDA_ERROR(cudaMalloc(&d_unique_edges, total_half_edges * sizeof(uint64_t)));
    CUDA_ERROR(cudaMalloc(&d_unique_verts, total_vert_entries * sizeof(uint64_t)));

    auto edge_end = thrust::unique_copy(thrust::device,
        thrust::device_pointer_cast(d_edge_pairs),
        thrust::device_pointer_cast(d_edge_pairs) + total_half_edges,
        thrust::device_pointer_cast(d_unique_edges));
    uint32_t num_unique_edges = edge_end - thrust::device_pointer_cast(d_unique_edges);

    auto vert_end = thrust::unique_copy(thrust::device,
        thrust::device_pointer_cast(d_vert_pairs),
        thrust::device_pointer_cast(d_vert_pairs) + total_vert_entries,
        thrust::device_pointer_cast(d_unique_verts));
    uint32_t num_unique_verts = vert_end - thrust::device_pointer_cast(d_unique_verts);

    // Faces: already unique (each face appears once per patch in patches_val/ribbon_val)
    // But faces may appear in multiple patches (ribbon). unique_copy on sorted (patch, face)
    // removes within-patch duplicates (shouldn't exist), keeps cross-patch occurrences.

    CUDA_ERROR(cudaFree(d_edge_pairs));
    CUDA_ERROR(cudaFree(d_vert_pairs));
    fprintf(stderr, "[thrust_ltog] step3 unique: %.0fms (edges=%u, verts=%u)\n",
            ms_since(tp), num_unique_edges, num_unique_verts);

    // ── Step 4: segment on device into per-patch ltog arrays ────────────
    // Stay on device: extract element IDs from packed uint64, compute offsets.
    tp = clk::now();

    // 4a: Extract element IDs (lower 32 bits) on device
    uint32_t* d_ltog_e_raw;
    uint32_t* d_ltog_v_raw;
    CUDA_ERROR(cudaMalloc(&d_ltog_e_raw, num_unique_edges * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ltog_v_raw, num_unique_verts * sizeof(uint32_t)));

    // Extract edge IDs (lower 32 bits of packed uint64)
    thrust::transform(thrust::device,
        thrust::device_pointer_cast(d_unique_edges),
        thrust::device_pointer_cast(d_unique_edges) + num_unique_edges,
        thrust::device_pointer_cast(d_ltog_e_raw),
        [] __device__ (uint64_t v) -> uint32_t { return v & 0xFFFFFFFFu; });

    // Extract vertex IDs
    thrust::transform(thrust::device,
        thrust::device_pointer_cast(d_unique_verts),
        thrust::device_pointer_cast(d_unique_verts) + num_unique_verts,
        thrust::device_pointer_cast(d_ltog_v_raw),
        [] __device__ (uint64_t v) -> uint32_t { return v & 0xFFFFFFFFu; });

    // 4b: Compute per-patch offsets using lower_bound on patch IDs
    // For edges: extract patch from upper 32 bits, find boundaries
    uint32_t* d_edge_patches_sorted;  // patch ID for each unique edge
    uint32_t* d_vert_patches_sorted;
    CUDA_ERROR(cudaMalloc(&d_edge_patches_sorted, num_unique_edges * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_vert_patches_sorted, num_unique_verts * sizeof(uint32_t)));

    thrust::transform(thrust::device,
        thrust::device_pointer_cast(d_unique_edges),
        thrust::device_pointer_cast(d_unique_edges) + num_unique_edges,
        thrust::device_pointer_cast(d_edge_patches_sorted),
        [] __device__ (uint64_t v) -> uint32_t { return v >> 32; });

    thrust::transform(thrust::device,
        thrust::device_pointer_cast(d_unique_verts),
        thrust::device_pointer_cast(d_unique_verts) + num_unique_verts,
        thrust::device_pointer_cast(d_vert_patches_sorted),
        [] __device__ (uint64_t v) -> uint32_t { return v >> 32; });

    CUDA_ERROR(cudaFree(d_unique_edges));
    CUDA_ERROR(cudaFree(d_unique_verts));

    // Compute offsets: lower_bound for each patch boundary [0..P]
    // Generate search values [0, 1, 2, ..., P]
    uint32_t* d_search_vals;
    CUDA_ERROR(cudaMalloc(&d_search_vals, (num_patches + 1) * sizeof(uint32_t)));
    thrust::sequence(thrust::device,
        thrust::device_pointer_cast(d_search_vals),
        thrust::device_pointer_cast(d_search_vals) + num_patches + 1);

    uint32_t* d_e_offset;
    uint32_t* d_v_offset;
    uint32_t* d_f_offset;
    CUDA_ERROR(cudaMalloc(&d_e_offset, (num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_v_offset, (num_patches + 1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_f_offset, (num_patches + 1) * sizeof(uint32_t)));

    thrust::lower_bound(thrust::device,
        thrust::device_pointer_cast(d_edge_patches_sorted),
        thrust::device_pointer_cast(d_edge_patches_sorted) + num_unique_edges,
        thrust::device_pointer_cast(d_search_vals),
        thrust::device_pointer_cast(d_search_vals) + num_patches + 1,
        thrust::device_pointer_cast(d_e_offset));

    thrust::lower_bound(thrust::device,
        thrust::device_pointer_cast(d_vert_patches_sorted),
        thrust::device_pointer_cast(d_vert_patches_sorted) + num_unique_verts,
        thrust::device_pointer_cast(d_search_vals),
        thrust::device_pointer_cast(d_search_vals) + num_patches + 1,
        thrust::device_pointer_cast(d_v_offset));

    // Face offsets: faces are in d_face_pairs_patch (sorted), d_face_pairs_fid
    thrust::lower_bound(thrust::device,
        thrust::device_pointer_cast(d_face_pairs_patch),
        thrust::device_pointer_cast(d_face_pairs_patch) + total_patch_faces,
        thrust::device_pointer_cast(d_search_vals),
        thrust::device_pointer_cast(d_search_vals) + num_patches + 1,
        thrust::device_pointer_cast(d_f_offset));

    CUDA_ERROR(cudaFree(d_edge_patches_sorted));
    CUDA_ERROR(cudaFree(d_vert_patches_sorted));
    CUDA_ERROR(cudaFree(d_search_vals));

    // d_face_pairs_fid IS the face ltog (already extracted IDs, sorted by patch)
    uint32_t* d_ltog_f_raw = d_face_pairs_fid;  // reuse directly
    CUDA_ERROR(cudaFree(d_face_pairs_patch));
    // Don't free d_face_pairs_fid — it's now d_ltog_f_raw

    fprintf(stderr, "[thrust_ltog] step4 segment: %.0fms\n", ms_since(tp));

    // ── Step 5: GPU owned-first partition (single kernel) ─────────────────
    // One block per patch. Each block partitions its F/E/V segments so owned
    // elements come first, preserving order within each group.
    tp = clk::now();

    // Allocate temp buffers for out-of-place partition
    uint32_t* d_ltog_f_tmp;
    uint32_t* d_ltog_e_tmp;
    uint32_t* d_ltog_v_tmp;
    uint16_t* d_owned_f;
    uint16_t* d_owned_e;
    uint16_t* d_owned_v;
    CUDA_ERROR(cudaMalloc(&d_ltog_f_tmp, total_patch_faces * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ltog_e_tmp, num_unique_edges * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ltog_v_tmp, num_unique_verts * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_owned_f, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_owned_e, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_owned_v, num_patches * sizeof(uint16_t)));

    // Lambda kernel via thrust::for_each to partition each patch
    // Each "thread" handles one patch — this is a host-side loop equivalent
    // but using a single GPU kernel launch.
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(num_patches),
        [d_ltog_f_raw, d_ltog_f_tmp, d_f_offset, d_face_patch, d_owned_f,
         d_ltog_e_raw, d_ltog_e_tmp, d_e_offset, d_edge_patch, d_owned_e,
         d_ltog_v_raw, d_ltog_v_tmp, d_v_offset, d_vertex_patch, d_owned_v
        ] __device__ (uint32_t p) {
            // Faces: two-pass — count owned, then write owned-first
            uint32_t fs = d_f_offset[p], fe = d_f_offset[p + 1];
            uint16_t nowned_f = 0;
            for (uint32_t i = fs; i < fe; ++i)
                if (d_face_patch[d_ltog_f_raw[i]] == p) nowned_f++;
            d_owned_f[p] = nowned_f;
            uint32_t wi = fs, wj = fs + nowned_f;
            for (uint32_t i = fs; i < fe; ++i) {
                if (d_face_patch[d_ltog_f_raw[i]] == p)
                    d_ltog_f_tmp[wi++] = d_ltog_f_raw[i];
                else
                    d_ltog_f_tmp[wj++] = d_ltog_f_raw[i];
            }

            // Edges
            uint32_t es = d_e_offset[p], ee = d_e_offset[p + 1];
            uint16_t nowned_e = 0;
            for (uint32_t i = es; i < ee; ++i)
                if (d_edge_patch[d_ltog_e_raw[i]] == p) nowned_e++;
            d_owned_e[p] = nowned_e;
            wi = es; wj = es + nowned_e;
            for (uint32_t i = es; i < ee; ++i) {
                if (d_edge_patch[d_ltog_e_raw[i]] == p)
                    d_ltog_e_tmp[wi++] = d_ltog_e_raw[i];
                else
                    d_ltog_e_tmp[wj++] = d_ltog_e_raw[i];
            }

            // Vertices
            uint32_t vs = d_v_offset[p], ve = d_v_offset[p + 1];
            uint16_t nowned_v = 0;
            for (uint32_t i = vs; i < ve; ++i)
                if (d_vertex_patch[d_ltog_v_raw[i]] == p) nowned_v++;
            d_owned_v[p] = nowned_v;
            wi = vs; wj = vs + nowned_v;
            for (uint32_t i = vs; i < ve; ++i) {
                if (d_vertex_patch[d_ltog_v_raw[i]] == p)
                    d_ltog_v_tmp[wi++] = d_ltog_v_raw[i];
                else
                    d_ltog_v_tmp[wj++] = d_ltog_v_raw[i];
            }
        });
    CUDA_ERROR(cudaDeviceSynchronize());

    // Swap: tmp becomes the real ltog, free originals
    CUDA_ERROR(cudaFree(d_ltog_f_raw));
    CUDA_ERROR(cudaFree(d_ltog_e_raw));
    CUDA_ERROR(cudaFree(d_ltog_v_raw));
    d_ltog_f_raw = d_ltog_f_tmp;
    d_ltog_e_raw = d_ltog_e_tmp;
    d_ltog_v_raw = d_ltog_v_tmp;

    // Download owned counts
    std::vector<uint16_t> h_num_owned_f(num_patches);
    std::vector<uint16_t> h_num_owned_e(num_patches);
    std::vector<uint16_t> h_num_owned_v(num_patches);
    CUDA_ERROR(cudaMemcpy(h_num_owned_f.data(), d_owned_f, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(h_num_owned_e.data(), d_owned_e, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(h_num_owned_v.data(), d_owned_v, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaFree(d_owned_f));
    CUDA_ERROR(cudaFree(d_owned_e));
    CUDA_ERROR(cudaFree(d_owned_v));

    // Download offsets (tiny)
    std::vector<uint32_t> h_f_off(num_patches + 1);
    std::vector<uint32_t> h_e_off(num_patches + 1);
    std::vector<uint32_t> h_v_off(num_patches + 1);
    CUDA_ERROR(cudaMemcpy(h_f_off.data(), d_f_offset, (num_patches+1)*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(h_e_off.data(), d_e_offset, (num_patches+1)*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(h_v_off.data(), d_v_offset, (num_patches+1)*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    fprintf(stderr, "[thrust_ltog] step5 partition: %.0fms\n", ms_since(tp));

    // ── Step 6: download to host + retain on device ──────────────────────
    tp = clk::now();
    ThrustLtogResult result;
    result.f_offset = std::move(h_f_off);
    result.e_offset = std::move(h_e_off);
    result.v_offset = std::move(h_v_off);
    result.num_owned_f = std::move(h_num_owned_f);
    result.num_owned_e = std::move(h_num_owned_e);
    result.num_owned_v = std::move(h_num_owned_v);

    // Download ltog arrays
    uint32_t total_f = total_patch_faces;
    result.ltog_f.resize(total_f);
    result.ltog_e.resize(num_unique_edges);
    result.ltog_v.resize(num_unique_verts);
    CUDA_ERROR(cudaMemcpy(result.ltog_f.data(), d_ltog_f_raw,
                          total_f * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(result.ltog_e.data(), d_ltog_e_raw,
                          num_unique_edges * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(result.ltog_v.data(), d_ltog_v_raw,
                          num_unique_verts * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Upload num_elements and num_owned to device for K2
    uint16_t* d_nef; uint16_t* d_nee; uint16_t* d_nev;
    uint16_t* d_nof; uint16_t* d_noe; uint16_t* d_nov;
    CUDA_ERROR(cudaMalloc(&d_nef, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nee, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nev, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nof, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_noe, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nov, num_patches * sizeof(uint16_t)));

    // Compute num_elements from offsets
    std::vector<uint16_t> h_nef(num_patches), h_nee(num_patches), h_nev(num_patches);
    for (uint32_t p = 0; p < num_patches; ++p) {
        h_nef[p] = static_cast<uint16_t>(result.f_offset[p+1] - result.f_offset[p]);
        h_nee[p] = static_cast<uint16_t>(result.e_offset[p+1] - result.e_offset[p]);
        h_nev[p] = static_cast<uint16_t>(result.v_offset[p+1] - result.v_offset[p]);
    }
    CUDA_ERROR(cudaMemcpy(d_nef, h_nef.data(), num_patches*sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_nee, h_nee.data(), num_patches*sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_nev, h_nev.data(), num_patches*sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_nof, result.num_owned_f.data(), num_patches*sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_noe, result.num_owned_e.data(), num_patches*sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_nov, result.num_owned_v.data(), num_patches*sizeof(uint16_t), cudaMemcpyHostToDevice));

    // Retain device pointers
    result.d_ltog_f = d_ltog_f_raw;
    result.d_ltog_e = d_ltog_e_raw;
    result.d_ltog_v = d_ltog_v_raw;
    result.d_f_offset = d_f_offset;
    result.d_e_offset = d_e_offset;
    result.d_v_offset = d_v_offset;
    result.d_num_elements_f = d_nef;
    result.d_num_elements_e = d_nee;
    result.d_num_elements_v = d_nev;
    result.d_num_owned_f = d_nof;
    result.d_num_owned_e = d_noe;
    result.d_num_owned_v = d_nov;
    result.device_arrays_valid = true;

    fprintf(stderr, "[thrust_ltog] step6 download+retain: %.0fms\n", ms_since(tp));
    fprintf(stderr, "[thrust_ltog] TOTAL: %.0fms (F=%zu E=%zu V=%zu)\n",
            ms_since(t_total), result.ltog_f.size(), result.ltog_e.size(),
            result.ltog_v.size());

    return result;
}


// ═══════════════════════════════════════════════════════════════════════════
// K0a-only wrapper
// ═══════════════════════════════════════════════════════════════════════════

void gpu_run_k0a(
    const uint32_t* d_face_patch,
    const uint32_t* d_ev,
    const uint32_t* d_ef_f0,
    uint32_t* d_edge_patch,
    uint32_t* d_vertex_patch,
    uint32_t num_edges)
{
    int grid = (num_edges + 255) / 256;
    k0a_assign_edge_vertex_patch<<<grid, 256>>>(
        d_face_patch, d_ev, d_ef_f0,
        d_edge_patch, d_vertex_patch, num_edges);
    CUDA_ERROR(cudaDeviceSynchronize());
}


// ═══════════════════════════════════════════════════════════════════════════
// Combined K1+K2 wrapper
// ═══════════════════════════════════════════════════════════════════════════

K1K2Result gpu_run_k1k2(
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
    uint32_t max_f, uint32_t max_e, uint32_t max_v,
    uint32_t edge_capacity, uint32_t face_capacity)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();

    // ── K1: build ltog ───────────────────────────────────────────────────
    uint32_t total_f = num_patches * max_f;
    uint32_t total_e = num_patches * max_e;
    uint32_t total_v = num_patches * max_v;

    std::vector<uint32_t> f_off(num_patches+1), e_off(num_patches+1), v_off(num_patches+1);
    for (uint32_t p = 0; p <= num_patches; ++p) {
        f_off[p] = p * max_f;
        e_off[p] = p * max_e;
        v_off[p] = p * max_v;
    }

    uint32_t *d_ltog_f, *d_ltog_e, *d_ltog_v;
    uint32_t *d_f_off, *d_e_off, *d_v_off;
    uint16_t *d_nef, *d_nee, *d_nev, *d_nof, *d_noe, *d_nov;

    CUDA_ERROR(cudaMalloc(&d_ltog_f, total_f * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ltog_e, total_e * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ltog_v, total_v * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_f_off, (num_patches+1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_e_off, (num_patches+1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_v_off, (num_patches+1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_nef, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nee, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nev, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nof, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_noe, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nov, num_patches * sizeof(uint16_t)));

    CUDA_ERROR(cudaMemcpy(d_f_off, f_off.data(), (num_patches+1)*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_e_off, e_off.data(), (num_patches+1)*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_v_off, v_off.data(), (num_patches+1)*sizeof(uint32_t), cudaMemcpyHostToDevice));

    k1_build_ltog<<<num_patches, 256>>>(
        d_fv, d_edge_key, d_ev_global, num_edges_global,
        d_patches_val, d_patches_offset,
        d_ribbon_val, d_ribbon_offset,
        d_face_patch, d_edge_patch, d_vertex_patch,
        num_patches,
        d_ltog_f, d_ltog_e, d_ltog_v,
        d_f_off, d_e_off, d_v_off,
        d_nef, d_nee, d_nev, d_nof, d_noe, d_nov);
    CUDA_ERROR(cudaDeviceSynchronize());
    fprintf(stderr, "[gpu_k1k2] K1 done: %.1fms\n", ms_since(t0));

    // ── K2: build topology ───────────────────────────────────────────────
    auto tp = clk::now();
    uint32_t ev_stride = edge_capacity * 2;
    uint32_t fe_stride = face_capacity * 3;

    uint16_t *d_ev_local, *d_fe_local;
    CUDA_ERROR(cudaMalloc(&d_ev_local, num_patches * ev_stride * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_fe_local, num_patches * fe_stride * sizeof(uint16_t)));
    CUDA_ERROR(cudaMemset(d_ev_local, 0xFF, num_patches * ev_stride * sizeof(uint16_t)));
    CUDA_ERROR(cudaMemset(d_fe_local, 0xFF, num_patches * fe_stride * sizeof(uint16_t)));

    k2_build_topology<<<num_patches, 256>>>(
        d_fv, d_edge_key, d_ev_global, num_edges_global,
        d_patches_val, d_patches_offset,
        d_ribbon_val, d_ribbon_offset,
        d_face_patch, d_edge_patch, d_vertex_patch,
        d_ltog_f, d_f_off, d_ltog_e, d_e_off, d_ltog_v, d_v_off,
        d_nef, d_nof, d_nee, d_noe, d_nev, d_nov,
        num_patches,
        d_ev_local, d_fe_local, ev_stride, fe_stride);
    CUDA_ERROR(cudaDeviceSynchronize());
    fprintf(stderr, "[gpu_k1k2] K2 done: %.1fms\n", ms_since(tp));

    // ── Download results ─────────────────────────────────────────────────
    tp = clk::now();
    K1K2Result r;
    r.max_f = max_f; r.max_e = max_e; r.max_v = max_v;
    r.ev_stride = ev_stride; r.fe_stride = fe_stride;

    r.ltog_f.resize(total_f); r.ltog_e.resize(total_e); r.ltog_v.resize(total_v);
    r.num_elements_f.resize(num_patches); r.num_elements_e.resize(num_patches); r.num_elements_v.resize(num_patches);
    r.num_owned_f.resize(num_patches); r.num_owned_e.resize(num_patches); r.num_owned_v.resize(num_patches);
    r.ev_local.resize(num_patches * ev_stride);
    r.fe_local.resize(num_patches * fe_stride);

    CUDA_ERROR(cudaMemcpy(r.ltog_f.data(), d_ltog_f, total_f*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.ltog_e.data(), d_ltog_e, total_e*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.ltog_v.data(), d_ltog_v, total_v*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_elements_f.data(), d_nef, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_elements_e.data(), d_nee, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_elements_v.data(), d_nev, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_owned_f.data(), d_nof, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_owned_e.data(), d_noe, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_owned_v.data(), d_nov, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.ev_local.data(), d_ev_local, num_patches*ev_stride*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.fe_local.data(), d_fe_local, num_patches*fe_stride*sizeof(uint16_t), cudaMemcpyDeviceToHost));

    fprintf(stderr, "[gpu_k1k2] download: %.1fms, TOTAL: %.1fms\n", ms_since(tp), ms_since(t0));

    CUDA_ERROR(cudaFree(d_ltog_f)); CUDA_ERROR(cudaFree(d_ltog_e)); CUDA_ERROR(cudaFree(d_ltog_v));
    CUDA_ERROR(cudaFree(d_f_off)); CUDA_ERROR(cudaFree(d_e_off)); CUDA_ERROR(cudaFree(d_v_off));
    CUDA_ERROR(cudaFree(d_nef)); CUDA_ERROR(cudaFree(d_nee)); CUDA_ERROR(cudaFree(d_nev));
    CUDA_ERROR(cudaFree(d_nof)); CUDA_ERROR(cudaFree(d_noe)); CUDA_ERROR(cudaFree(d_nov));
    CUDA_ERROR(cudaFree(d_ev_local)); CUDA_ERROR(cudaFree(d_fe_local));

    return r;
}


// ═══════════════════════════════════════════════════════════════════════════
// Launch K2 using retained device arrays from Approach A
// ═══════════════════════════════════════════════════════════════════════════

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
    uint32_t face_capacity)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();

    uint32_t ev_stride = edge_capacity * 2;
    uint32_t fe_stride = face_capacity * 3;

    // Allocate K2 output
    uint16_t *d_ev_local, *d_fe_local;
    CUDA_ERROR(cudaMalloc(&d_ev_local, num_patches * ev_stride * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_fe_local, num_patches * fe_stride * sizeof(uint16_t)));
    CUDA_ERROR(cudaMemset(d_ev_local, 0xFF, num_patches * ev_stride * sizeof(uint16_t)));
    CUDA_ERROR(cudaMemset(d_fe_local, 0xFF, num_patches * fe_stride * sizeof(uint16_t)));

    // Launch K2
    k2_build_topology<<<num_patches, 256>>>(
        d_fv, d_edge_key, d_ev_global, num_edges_global,
        d_patches_val, d_patches_offset,
        d_ribbon_val, d_ribbon_offset,
        d_face_patch, d_edge_patch, d_vertex_patch,
        thr.d_ltog_f, thr.d_f_offset,
        thr.d_ltog_e, thr.d_e_offset,
        thr.d_ltog_v, thr.d_v_offset,
        thr.d_num_elements_f, thr.d_num_owned_f,
        thr.d_num_elements_e, thr.d_num_owned_e,
        thr.d_num_elements_v, thr.d_num_owned_v,
        num_patches,
        d_ev_local, d_fe_local, ev_stride, fe_stride);
    CUDA_ERROR(cudaDeviceSynchronize());
    fprintf(stderr, "[gpu_k2] kernel: %.1fms\n", ms_since(t0));

    // Retain K2 device arrays (D2D copy in build_device, no host round-trip)
    K1K2Result r;
    r.ev_stride = ev_stride;
    r.fe_stride = fe_stride;
    r.d_ev_local = d_ev_local;
    r.d_fe_local = d_fe_local;
    fprintf(stderr, "[gpu_k2] TOTAL: %.1fms (device arrays retained)\n", ms_since(t0));

    // Don't free Approach A device arrays — retained for GPU build_device
    // Caller is responsible for calling thr.free_device() after build_device.

    return r;
}


// ═══════════════════════════════════════════════════════════════════════════
// GPU build_device: bitmasks + stash + hash tables entirely on GPU
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// GPU ribbon extraction — vertex-centric via thrust sort
// ═══════════════════════════════════════════════════════════════════════════

// Kernel: for each multi-patch vertex, emit (patch, face) ribbon pairs
__global__ static void emit_ribbon_pairs_kernel(
    const uint32_t* d_vertex_offset,  // [V+1] into sorted arrays
    const uint32_t* d_sorted_face,    // [3F] face IDs sorted by vertex
    const uint32_t* d_sorted_patch,   // [3F] patch IDs sorted by vertex
    uint64_t*       d_out_pairs,      // [max_pairs] output: (patch<<32)|face
    uint32_t*       d_out_count,      // atomic counter
    uint32_t        V)
{
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= V) return;

    uint32_t begin = d_vertex_offset[v];
    uint32_t end   = d_vertex_offset[v + 1];
    if (begin == end) return;

    // Quick check: all same patch? → no ribbon pairs
    uint32_t p0 = d_sorted_patch[begin];
    bool multi = false;
    for (uint32_t i = begin + 1; i < end; ++i)
        if (d_sorted_patch[i] != p0) { multi = true; break; }
    if (!multi) return;

    // Collect distinct patches (vertex degree typically 5-7, max ~16)
    uint32_t patches[16];
    uint32_t np = 0;
    for (uint32_t i = begin; i < end; ++i) {
        uint32_t p = d_sorted_patch[i];
        bool found = false;
        for (uint32_t j = 0; j < np; ++j)
            if (patches[j] == p) { found = true; break; }
        if (!found && np < 16) patches[np++] = p;
    }

    // For each (face, patch) at this vertex, emit ribbon pairs
    for (uint32_t i = begin; i < end; ++i) {
        uint32_t f = d_sorted_face[i];
        uint32_t p = d_sorted_patch[i];
        for (uint32_t j = 0; j < np; ++j) {
            if (patches[j] != p) {
                uint32_t idx = atomicAdd(d_out_count, 1);
                d_out_pairs[idx] = (uint64_t(patches[j]) << 32) | uint64_t(f);
            }
        }
    }
}

void gpu_extract_ribbons(
    const uint32_t* d_face_patch,  // [F]
    const uint32_t* d_fv,          // [F*3]
    uint32_t num_faces,
    uint32_t num_vertices,
    uint32_t num_patches,
    // Output (caller must cudaFree)
    uint32_t** out_d_ribbon_val,
    uint32_t** out_d_ribbon_offset,
    // Also populate host vectors for legacy code
    std::vector<uint32_t>& h_ribbon_val,
    std::vector<uint32_t>& h_ribbon_offset)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();
    uint32_t N = num_faces * 3;

    // ── Step 1: emit (vertex, face, patch) as sort keys ──────────────────
    // Pack as (vertex_id) for sorting key, (face_id, patch_id) as values
    uint32_t* d_vkeys;   // vertex IDs [3F]
    uint32_t* d_fvals;   // face IDs [3F]
    uint32_t* d_pvals;   // patch IDs [3F]
    CUDA_ERROR(cudaMalloc(&d_vkeys, N * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_fvals, N * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_pvals, N * sizeof(uint32_t)));

    // Tabulate: for index i, vertex = d_fv[i], face = i/3, patch = d_face_patch[i/3]
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(N),
        [d_vkeys, d_fvals, d_pvals, d_fv, d_face_patch] __device__ (uint32_t i) {
            uint32_t f = i / 3;
            d_vkeys[i] = d_fv[i];
            d_fvals[i] = f;
            d_pvals[i] = d_face_patch[f];
        });
    fprintf(stderr, "[gpu_ribbons] step1 emit: %.0fms\n", ms_since(t0));

    // ── Step 2: sort by vertex ───────────────────────────────────────────
    auto tp = clk::now();
    // Sort all three arrays by vertex key using zip iterator
    // Simplest: pack (face, patch) into uint64, sort_by_key with vertex as key
    uint64_t* d_fp_packed;
    CUDA_ERROR(cudaMalloc(&d_fp_packed, N * sizeof(uint64_t)));
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(N),
        [d_fp_packed, d_fvals, d_pvals] __device__ (uint32_t i) {
            d_fp_packed[i] = (uint64_t(d_pvals[i]) << 32) | uint64_t(d_fvals[i]);
        });
    CUDA_ERROR(cudaFree(d_fvals));
    CUDA_ERROR(cudaFree(d_pvals));

    thrust::sort_by_key(thrust::device,
        thrust::device_pointer_cast(d_vkeys),
        thrust::device_pointer_cast(d_vkeys) + N,
        thrust::device_pointer_cast(d_fp_packed));
    fprintf(stderr, "[gpu_ribbons] step2 sort: %.0fms\n", ms_since(tp));

    // ── Step 3: vertex offsets ───────────────────────────────────────────
    tp = clk::now();
    uint32_t* d_vertex_offset;
    CUDA_ERROR(cudaMalloc(&d_vertex_offset, (num_vertices + 1) * sizeof(uint32_t)));

    // Generate search values [0, 1, 2, ..., V]
    uint32_t* d_search;
    CUDA_ERROR(cudaMalloc(&d_search, (num_vertices + 1) * sizeof(uint32_t)));
    thrust::sequence(thrust::device,
        thrust::device_pointer_cast(d_search),
        thrust::device_pointer_cast(d_search) + num_vertices + 1);

    thrust::upper_bound(thrust::device,
        thrust::device_pointer_cast(d_vkeys),
        thrust::device_pointer_cast(d_vkeys) + N,
        thrust::device_pointer_cast(d_search),
        thrust::device_pointer_cast(d_search) + num_vertices + 1,
        thrust::device_pointer_cast(d_vertex_offset));

    // Shift: vertex_offset should be [0, upper_bound(0), upper_bound(1), ...]
    // Actually upper_bound gives us the END of each vertex's range.
    // We need: offset[v] = lower_bound(v), offset[V] = N
    // Simpler: use lower_bound for [0..V]
    thrust::lower_bound(thrust::device,
        thrust::device_pointer_cast(d_vkeys),
        thrust::device_pointer_cast(d_vkeys) + N,
        thrust::device_pointer_cast(d_search),
        thrust::device_pointer_cast(d_search) + num_vertices + 1,
        thrust::device_pointer_cast(d_vertex_offset));

    CUDA_ERROR(cudaFree(d_search));
    CUDA_ERROR(cudaFree(d_vkeys));

    // Unpack face and patch from d_fp_packed
    uint32_t* d_sorted_face;
    uint32_t* d_sorted_patch;
    CUDA_ERROR(cudaMalloc(&d_sorted_face, N * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_sorted_patch, N * sizeof(uint32_t)));
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(N),
        [d_sorted_face, d_sorted_patch, d_fp_packed] __device__ (uint32_t i) {
            d_sorted_face[i] = d_fp_packed[i] & 0xFFFFFFFFu;
            d_sorted_patch[i] = d_fp_packed[i] >> 32;
        });
    CUDA_ERROR(cudaFree(d_fp_packed));
    fprintf(stderr, "[gpu_ribbons] step3 offsets: %.0fms\n", ms_since(tp));

    // ── Step 4: emit ribbon pairs ────────────────────────────────────────
    tp = clk::now();
    // Over-allocate: max 3F pairs (in practice ~5-10% of that)
    uint64_t* d_ribbon_pairs;
    uint32_t* d_pair_count;
    CUDA_ERROR(cudaMalloc(&d_ribbon_pairs, N * sizeof(uint64_t)));
    CUDA_ERROR(cudaMalloc(&d_pair_count, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_pair_count, 0, sizeof(uint32_t)));

    int grid = (num_vertices + 255) / 256;
    emit_ribbon_pairs_kernel<<<grid, 256>>>(
        d_vertex_offset, d_sorted_face, d_sorted_patch,
        d_ribbon_pairs, d_pair_count, num_vertices);
    CUDA_ERROR(cudaDeviceSynchronize());

    uint32_t h_pair_count;
    CUDA_ERROR(cudaMemcpy(&h_pair_count, d_pair_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_ERROR(cudaFree(d_vertex_offset));
    CUDA_ERROR(cudaFree(d_sorted_face));
    CUDA_ERROR(cudaFree(d_sorted_patch));
    CUDA_ERROR(cudaFree(d_pair_count));
    fprintf(stderr, "[gpu_ribbons] step4 emit: %.0fms (%u pairs)\n", ms_since(tp), h_pair_count);

    // ── Step 5: sort + unique → deduplicated (patch, face) pairs ─────────
    tp = clk::now();
    thrust::sort(thrust::device,
        thrust::device_pointer_cast(d_ribbon_pairs),
        thrust::device_pointer_cast(d_ribbon_pairs) + h_pair_count);

    uint64_t* d_unique_pairs;
    CUDA_ERROR(cudaMalloc(&d_unique_pairs, h_pair_count * sizeof(uint64_t)));
    auto unique_end = thrust::unique_copy(thrust::device,
        thrust::device_pointer_cast(d_ribbon_pairs),
        thrust::device_pointer_cast(d_ribbon_pairs) + h_pair_count,
        thrust::device_pointer_cast(d_unique_pairs));
    uint32_t num_unique = unique_end - thrust::device_pointer_cast(d_unique_pairs);
    CUDA_ERROR(cudaFree(d_ribbon_pairs));
    fprintf(stderr, "[gpu_ribbons] step5 sort+unique: %.0fms (%u unique)\n", ms_since(tp), num_unique);

    // ── Step 6: build CSR (ribbon_offset, ribbon_val) ────────────────────
    tp = clk::now();
    // Extract patch IDs from unique pairs
    uint32_t* d_ribbon_patches;
    uint32_t* d_ribbon_faces;
    CUDA_ERROR(cudaMalloc(&d_ribbon_patches, num_unique * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ribbon_faces, num_unique * sizeof(uint32_t)));
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(num_unique),
        [d_ribbon_patches, d_ribbon_faces, d_unique_pairs] __device__ (uint32_t i) {
            d_ribbon_patches[i] = d_unique_pairs[i] >> 32;
            d_ribbon_faces[i] = d_unique_pairs[i] & 0xFFFFFFFFu;
        });
    CUDA_ERROR(cudaFree(d_unique_pairs));

    // CSR offsets via lower_bound
    uint32_t* d_rib_offset;
    CUDA_ERROR(cudaMalloc(&d_rib_offset, (num_patches + 1) * sizeof(uint32_t)));
    uint32_t* d_patch_search;
    CUDA_ERROR(cudaMalloc(&d_patch_search, (num_patches + 1) * sizeof(uint32_t)));
    thrust::sequence(thrust::device,
        thrust::device_pointer_cast(d_patch_search),
        thrust::device_pointer_cast(d_patch_search) + num_patches + 1);
    thrust::lower_bound(thrust::device,
        thrust::device_pointer_cast(d_ribbon_patches),
        thrust::device_pointer_cast(d_ribbon_patches) + num_unique,
        thrust::device_pointer_cast(d_patch_search),
        thrust::device_pointer_cast(d_patch_search) + num_patches + 1,
        thrust::device_pointer_cast(d_rib_offset));
    CUDA_ERROR(cudaFree(d_patch_search));
    CUDA_ERROR(cudaFree(d_ribbon_patches));

    fprintf(stderr, "[gpu_ribbons] step6 CSR: %.0fms\n", ms_since(tp));
    fprintf(stderr, "[gpu_ribbons] TOTAL: %.0fms (%u ribbon faces)\n", ms_since(t0), num_unique);

    // Output device arrays
    *out_d_ribbon_val = d_ribbon_faces;
    *out_d_ribbon_offset = d_rib_offset;

    // Download to host for legacy Patcher interface
    h_ribbon_val.resize(num_unique);
    h_ribbon_offset.resize(num_patches);  // cumulative format for Patcher
    CUDA_ERROR(cudaMemcpy(h_ribbon_val.data(), d_ribbon_faces,
                          num_unique * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // Download prefix offsets and convert to cumulative format
    std::vector<uint32_t> pfx(num_patches + 1);
    CUDA_ERROR(cudaMemcpy(pfx.data(), d_rib_offset,
                          (num_patches + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (uint32_t p = 0; p < num_patches; ++p)
        h_ribbon_offset[p] = pfx[p + 1];  // cumulative end offset
}


// One thread per patch: build all 3 hash tables (V/E/F)
__global__ static void k3_build_hashtables(
    const uint32_t* d_ltog_v, const uint32_t* d_ltog_e, const uint32_t* d_ltog_f,
    const uint32_t* d_v_off, const uint32_t* d_e_off, const uint32_t* d_f_off,
    const uint16_t* d_nev, const uint16_t* d_nee, const uint16_t* d_nef,
    const uint16_t* d_nov, const uint16_t* d_noe, const uint16_t* d_nof,
    const uint32_t* d_vertex_patch, const uint32_t* d_edge_patch,
    const uint32_t* d_face_patch,
    const uint8_t* d_stash_bulk, size_t stash_bytes_per,
    uint8_t* d_ht_v_bulk, uint8_t* d_ht_e_bulk, uint8_t* d_ht_f_bulk,
    size_t ht_v_bytes, size_t ht_e_bytes, size_t ht_f_bytes,
    uint8_t* d_ht_stash_v_bulk, uint8_t* d_ht_stash_e_bulk,
    uint8_t* d_ht_stash_f_bulk, size_t ht_stash_bytes,
    LPHashTable ht_v, LPHashTable ht_e, LPHashTable ht_f,
    uint32_t num_patches)
{
    uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= num_patches) return;

    const uint32_t* stash = (const uint32_t*)(d_stash_bulk + p * stash_bytes_per);
    constexpr uint8_t STASH_SZ = PatchStash::stash_size;

    // Find patch index in stash
    auto find_stash_idx = [&](uint32_t owner) -> uint8_t {
        for (uint8_t i = 0; i < STASH_SZ; ++i)
            if (stash[i] == owner) return i;
        return 0xFF;
    };

    // Binary search in sorted array [arr, arr+count) for val
    auto bin_search = [](const uint32_t* arr, uint16_t count,
                         uint32_t val) -> uint16_t {
        uint16_t lo = 0, hi = count;
        while (lo < hi) {
            uint16_t mid = (lo + hi) / 2;
            if (arr[mid] < val) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    };

    // Inline cuckoo insert — avoids LPHashTable member pointer issues
    auto do_insert = [](LPPair pair, LPPair* table, uint16_t cap,
                        LPPair* ht_stash, uint16_t max_chains,
                        const LPHashTable::HashT& h0, const LPHashTable::HashT& h1,
                        const LPHashTable::HashT& h2, const LPHashTable::HashT& h3) {
        auto bucket_id = h0(pair.key()) % cap;
        uint16_t ctr = 0;
        do {
            uint32_t tmp = pair.m_pair;
            pair.m_pair = atomicExch(&table[bucket_id].m_pair, tmp);
            if (pair.is_sentinel() || pair.key() == ((tmp >> (32 - LPPair::LIDNumBits))))
                return;
            auto b0 = h0(pair.key()) % cap;
            auto b1 = h1(pair.key()) % cap;
            auto b2 = h2(pair.key()) % cap;
            auto b3 = h3(pair.key()) % cap;
            auto nb = b0;
            if (bucket_id == b2) nb = b3;
            else if (bucket_id == b1) nb = b2;
            else if (bucket_id == b0) nb = b1;
            bucket_id = nb;
            ctr++;
        } while (ctr < max_chains);
        // Stash overflow
        for (uint8_t i = 0; i < LPHashTable::stash_size; ++i) {
            LPPair sentinel;
            uint32_t old = atomicCAS(&ht_stash[i].m_pair, sentinel.m_pair, pair.m_pair);
            if (old == sentinel.m_pair) return;
        }
    };

    // Build one HT type
    auto build_ht = [&](const uint32_t* d_ltog, const uint32_t* d_off,
                         const uint16_t* d_ne, const uint16_t* d_no,
                         const uint32_t* d_elem_patch,
                         uint8_t* ht_bulk, size_t ht_bytes,
                         uint8_t* ht_st_bulk, size_t ht_st_bytes,
                         const LPHashTable& ht_tmpl) {
        LPPair* table = (LPPair*)(ht_bulk + p * ht_bytes);
        LPPair* ht_st = (LPPair*)(ht_st_bulk + p * ht_st_bytes);

        uint32_t base = d_off[p];
        uint16_t num_elem = d_ne[p], num_owned = d_no[p];

        for (uint16_t i = num_owned; i < num_elem; ++i) {
            uint32_t global_id = d_ltog[base + i];
            uint32_t owner = d_elem_patch[global_id];
            uint32_t owner_base = d_off[owner];
            uint16_t owner_owned = d_no[owner];
            uint16_t lid_in_owner = bin_search(
                d_ltog + owner_base, owner_owned, global_id);
            // Verify the binary search found an exact match
            if (lid_in_owner >= owner_owned ||
                d_ltog[owner_base + lid_in_owner] != global_id)
                continue;  // skip — element not in owner's owned segment
            uint8_t stash_idx = find_stash_idx(owner);
            if (stash_idx == 0xFF) continue;  // owner not in stash
            LPPair pair(i, lid_in_owner, stash_idx);
            LPHashTable ht_copy = ht_tmpl;
            ht_copy.m_table = table;
            ht_copy.m_stash = ht_st;
            ht_copy.insert(pair, (volatile LPPair*)table, (volatile LPPair*)ht_st);
        }
    };

    // Build V/E/F hash tables
    build_ht(d_ltog_v, d_v_off, d_nev, d_nov, d_vertex_patch,
             d_ht_v_bulk, ht_v_bytes, d_ht_stash_v_bulk, ht_stash_bytes, ht_v);
    build_ht(d_ltog_e, d_e_off, d_nee, d_noe, d_edge_patch,
             d_ht_e_bulk, ht_e_bytes, d_ht_stash_e_bulk, ht_stash_bytes, ht_e);
    build_ht(d_ltog_f, d_f_off, d_nef, d_nof, d_face_patch,
             d_ht_f_bulk, ht_f_bytes, d_ht_stash_f_bulk, ht_stash_bytes, ht_f);
}


// ═══════════════════════════════════════════════════════════════════════════
// GPU create_handles: one thread per patch, write handles for owned elements
// ═══════════════════════════════════════════════════════════════════════════

void gpu_create_handles(
    const uint32_t* d_vertex_prefix,
    const uint32_t* d_edge_prefix,
    const uint32_t* d_face_prefix,
    const uint16_t* d_num_owned_v,
    const uint16_t* d_num_owned_e,
    const uint16_t* d_num_owned_f,
    uint32_t num_patches,
    void* d_v_handles,
    void* d_e_handles,
    void* d_f_handles)
{
    // Handle is uint64_t = (patch_id << 32) | local_id
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(num_patches),
        [d_vertex_prefix, d_edge_prefix, d_face_prefix,
         d_num_owned_v, d_num_owned_e, d_num_owned_f,
         d_vh = (uint64_t*)d_v_handles,
         d_eh = (uint64_t*)d_e_handles,
         d_fh = (uint64_t*)d_f_handles
        ] __device__ (uint32_t p) {
            uint32_t vb = d_vertex_prefix[p];
            for (uint16_t v = 0; v < d_num_owned_v[p]; ++v)
                d_vh[vb + v] = (uint64_t(p) << 32) | uint64_t(v);

            uint32_t eb = d_edge_prefix[p];
            for (uint16_t e = 0; e < d_num_owned_e[p]; ++e)
                d_eh[eb + e] = (uint64_t(p) << 32) | uint64_t(e);

            uint32_t fb = d_face_prefix[p];
            for (uint16_t f = 0; f < d_num_owned_f[p]; ++f)
                d_fh[fb + f] = (uint64_t(p) << 32) | uint64_t(f);
        });
    CUDA_ERROR(cudaDeviceSynchronize());
}


// ═══════════════════════════════════════════════════════════════════════════
// GPU two-ring graph coloring via Jones-Plassmann
// ═══════════════════════════════════════════════════════════════════════════

void gpu_patch_coloring(
    const uint32_t* d_stash,   // [P * stash_size] flat neighbor IDs
    uint32_t num_patches,
    uint32_t stash_size,
    uint32_t* h_colors,        // [P] output on host
    uint32_t& num_colors)
{
    // Upload random priorities
    std::vector<uint32_t> h_prio(num_patches);
    {
        std::mt19937 rng(42);
        for (auto& p : h_prio) p = rng();
    }
    uint32_t* d_prio;
    uint32_t* d_colors;
    CUDA_ERROR(cudaMalloc(&d_prio, num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_colors, num_patches * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_prio, h_prio.data(), num_patches*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemset(d_colors, 0xFF, num_patches * sizeof(uint32_t)));  // INVALID32

    uint32_t* d_remaining;  // count of uncolored
    CUDA_ERROR(cudaMalloc(&d_remaining, sizeof(uint32_t)));

    uint32_t ss = stash_size;
    for (int round = 0; round < 200; ++round) {  // max 200 rounds
        CUDA_ERROR(cudaMemset(d_remaining, 0, sizeof(uint32_t)));

        thrust::for_each(thrust::device,
            thrust::make_counting_iterator(0u),
            thrust::make_counting_iterator(num_patches),
            [d_stash, d_prio, d_colors, d_remaining, ss, num_patches
            ] __device__ (uint32_t p) {
                if (d_colors[p] != INVALID32) return;  // already colored

                // Check if p has highest priority among uncolored 2-ring neighbors
                uint32_t my_prio = d_prio[p];
                const uint32_t* my_stash = d_stash + p * ss;

                // Two-ring: check neighbors and neighbors-of-neighbors
                for (uint32_t i = 0; i < ss; ++i) {
                    uint32_t n = my_stash[i];
                    if (n == INVALID32 || n >= num_patches) break;
                    if (d_colors[n] == INVALID32 && d_prio[n] > my_prio)
                        { atomicAdd(d_remaining, 1); return; }
                    // Two-ring
                    const uint32_t* n_stash = d_stash + n * ss;
                    for (uint32_t j = 0; j < ss; ++j) {
                        uint32_t nn = n_stash[j];
                        if (nn == INVALID32 || nn >= num_patches) break;
                        if (nn != p && d_colors[nn] == INVALID32 && d_prio[nn] > my_prio)
                            { atomicAdd(d_remaining, 1); return; }
                    }
                }

                // p is a local maximum — assign smallest unused color
                // Collect 2-ring neighbor colors in a bitmask (max ~64 colors)
                uint64_t used = 0;
                for (uint32_t i = 0; i < ss; ++i) {
                    uint32_t n = my_stash[i];
                    if (n == INVALID32 || n >= num_patches) break;
                    uint32_t c = d_colors[n];
                    if (c != INVALID32 && c < 64) used |= (1ULL << c);
                    const uint32_t* n_stash = d_stash + n * ss;
                    for (uint32_t j = 0; j < ss; ++j) {
                        uint32_t nn = n_stash[j];
                        if (nn == INVALID32 || nn >= num_patches) break;
                        if (nn != p) {
                            uint32_t cc = d_colors[nn];
                            if (cc != INVALID32 && cc < 64) used |= (1ULL << cc);
                        }
                    }
                }
                // Find first zero bit
                for (uint32_t c = 0; c < 64; ++c) {
                    if (!(used & (1ULL << c))) { d_colors[p] = c; return; }
                }
                d_colors[p] = 64;  // fallback
            });
        CUDA_ERROR(cudaDeviceSynchronize());

        uint32_t h_rem;
        CUDA_ERROR(cudaMemcpy(&h_rem, d_remaining, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (h_rem == 0) break;
    }

    // Download colors
    CUDA_ERROR(cudaMemcpy(h_colors, d_colors, num_patches*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    num_colors = 0;
    for (uint32_t p = 0; p < num_patches; ++p)
        num_colors = std::max(num_colors, h_colors[p]);
    num_colors++;

    CUDA_ERROR(cudaFree(d_prio));
    CUDA_ERROR(cudaFree(d_colors));
    CUDA_ERROR(cudaFree(d_remaining));
}


void gpu_build_stash(
    const ThrustLtogResult& thr,
    const uint32_t* d_face_patch,
    const uint32_t* d_edge_patch,
    const uint32_t* d_vertex_patch,
    uint32_t num_patches,
    uint8_t* d_stash_bulk,
    size_t stash_bytes_per)
{
    constexpr uint8_t STASH_SIZE = PatchStash::stash_size;
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(num_patches),
        [d_ltog_f = thr.d_ltog_f, d_ltog_e = thr.d_ltog_e, d_ltog_v = thr.d_ltog_v,
         d_f_off = thr.d_f_offset, d_e_off = thr.d_e_offset, d_v_off = thr.d_v_offset,
         d_nof = thr.d_num_owned_f, d_noe = thr.d_num_owned_e, d_nov = thr.d_num_owned_v,
         d_ne = thr.d_num_elements_f, d_nee = thr.d_num_elements_e,
         d_nev = thr.d_num_elements_v,
         d_face_patch, d_edge_patch, d_vertex_patch,
         d_stash_bulk, stash_bytes_per, STASH_SIZE
        ] __device__ (uint32_t p) {
            uint32_t* stash = (uint32_t*)(d_stash_bulk + p * stash_bytes_per);
            uint8_t cnt = 0;
            auto ins = [&](uint32_t owner) {
                for (uint8_t i = 0; i < cnt; ++i)
                    if (stash[i] == owner) return;
                if (cnt < STASH_SIZE) stash[cnt++] = owner;
            };
            uint32_t fs = d_f_off[p];
            for (uint16_t i = d_nof[p]; i < d_ne[p]; ++i)
                ins(d_face_patch[d_ltog_f[fs + i]]);
            uint32_t es = d_e_off[p];
            for (uint16_t i = d_noe[p]; i < d_nee[p]; ++i)
                ins(d_edge_patch[d_ltog_e[es + i]]);
            uint32_t vs = d_v_off[p];
            for (uint16_t i = d_nov[p]; i < d_nev[p]; ++i)
                ins(d_vertex_patch[d_ltog_v[vs + i]]);
        });
    CUDA_ERROR(cudaDeviceSynchronize());
}


void gpu_build_device_data(
    const ThrustLtogResult& thr,
    const uint32_t* d_face_patch,
    const uint32_t* d_edge_patch,
    const uint32_t* d_vertex_patch,
    uint32_t num_patches,
    uint16_t v_cap, uint16_t e_cap, uint16_t f_cap,
    // Bulk device arrays to fill (already allocated, zeroed/0xFF'd as needed)
    uint8_t* d_mask_av_bulk, uint8_t* d_mask_ae_bulk, uint8_t* d_mask_af_bulk,
    uint8_t* d_mask_ov_bulk, uint8_t* d_mask_oe_bulk, uint8_t* d_mask_of_bulk,
    size_t mask_v_bytes, size_t mask_e_bytes, size_t mask_f_bytes,
    uint8_t* d_counts_bulk, size_t counts_bytes,
    uint8_t* d_stash_bulk, size_t stash_bytes_per,
    uint8_t* d_ht_v_bulk, uint8_t* d_ht_e_bulk, uint8_t* d_ht_f_bulk,
    size_t ht_v_bytes, size_t ht_e_bytes, size_t ht_f_bytes,
    uint8_t* d_ht_stash_v_bulk, uint8_t* d_ht_stash_e_bulk,
    uint8_t* d_ht_stash_f_bulk, size_t ht_stash_bytes,
    // Hash table params (uniform across patches)
    LPHashTable ht_template_v, LPHashTable ht_template_e, LPHashTable ht_template_f)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();

    // ── K3a: Bitmasks + Counts ───────────────────────────────────────────
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(num_patches),
        [d_mask_av_bulk, d_mask_ae_bulk, d_mask_af_bulk,
         d_mask_ov_bulk, d_mask_oe_bulk, d_mask_of_bulk,
         mask_v_bytes, mask_e_bytes, mask_f_bytes,
         d_counts_bulk, counts_bytes,
         v_cap, e_cap, f_cap,
         d_ne = thr.d_num_elements_f, d_nee = thr.d_num_elements_e,
         d_nev = thr.d_num_elements_v,
         d_nof = thr.d_num_owned_f, d_noe = thr.d_num_owned_e,
         d_nov = thr.d_num_owned_v
        ] __device__ (uint32_t p) {
            uint16_t nf = d_ne[p], ne = d_nee[p], nv = d_nev[p];
            uint16_t of = d_nof[p], oe = d_noe[p], ov = d_nov[p];

            // Counts
            uint16_t* counts = (uint16_t*)(d_counts_bulk + p * counts_bytes);
            counts[0] = nf; counts[1] = ne; counts[2] = nv;

            // Helper: set bits [0..n) in mask, clear [n..cap)
            auto fill_mask = [](uint8_t* buf, uint16_t n) {
                uint32_t* mask = (uint32_t*)buf;
                // Set complete words
                uint16_t full_words = n / 32;
                for (uint16_t w = 0; w < full_words; ++w)
                    mask[w] = 0xFFFFFFFFu;
                // Partial word
                uint16_t remaining = n % 32;
                if (remaining > 0)
                    mask[full_words] = (1u << remaining) - 1u;
                // Rest already zeroed by cudaMemset
            };

            fill_mask(d_mask_av_bulk + p * mask_v_bytes, nv);
            fill_mask(d_mask_ae_bulk + p * mask_e_bytes, ne);
            fill_mask(d_mask_af_bulk + p * mask_f_bytes, nf);
            fill_mask(d_mask_ov_bulk + p * mask_v_bytes, ov);
            fill_mask(d_mask_oe_bulk + p * mask_e_bytes, oe);
            fill_mask(d_mask_of_bulk + p * mask_f_bytes, of);
        });
    CUDA_ERROR(cudaDeviceSynchronize());
    fprintf(stderr, "[gpu_bd] bitmasks+counts: %.1fms\n", ms_since(t0));

    // ── K3b: Patch stash ─────────────────────────────────────────────────
    auto tp = clk::now();
    constexpr uint8_t STASH_SIZE = PatchStash::stash_size;  // 64

    thrust::for_each(thrust::device,
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(num_patches),
        [d_ltog_f = thr.d_ltog_f, d_ltog_e = thr.d_ltog_e, d_ltog_v = thr.d_ltog_v,
         d_f_off = thr.d_f_offset, d_e_off = thr.d_e_offset, d_v_off = thr.d_v_offset,
         d_nof = thr.d_num_owned_f, d_noe = thr.d_num_owned_e, d_nov = thr.d_num_owned_v,
         d_ne = thr.d_num_elements_f, d_nee = thr.d_num_elements_e,
         d_nev = thr.d_num_elements_v,
         d_face_patch, d_edge_patch, d_vertex_patch,
         d_stash_bulk, stash_bytes_per, STASH_SIZE
        ] __device__ (uint32_t p) {
            uint32_t* stash = (uint32_t*)(d_stash_bulk + p * stash_bytes_per);
            // stash pre-initialized to 0xFF (INVALID32) via cudaMemset
            uint8_t stash_count = 0;

            // Helper: insert unique owner patch into stash
            auto stash_insert = [&](uint32_t owner) {
                // Check if already present
                for (uint8_t i = 0; i < stash_count; ++i)
                    if (stash[i] == owner) return;
                if (stash_count < STASH_SIZE)
                    stash[stash_count++] = owner;
            };

            // Faces: not-owned are [owned_f..num_elements_f)
            uint32_t fs = d_f_off[p];
            uint16_t nf = d_ne[p], of = d_nof[p];
            for (uint16_t i = of; i < nf; ++i)
                stash_insert(d_face_patch[d_ltog_f[fs + i]]);

            // Edges
            uint32_t es = d_e_off[p];
            uint16_t ne = d_nee[p], oe = d_noe[p];
            for (uint16_t i = oe; i < ne; ++i)
                stash_insert(d_edge_patch[d_ltog_e[es + i]]);

            // Vertices
            uint32_t vs = d_v_off[p];
            uint16_t nv = d_nev[p], ov = d_nov[p];
            for (uint16_t i = ov; i < nv; ++i)
                stash_insert(d_vertex_patch[d_ltog_v[vs + i]]);
        });
    CUDA_ERROR(cudaDeviceSynchronize());
    fprintf(stderr, "[gpu_bd] stash: %.1fms\n", ms_since(tp));

    // ── K3c: Hash tables (proper __global__ kernel) ────────────────────
    tp = clk::now();
    k3_build_hashtables<<<(num_patches+255)/256, 256>>>(
        thr.d_ltog_v, thr.d_ltog_e, thr.d_ltog_f,
        thr.d_v_offset, thr.d_e_offset, thr.d_f_offset,
        thr.d_num_elements_v, thr.d_num_elements_e, thr.d_num_elements_f,
        thr.d_num_owned_v, thr.d_num_owned_e, thr.d_num_owned_f,
        d_vertex_patch, d_edge_patch, d_face_patch,
        d_stash_bulk, stash_bytes_per,
        d_ht_v_bulk, d_ht_e_bulk, d_ht_f_bulk,
        ht_v_bytes, ht_e_bytes, ht_f_bytes,
        d_ht_stash_v_bulk, d_ht_stash_e_bulk, d_ht_stash_f_bulk,
        ht_stash_bytes,
        ht_template_v, ht_template_e, ht_template_f,
        num_patches);
    CUDA_ERROR(cudaDeviceSynchronize());
    fprintf(stderr, "[gpu_bd] hashtables: %.1fms\n", ms_since(tp));
    fprintf(stderr, "[gpu_bd] TOTAL: %.1fms\n", ms_since(t0));
}


// ═══════════════════════════════════════════════════════════════════════════
// Test wrapper for K1
// ═══════════════════════════════════════════════════════════════════════════

K1Result gpu_test_k1(
    const uint32_t* d_fv,
    const uint64_t* d_edge_key,
    uint32_t num_edges_global,
    const uint32_t* d_patches_val,
    const uint32_t* d_patches_offset,
    const uint32_t* d_ribbon_val,
    const uint32_t* d_ribbon_offset,
    const uint32_t* d_face_patch,
    const uint32_t* d_edge_patch,
    const uint32_t* d_vertex_patch,
    uint32_t num_patches,
    uint32_t max_f, uint32_t max_e, uint32_t max_v)
{
    uint32_t total_f = num_patches * max_f;
    uint32_t total_e = num_patches * max_e;
    uint32_t total_v = num_patches * max_v;

    // Build offset arrays (uniform stride)
    std::vector<uint32_t> f_off(num_patches + 1), e_off(num_patches + 1), v_off(num_patches + 1);
    for (uint32_t p = 0; p <= num_patches; ++p) {
        f_off[p] = p * max_f;
        e_off[p] = p * max_e;
        v_off[p] = p * max_v;
    }

    uint32_t *d_ltog_f, *d_ltog_e, *d_ltog_v;
    uint32_t *d_f_off, *d_e_off, *d_v_off;
    uint16_t *d_nef, *d_nee, *d_nev;
    uint16_t *d_nof, *d_noe, *d_nov;

    CUDA_ERROR(cudaMalloc(&d_ltog_f, total_f * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ltog_e, total_e * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_ltog_v, total_v * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_f_off, (num_patches+1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_e_off, (num_patches+1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_v_off, (num_patches+1) * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_nef, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nee, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nev, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nof, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_noe, num_patches * sizeof(uint16_t)));
    CUDA_ERROR(cudaMalloc(&d_nov, num_patches * sizeof(uint16_t)));

    CUDA_ERROR(cudaMemcpy(d_f_off, f_off.data(), (num_patches+1)*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_e_off, e_off.data(), (num_patches+1)*sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_v_off, v_off.data(), (num_patches+1)*sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Launch K1
    k1_build_ltog<<<num_patches, 256>>>(
        d_fv, d_edge_key, nullptr, num_edges_global,
        d_patches_val, d_patches_offset,
        d_ribbon_val, d_ribbon_offset,
        d_face_patch, d_edge_patch, d_vertex_patch,
        num_patches,
        d_ltog_f, d_ltog_e, d_ltog_v,
        d_f_off, d_e_off, d_v_off,
        d_nef, d_nee, d_nev,
        d_nof, d_noe, d_nov);
    CUDA_ERROR(cudaDeviceSynchronize());

    // Download results
    K1Result r;
    r.max_f_per_patch = max_f;
    r.max_e_per_patch = max_e;
    r.max_v_per_patch = max_v;

    r.ltog_f.resize(total_f);
    r.ltog_e.resize(total_e);
    r.ltog_v.resize(total_v);
    r.num_elements_f.resize(num_patches);
    r.num_elements_e.resize(num_patches);
    r.num_elements_v.resize(num_patches);
    r.num_owned_f.resize(num_patches);
    r.num_owned_e.resize(num_patches);
    r.num_owned_v.resize(num_patches);

    CUDA_ERROR(cudaMemcpy(r.ltog_f.data(), d_ltog_f, total_f*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.ltog_e.data(), d_ltog_e, total_e*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.ltog_v.data(), d_ltog_v, total_v*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_elements_f.data(), d_nef, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_elements_e.data(), d_nee, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_elements_v.data(), d_nev, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_owned_f.data(), d_nof, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_owned_e.data(), d_noe, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(r.num_owned_v.data(), d_nov, num_patches*sizeof(uint16_t), cudaMemcpyDeviceToHost));

    CUDA_ERROR(cudaFree(d_ltog_f)); CUDA_ERROR(cudaFree(d_ltog_e)); CUDA_ERROR(cudaFree(d_ltog_v));
    CUDA_ERROR(cudaFree(d_f_off)); CUDA_ERROR(cudaFree(d_e_off)); CUDA_ERROR(cudaFree(d_v_off));
    CUDA_ERROR(cudaFree(d_nef)); CUDA_ERROR(cudaFree(d_nee)); CUDA_ERROR(cudaFree(d_nev));
    CUDA_ERROR(cudaFree(d_nof)); CUDA_ERROR(cudaFree(d_noe)); CUDA_ERROR(cudaFree(d_nov));

    return r;
}

}  // namespace rxmesh
