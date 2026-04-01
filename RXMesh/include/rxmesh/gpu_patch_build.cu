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
#include <thrust/execution_policy.h>

#include <chrono>
#include <cstdio>

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
#define K1_MAX_EDGES  2048
#define K1_MAX_VERTS  1024

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

    // ── Sort and deduplicate in shared memory ────────────────────────────
    // Thread 0 does the sort+unique (small arrays, <2048 elements)
    // TODO: parallel bitonic sort would be faster but sequential is simpler
    if (threadIdx.x == 0) {
        uint32_t nv_raw = total_faces * 3;
        uint32_t ne_raw = total_faces * 3;

        // Simple insertion sort + dedup for vertices
        // Copy to s_verts, sort, unique
        s_nv = 0;
        for (uint32_t i = 0; i < nv_raw && s_nv < K1_MAX_VERTS; ++i) {
            uint32_t val = s_all_verts[i];
            // Binary insert into sorted s_verts
            uint16_t pos = 0;
            while (pos < s_nv && s_verts[pos] < val) pos++;
            if (pos < s_nv && s_verts[pos] == val) continue; // duplicate
            // Shift right
            for (uint16_t j = s_nv; j > pos; --j) s_verts[j] = s_verts[j-1];
            s_verts[pos] = val;
            s_nv++;
        }

        // Same for edges
        s_ne = 0;
        for (uint32_t i = 0; i < ne_raw && s_ne < K1_MAX_EDGES; ++i) {
            uint32_t val = s_all_edges[i];
            uint16_t pos = 0;
            while (pos < s_ne && s_edges[pos] < val) pos++;
            if (pos < s_ne && s_edges[pos] == val) continue;
            for (uint16_t j = s_ne; j > pos; --j) s_edges[j] = s_edges[j-1];
            s_edges[pos] = val;
            s_ne++;
        }

        // Sort faces too
        // Simple bubble sort (small array)
        for (uint16_t i = 0; i < total_faces; ++i) {
            for (uint16_t j = i + 1; j < total_faces; ++j) {
                if (s_faces[j] < s_faces[i]) {
                    uint32_t tmp = s_faces[i];
                    s_faces[i] = s_faces[j];
                    s_faces[j] = tmp;
                }
            }
        }
    }
    __syncthreads();

    uint32_t nv = s_nv, ne = s_ne, nf = total_faces;

    // ── Stable partition: owned first, then not-owned ────────────────────
    // Thread 0 does the partition (small arrays)
    __shared__ uint16_t s_num_owned_f, s_num_owned_e, s_num_owned_v;

    if (threadIdx.x == 0) {
        // Partition faces: owned (face_patch[f] == p) first
        // Since s_faces is sorted, and owned faces are a subset, we can
        // stable_partition by swapping sections. Simpler: two-pass copy.
        uint32_t temp[K1_MAX_FACES];
        uint16_t owned = 0, notowned_start = 0;

        // Count owned faces
        for (uint16_t i = 0; i < nf; ++i)
            if (d_face_patch[s_faces[i]] == p) owned++;
        s_num_owned_f = owned;

        // Copy owned first, then not-owned (maintaining sort within each)
        uint16_t wi = 0;
        for (uint16_t i = 0; i < nf; ++i)
            if (d_face_patch[s_faces[i]] == p) temp[wi++] = s_faces[i];
        for (uint16_t i = 0; i < nf; ++i)
            if (d_face_patch[s_faces[i]] != p) temp[wi++] = s_faces[i];
        for (uint16_t i = 0; i < nf; ++i) s_faces[i] = temp[i];

        // Partition edges
        uint32_t temp_e[K1_MAX_EDGES];
        owned = 0;
        for (uint16_t i = 0; i < ne; ++i)
            if (d_edge_patch[s_edges[i]] == p) owned++;
        s_num_owned_e = owned;
        wi = 0;
        for (uint16_t i = 0; i < ne; ++i)
            if (d_edge_patch[s_edges[i]] == p) temp_e[wi++] = s_edges[i];
        for (uint16_t i = 0; i < ne; ++i)
            if (d_edge_patch[s_edges[i]] != p) temp_e[wi++] = s_edges[i];
        for (uint16_t i = 0; i < ne; ++i) s_edges[i] = temp_e[i];

        // Partition vertices
        uint32_t temp_v[K1_MAX_VERTS];
        owned = 0;
        for (uint16_t i = 0; i < nv; ++i)
            if (d_vertex_patch[s_verts[i]] == p) owned++;
        s_num_owned_v = owned;
        wi = 0;
        for (uint16_t i = 0; i < nv; ++i)
            if (d_vertex_patch[s_verts[i]] == p) temp_v[wi++] = s_verts[i];
        for (uint16_t i = 0; i < nv; ++i)
            if (d_vertex_patch[s_verts[i]] != p) temp_v[wi++] = s_verts[i];
        for (uint16_t i = 0; i < nv; ++i) s_verts[i] = temp_v[i];
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
