// Cross collapse kernel: remove vertices with valence 3 or 4.
// Based on VCG's CollapseCrosses (isotropic_remeshing.h:1093-1132).
// Uses RXMesh's CavityOp::EV with atomicCAS conflict resolution
// (same pattern as the official collapse kernel in RXMesh/apps/Remesh/collapse.cuh).

#pragma once

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.h"
#include "rxmesh/rxmesh_dynamic.h"
#include "rxmesh/util/macros.h"

#include "feature_remesh.cuh"  // for EdgeStatus, fill_n, is_done

// Kernel: collapse edges incident to low-valence vertices.
// For each vertex with valence 3 or 4 (not boundary, not feature):
//   find shortest incident edge → claim via atomicCAS → collapse.
template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    cross_collapse_kernel(
        rxmesh::Context                       context,
        const rxmesh::VertexAttribute<T>      coords,
        rxmesh::EdgeAttribute<EdgeStatus>     edge_status,
        rxmesh::VertexAttribute<bool>         v_boundary,
        const rxmesh::VertexAttribute<int>    vertex_is_feature,
        const T                               high_edge_len_sq)
{
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    CavityManager<blockThreads, CavityOp::EV> cavity(
        block, context, shrd_alloc, true);

    const uint32_t pid = cavity.patch_id();
    if (pid == INVALID32) return;

    Bitmask is_updated(cavity.patch_info().edges_capacity, shrd_alloc);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    // Per-vertex info for atomicCAS conflict resolution
    uint16_t* v_info =
        shrd_alloc.alloc<uint16_t>(2 * cavity.patch_info().vertices_capacity);
    fill_n<blockThreads>(
        v_info, 2 * cavity.patch_info().vertices_capacity, uint16_t(INVALID16));

    Bitmask e_collapse(cavity.patch_info().edges_capacity, shrd_alloc);
    e_collapse.reset(block);
    block.sync();

    // Compute vertex valence via VV query
    Query<blockThreads> query(context, pid);
    query.compute_vertex_valence(block, shrd_alloc);
    block.sync();

    // Mark edges for cross collapse: for each low-valence vertex,
    // find its shortest incident edge and claim it.
    auto select_cross_edge = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        if (edge_status(eh) != UNSEEN) return;

        assert(iter.size() == 4);
        const VertexHandle v0 = iter[0], v1 = iter[2];

        if (!v0.is_valid() || !v1.is_valid()) return;

        // Check if either endpoint is a cross vertex (valence 3 or 4,
        // not boundary, not feature)
        auto is_cross = [&](VertexHandle v) -> bool {
            if (v_boundary(v)) return false;
            if (vertex_is_feature(v)) return false;
            uint16_t val = query.vertex_valence(v);
            return (val == 3 || val == 4);
        };

        bool v0_cross = is_cross(v0);
        bool v1_cross = is_cross(v1);

        if (!v0_cross && !v1_cross) return;

        // Try to claim one vertex via atomicCAS
        const uint16_t c0(iter.local(0)), c1(iter.local(2));

        // Prefer claiming the cross vertex
        if (v0_cross) {
            uint16_t ret = ::atomicCAS(v_info + 2 * c0, INVALID16, c1);
            if (ret == INVALID16) {
                v_info[2 * c0 + 1] = eh.local_id();
                e_collapse.set(eh.local_id(), true);
                return;
            }
        }
        if (v1_cross) {
            uint16_t ret = ::atomicCAS(v_info + 2 * c1, INVALID16, c0);
            if (ret == INVALID16) {
                v_info[2 * c1 + 1] = eh.local_id();
                e_collapse.set(eh.local_id(), true);
            }
        }
    };

    query.dispatch<Op::EVDiamond>(block, shrd_alloc, select_cross_edge);
    block.sync();

    // Link condition via VV query
    auto check_edges = [&](const VertexHandle& vh, const VertexIterator& iter) {
        uint16_t opposite_v = v_info[2 * vh.local_id()];
        if (opposite_v != INVALID16) {
            int num_shared_v = 0;
            const VertexIterator opp_iter =
                query.template get_iterator<VertexIterator>(opposite_v);

            for (uint16_t v = 0; v < iter.size(); ++v) {
                for (uint16_t ov = 0; ov < opp_iter.size(); ++ov) {
                    if (iter.local(v) == opp_iter.local(ov)) {
                        num_shared_v++;
                        break;
                    }
                }
            }
            if (num_shared_v > 2) {
                e_collapse.reset(v_info[2 * vh.local_id() + 1], true);
            }
        }
    };

    query.dispatch<Op::VV>(block, shrd_alloc, check_edges,
        [](VertexHandle) { return true; }, false, true);
    block.sync();

    // One-per-face limit (matches CPU's break-after-first per face)
    for_each_face(cavity.patch_info(), [&](FaceHandle fh) {
        const uint16_t f = fh.local_id();
        const uint16_t e0 = cavity.patch_info().fe[3 * f + 0].id;
        const uint16_t e1 = cavity.patch_info().fe[3 * f + 1].id;
        const uint16_t e2 = cavity.patch_info().fe[3 * f + 2].id;

        int count = int(e_collapse(e0)) + int(e_collapse(e1)) + int(e_collapse(e2));
        if (count > 1) {
            // Keep only first marked, unmark others
            bool kept = false;
            if (e_collapse(e0)) { kept = true; }
            if (e_collapse(e1)) { if (kept) e_collapse.reset(e1, true); else kept = true; }
            if (e_collapse(e2)) { if (kept) e_collapse.reset(e2, true); }
        }
    });
    block.sync();

    // Create cavities
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (e_collapse(eh.local_id())) {
            cavity.create(eh);
        } else {
            edge_status(eh) = SKIP;
        }
    });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    // Cavity fill — collapse to midpoint
    if (cavity.prologue(block, shrd_alloc, coords, edge_status, v_boundary)) {
        is_updated.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);
            VertexHandle v0, v1;
            cavity.get_vertices(src, v0, v1);

            const vec3<T> p0 = coords.to_glm<3>(v0);
            const vec3<T> p1 = coords.to_glm<3>(v1);
            const vec3<T> new_p((p0[0] + p1[0]) * T(0.5),
                                (p0[1] + p1[1]) * T(0.5),
                                (p0[2] + p1[2]) * T(0.5));

            // Quality checks (matching regular collapse)
            bool reject = false;
            for (uint16_t i = 0; i < size; ++i) {
                const vec3<T> bp0 = coords.to_glm<3>(cavity.get_cavity_vertex(c, i));
                const vec3<T> bp1 = coords.to_glm<3>(cavity.get_cavity_vertex(c, (i + 1) % size));

                // Long edge check
                if (glm::distance2(bp0, new_p) > high_edge_len_sq) { reject = true; break; }

                // Normal flip check
                vec3<T> old_n = glm::cross(bp0 - p0, bp1 - p0);
                vec3<T> new_n = glm::cross(bp0 - new_p, bp1 - new_p);
                T old_len = glm::length(old_n);
                T new_len = glm::length(new_n);

                if (old_len > T(1e-10) && new_len > T(1e-10)) {
                    if (glm::dot(old_n / old_len, new_n / new_len) < T(0.7)) { reject = true; break; }

                    // QualityRadii with absolute floor
                    constexpr T K = T(6.928203230275509);
                    T old_area = old_len * T(0.5);
                    T new_area = new_len * T(0.5);
                    T old_c2 = glm::distance2(bp0, bp1);
                    T old_denom = glm::distance2(bp0, p0) + glm::distance2(bp1, p0) + old_c2;
                    T new_denom = glm::distance2(bp0, new_p) + glm::distance2(bp1, new_p) + old_c2;
                    T oldQ = (old_denom > T(1e-20)) ? (K * old_area) / old_denom : T(0);
                    T newQ = (new_denom > T(1e-20)) ? (K * new_area) / new_denom : T(0);
                    if (newQ < T(0.5) * oldQ) { reject = true; break; }
                    if (newQ < T(0.3)) { reject = true; break; }
                }
            }

            if (reject) {
                cavity.recover(src);
                edge_status(src) = SKIP;
            } else {
                const VertexHandle new_v = cavity.add_vertex();
                if (new_v.is_valid()) {
                    coords(new_v, 0) = new_p[0];
                    coords(new_v, 1) = new_p[1];
                    coords(new_v, 2) = new_p[2];

                    DEdgeHandle e0 = cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));
                    if (e0.is_valid()) {
                        is_updated.set(e0.local_id(), true);
                        const DEdgeHandle e_init = e0;

                        for (uint16_t i = 0; i < size; ++i) {
                            const DEdgeHandle e = cavity.get_cavity_edge(c, i);
                            const DEdgeHandle e1 =
                                (i == size - 1) ?
                                    e_init.get_flip_dedge() :
                                    cavity.add_edge(cavity.get_cavity_vertex(c, i + 1), new_v);
                            if (!e1.is_valid()) break;
                            if (i != size - 1) is_updated.set(e1.local_id(), true);
                            const FaceHandle new_f = cavity.add_face(e0, e, e1);
                            if (!new_f.is_valid()) break;
                            e0 = e1.get_flip_dedge();
                        }
                    }
                }
            }
        });
    }
    block.sync();

    cavity.epilogue(block);
    block.sync();

    if (cavity.is_successful()) {
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (is_updated(eh.local_id()) || cavity.is_recovered(eh)) {
                edge_status(eh) = ADDED;
            }
        });
    }
}

// Wrapper: run cross collapse until convergence
template <typename T>
inline void collapse_crosses(
    rxmesh::RXMeshDynamic&             rx,
    rxmesh::VertexAttribute<T>*        coords,
    rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
    rxmesh::VertexAttribute<bool>*     v_boundary,
    rxmesh::VertexAttribute<int>*      vertex_is_feature,
    const T                            high_edge_len_sq,
    rxmesh::Timers<rxmesh::GPUTimer>&  timers,
    int*                               d_buffer,
    bool                               verbose = false)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    LaunchBox<blockThreads> lb;
    rx.update_launch_box({Op::EVDiamond, Op::VV}, lb,
        (void*)cross_collapse_kernel<T, blockThreads>,
        true, false, true,  // is_dyn, oriented, with_vertex_valence
        false);

    int total_removed = 0;
    for (int pass = 0; pass < 1; pass++) {
        edge_status->reset(UNSEEN, DEVICE);
        uint32_t pre_v = rx.get_num_vertices(true);

        rx.reset_scheduler();
        while (!rx.is_queue_empty()) {
            cross_collapse_kernel<T, blockThreads>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    rx.get_context(), *coords, *edge_status, *v_boundary,
                    *vertex_is_feature, high_edge_len_sq);

            rx.cleanup();
            rx.slice_patches(*coords, *edge_status, *v_boundary);
            rx.cleanup();
        }

        uint32_t post_v = rx.get_num_vertices(true);
        int removed = pre_v - post_v;
        total_removed += removed;

        if (verbose)
            fprintf(stderr, "    [gpu] crosses pass %d: V %u → %u (-%d)\n",
                    pass, pre_v, post_v, removed);

        if (removed == 0) break;
    }

    if (verbose)
        fprintf(stderr, "    [gpu] crosses total: -%d vertices\n", total_removed);
}
