// Feature-aware remeshing kernels — copies of RXMesh's split/collapse/flip
// with added EdgeAttribute<int> edge_is_feature to skip feature edges.
//
// Changes from originals marked with "// FEATURE:"

#pragma once
#include <cuda_profiler_api.h>

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.h"

// Use RXMESH_CUDA_ERROR instead of CUDA_ERROR to avoid include issues
#ifndef RXMESH_CUDA_CHECK
#define RXMESH_CUDA_CHECK(err) do { \
    cudaError_t e_ = (err); \
    if (e_ != cudaSuccess) \
        fprintf(stderr, "[%s:%d] CUDA Error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
} while(0)
#endif
#include "rxmesh/rxmesh_dynamic.h"

#include "Remesh/util.cuh"
#include "Remesh/link_condition.cuh"
#include "checkpoint.h"

// Helper: snapshot current RXMeshDynamic state into a Checkpoint.
// Exports via OBJ round-trip (clean, consistent vertex/face indexing).
inline Checkpoint snapshot_mesh(
    rxmesh::RXMeshDynamic& rx,
    rxmesh::VertexAttribute<float>* coords,
    const std::string& name,
    const std::string& description,
    double elapsed_ms = 0)
{
    using namespace rxmesh;
    Checkpoint cp;
    cp.name = name;
    cp.description = description;
    cp.elapsed_ms = elapsed_ms;

    // Export to temp OBJ and re-read for clean indexing
    auto tmp = std::filesystem::temp_directory_path() / ("_ckpt_" + name + ".obj");
    rx.update_host();
    coords->move(DEVICE, HOST);
    rx.export_obj(tmp.string(), *coords);

    std::vector<std::vector<float>> verts;
    std::vector<std::vector<uint32_t>> faces;
    import_obj(tmp.string(), verts, faces);
    std::filesystem::remove(tmp);

    cp.num_vertices = verts.size();
    cp.num_faces = faces.size();
    cp.vertices.resize(cp.num_vertices * 3);
    cp.faces.resize(cp.num_faces * 3);
    for (int i = 0; i < cp.num_vertices; i++)
        for (int j = 0; j < 3; j++)
            cp.vertices[i*3+j] = verts[i][j];
    for (int i = 0; i < cp.num_faces; i++)
        for (int j = 0; j < 3; j++)
            cp.faces[i*3+j] = faces[i][j];

    coords->move(HOST, DEVICE);
    return cp;
}

// Helper: add per-vertex scalar from RXMesh attribute to a checkpoint
template <typename T>
inline void checkpoint_add_vertex_scalar(
    Checkpoint& cp,
    rxmesh::RXMeshDynamic& rx,
    rxmesh::VertexAttribute<T>* attr,
    const std::string& name)
{
    using namespace rxmesh;
    attr->move(DEVICE, HOST);
    rx.update_host();

    // Map to global vertex indices via export order
    std::vector<double> data(cp.num_vertices, 0.0);
    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t gid = rx.map_to_global(vh);
        if (gid < (uint32_t)cp.num_vertices)
            data[gid] = static_cast<double>((*attr)(vh, 0));
    });
    cp.vertex_scalars[name] = std::move(data);
    attr->move(HOST, DEVICE);
}

// Helper: add per-edge scalar (mapped to per-face for visualization)
// Each face edge gets the edge's scalar value
template <typename T>
inline void checkpoint_add_edge_as_face_scalar(
    Checkpoint& cp,
    rxmesh::RXMeshDynamic& rx,
    rxmesh::EdgeAttribute<T>* attr,
    const std::string& name)
{
    using namespace rxmesh;
    // For simplicity, store max of the 3 edge values per face
    attr->move(DEVICE, HOST);
    rx.update_host();

    std::vector<double> data(cp.num_faces, 0.0);
    // Can't easily map edges to faces without queries on host.
    // Just store as zeros — the OBJ + raw files are the real output.
    cp.face_scalars[name] = std::move(data);
    attr->move(HOST, DEVICE);
}

// compute_valence from Remesh/flip.cuh (copied to avoid ODR with non-feature flip)
template <uint32_t blockThreads>
__global__ static void feature_compute_valence(
    rxmesh::Context                        context,
    const rxmesh::VertexAttribute<uint8_t> v_valence)
{
    using namespace rxmesh;
    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;
    Query<blockThreads> query(context);
    query.compute_vertex_valence(block, shrd_alloc);
    block.sync();
    for_each_vertex(query.get_patch_info(), [&](VertexHandle vh) {
        v_valence(vh) = query.vertex_valence(vh);
    });
}

// =========================================================================
// Feature-aware edge split
// =========================================================================

// Based on official RXMesh split kernel (RXMesh/apps/Remesh/split.cuh).
// Differences from official: no min_new_edge_len check (CPU doesn't have it),
// no boundary vertex skip (CPU splits boundary edges).
template <typename T, uint32_t blockThreads>
__global__ static void feature_edge_split(
    rxmesh::Context                       context,
    const rxmesh::VertexAttribute<T>      coords,
    rxmesh::EdgeAttribute<EdgeStatus>     edge_status,
    rxmesh::VertexAttribute<bool>         v_boundary,
    const T high_edge_len_sq,
    const T low_edge_len_sq,
    int     iteration)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, true);

    if (cavity.patch_id() == INVALID32) return;

    Bitmask is_updated(cavity.patch_info().edges_capacity, shrd_alloc);
    is_updated.reset(block);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    auto should_split = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        assert(iter.size() == 4);

        if (edge_status(eh) == UNSEEN) {
            const VertexHandle va = iter[0];
            const VertexHandle vb = iter[2];
            const VertexHandle vc = iter[1];
            const VertexHandle vd = iter[3];

            // Don't split boundary edges (no adjacent face on one side)
            if (!vc.is_valid() || !vd.is_valid() || !va.is_valid() || !vb.is_valid()) {
                edge_status(eh) = SKIP;
                return;
            }

            // Degenerate cases
            if (va == vb || vb == vc || vc == va || va == vd || vb == vd || vc == vd) {
                edge_status(eh) = SKIP;
                return;
            }

            const T edge_len = glm::distance2(coords.to_glm<3>(va), coords.to_glm<3>(vb));

            if (edge_len > high_edge_len_sq) {
                // No min_new_edge_len check — matches CPU behavior
                cavity.create(eh);
            } else {
                edge_status(eh) = SKIP;
            }
        }
    };

    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_split);
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    // Only 3 attributes in prologue (RXMesh limit).
    // Feature flags are re-detected after split via dihedral angle.
    if (cavity.prologue(block, shrd_alloc, coords, edge_status, v_boundary)) {
        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);
            const VertexHandle v0 = cavity.get_cavity_vertex(c, 0);
            const VertexHandle v1 = cavity.get_cavity_vertex(c, 2);
            const VertexHandle new_v = cavity.add_vertex();

            if (new_v.is_valid()) {
                coords(new_v, 0) = (coords(v0, 0) + coords(v1, 0)) * T(0.5);
                coords(new_v, 1) = (coords(v0, 1) + coords(v1, 1)) * T(0.5);
                coords(new_v, 2) = (coords(v0, 2) + coords(v1, 2)) * T(0.5);

                DEdgeHandle e0 = cavity.add_edge(new_v, cavity.get_cavity_vertex(c, 0));
                const DEdgeHandle e_init = e0;

                if (e0.is_valid()) {
                    is_updated.set(e0.local_id(), true);
                    for (uint16_t i = 0; i < size; ++i) {
                        const DEdgeHandle e = cavity.get_cavity_edge(c, i);
                        const DEdgeHandle e1 =
                            (i == size - 1) ?
                                e_init.get_flip_dedge() :
                                cavity.add_edge(cavity.get_cavity_vertex(c, i + 1), new_v);
                        if (!e1.is_valid()) break;
                        is_updated.set(e1.local_id(), true);
                        const FaceHandle f = cavity.add_face(e0, e, e1);
                        if (!f.is_valid()) break;
                        e0 = e1.get_flip_dedge();
                    }
                }
            }
        });
    }

    cavity.epilogue(block);
    block.sync();

    if (cavity.is_successful()) {
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (is_updated(eh.local_id())) {
                edge_status(eh) = ADDED;
            }
        });
    }
}


// =========================================================================
// Edge collapse — uses official RXMesh pattern with atomicCAS conflict resolution.
// Feature edges are pre-marked as SKIP via pre_skip_feature_edges() before launch.
// =========================================================================

// Official RXMesh collapse pattern — atomicCAS conflict resolution + VV link condition.
// Feature edges are pre-marked as SKIP before this kernel launches.
template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    feature_edge_collapse(
        rxmesh::Context                       context,
        const rxmesh::VertexAttribute<T>      coords,
        rxmesh::EdgeAttribute<EdgeStatus>     edge_status,
        rxmesh::VertexAttribute<bool>         v_boundary,
        const rxmesh::VertexAttribute<int>    vertex_is_feature,
        const T                               low_edge_len_sq,
        const T                               high_edge_len_sq)
{
    using namespace rxmesh;
    auto           block = cooperative_groups::this_thread_block();
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

    // 1. Mark short edges for collapse using atomicCAS conflict resolution
    auto should_collapse = [&](const EdgeHandle& eh, const VertexIterator& iter) {
        if (edge_status(eh) == UNSEEN) {
            assert(iter.size() == 4);

            const VertexHandle v0 = iter[0], v1 = iter[2];
            const VertexHandle v2 = iter[1], v3 = iter[3];

            if (!v0.is_valid() || !v1.is_valid() || !v2.is_valid() || !v3.is_valid())
                return;
            // No boundary skip — CPU doesn't skip boundary vertices.

            // Feature vertex check: both immovable → skip. One moveable → allow.
            {
                bool v0_move = (vertex_is_feature(v0) == 0);
                bool v1_move = (vertex_is_feature(v1) == 0);
                if (!v0_move && !v1_move) return;
            }
            if (v0 == v1 || v0 == v2 || v0 == v3 || v1 == v2 || v1 == v3 || v2 == v3)
                return;

            const vec3<T> pp0 = coords.to_glm<3>(v0);
            const vec3<T> pp1 = coords.to_glm<3>(v1);
            const T edge_len_sq = glm::distance2(pp0, pp1);

            // Area check: also collapse edges in tiny-area faces (matches CPU)
            const vec3<T> pp2 = coords.to_glm<3>(v2);
            const T area0 = glm::length(glm::cross(pp1 - pp0, pp2 - pp0)) * T(0.5);
            bool tiny_area = (area0 < low_edge_len_sq / T(100));
            if (!tiny_area && v3.is_valid()) {
                const vec3<T> pp3 = coords.to_glm<3>(v3);
                const T area1 = glm::length(glm::cross(pp1 - pp0, pp3 - pp0)) * T(0.5);
                tiny_area = (area1 < low_edge_len_sq / T(100));
            }

            if (edge_len_sq < low_edge_len_sq || tiny_area) {
                const uint16_t c0(iter.local(0)), c1(iter.local(2));

                uint16_t ret = ::atomicCAS(v_info + 2 * c0, INVALID16, c1);
                if (ret == INVALID16) {
                    v_info[2 * c0 + 1] = eh.local_id();
                    e_collapse.set(eh.local_id(), true);
                } else {
                    ret = ::atomicCAS(v_info + 2 * c1, INVALID16, c0);
                    if (ret == INVALID16) {
                        v_info[2 * c1 + 1] = eh.local_id();
                        e_collapse.set(eh.local_id(), true);
                    }
                }
            }
        }
    };

    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_collapse);
    block.sync();

    // 2. Link condition check via VV query
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

    // 2b. One-per-face limit (matches CPU's break-after-first-collapse-per-face)
    for_each_face(cavity.patch_info(), [&](FaceHandle fh) {
        const uint16_t f = fh.local_id();
        const uint16_t e0 = cavity.patch_info().fe[3 * f + 0].id;
        const uint16_t e1 = cavity.patch_info().fe[3 * f + 1].id;
        const uint16_t e2 = cavity.patch_info().fe[3 * f + 2].id;

        int count = int(e_collapse(e0)) + int(e_collapse(e1)) + int(e_collapse(e2));
        if (count > 1) {
            // Keep only the shortest marked edge, unmark others
            // Read edge endpoints from EV to compute lengths
            auto get_len_sq = [&](uint16_t eid) -> T {
                const uint16_t va = cavity.patch_info().ev[2 * eid].id;
                const uint16_t vb = cavity.patch_info().ev[2 * eid + 1].id;
                return glm::distance2(
                    coords.to_glm<3>(VertexHandle(pid, va)),
                    coords.to_glm<3>(VertexHandle(pid, vb)));
            };

            T len0 = e_collapse(e0) ? get_len_sq(e0) : std::numeric_limits<T>::max();
            T len1 = e_collapse(e1) ? get_len_sq(e1) : std::numeric_limits<T>::max();
            T len2 = e_collapse(e2) ? get_len_sq(e2) : std::numeric_limits<T>::max();

            // Keep shortest
            if (len0 <= len1 && len0 <= len2) {
                if (e_collapse(e1)) e_collapse.reset(e1, true);
                if (e_collapse(e2)) e_collapse.reset(e2, true);
            } else if (len1 <= len0 && len1 <= len2) {
                if (e_collapse(e0)) e_collapse.reset(e0, true);
                if (e_collapse(e2)) e_collapse.reset(e2, true);
            } else {
                if (e_collapse(e0)) e_collapse.reset(e0, true);
                if (e_collapse(e1)) e_collapse.reset(e1, true);
            }
        }
    });
    block.sync();

    // 3. Create cavities
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (e_collapse(eh.local_id())) {
            cavity.create(eh);
        } else {
            edge_status(eh) = SKIP;
        }
    });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    // 4. Prologue + cavity fill
    if (cavity.prologue(block, shrd_alloc, coords, edge_status, v_boundary)) {
        is_updated.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);
            VertexHandle v0, v1;
            cavity.get_vertices(src, v0, v1);

            const vec3<T> p0 = coords.to_glm<3>(v0);
            const vec3<T> p1 = coords.to_glm<3>(v1);

            // Collapse toward immovable feature vertex (matches CPU)
            bool v0_feat = (vertex_is_feature(v0) != 0);
            bool v1_feat = (vertex_is_feature(v1) != 0);
            vec3<T> new_p;
            if (v0_feat && !v1_feat)
                new_p = p0;  // v0 immovable, collapse toward it
            else if (v1_feat && !v0_feat)
                new_p = p1;  // v1 immovable, collapse toward it
            else
                new_p = vec3<T>((p0[0] + p1[0]) * T(0.5),
                                (p0[1] + p1[1]) * T(0.5),
                                (p0[2] + p1[2]) * T(0.5));

            // Quality checks matching CPU (checkFacesAfterCollapse):
            // 1. Long edge check — don't create edges > split threshold
            // 2. Normal flip check — dot(oldN, newN) >= 0.7
            // 3. Quality check — new area >= 50% of old area
            bool reject = false;
            for (uint16_t i = 0; i < size; ++i) {
                const vec3<T> bp0 = coords.to_glm<3>(cavity.get_cavity_vertex(c, i));
                const vec3<T> bp1 = coords.to_glm<3>(cavity.get_cavity_vertex(c, (i + 1) % size));

                // Long edge check
                if (glm::distance2(bp0, new_p) > high_edge_len_sq) { reject = true; break; }

                // Old face normal (using v0 as reference — one of the two original verts)
                vec3<T> old_n = glm::cross(bp0 - p0, bp1 - p0);
                vec3<T> new_n = glm::cross(bp0 - new_p, bp1 - new_p);

                T old_len = glm::length(old_n);
                T new_len = glm::length(new_n);

                if (old_len > T(1e-10) && new_len > T(1e-10)) {
                    // Normal flip check
                    if (glm::dot(old_n / old_len, new_n / new_len) < T(0.7)) { reject = true; break; }

                    // QualityRadii: Q = 4*sqrt(3)*area / (a²+b²+c²)
                    // Compare new vs old quality, reject if < 50%
                    T old_area = old_len * T(0.5);  // |cross|/2
                    T new_area = new_len * T(0.5);
                    T old_a2 = glm::distance2(bp0, p0);
                    T old_b2 = glm::distance2(bp1, p0);
                    T old_c2 = glm::distance2(bp0, bp1);
                    T new_a2 = glm::distance2(bp0, new_p);
                    T new_b2 = glm::distance2(bp1, new_p);
                    // QualityRadii: Q = 4*sqrt(3)*area / (a²+b²+c²)
                    constexpr T K = T(6.928203230275509);  // 4*sqrt(3)
                    T old_denom = old_a2 + old_b2 + old_c2;
                    T new_denom = new_a2 + new_b2 + old_c2;
                    T oldQ = (old_denom > T(1e-20)) ? (K * old_area) / old_denom : T(0);
                    T newQ = (new_denom > T(1e-20)) ? (K * new_area) / new_denom : T(0);

                    // Relative check: reject if quality drops below 50%
                    if (newQ < T(0.5) * oldQ) { reject = true; break; }
                    // Absolute floor: reject if quality below 0.3 (matches CPU aspectRatioThr)
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


// =========================================================================
// Feature-aware edge flip
// =========================================================================

template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    feature_edge_flip(
        rxmesh::Context                        context,
        const rxmesh::VertexAttribute<T>       coords,
        const rxmesh::VertexAttribute<uint8_t> v_valence,
        rxmesh::EdgeAttribute<EdgeStatus>      edge_status,
        const rxmesh::EdgeAttribute<int>       edge_is_feature,  // FEATURE: added
        int*                                   d_buffer)
{
    using namespace rxmesh;

    auto block = cooperative_groups::this_thread_block();
    ShmemAllocator shrd_alloc;

    CavityManager<blockThreads, CavityOp::E> cavity(
        block, context, shrd_alloc, false, false);

    if (cavity.patch_id() == INVALID32) return;

    Bitmask edge_mask(cavity.patch_info().edges_capacity, shrd_alloc);
    edge_mask.reset(block);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    // Use vertices_capacity (not num_vertices[0]) to avoid shared memory
    // overflow when vertex local IDs exceed live count after compaction.
    Bitmask v0_mask(cavity.patch_info().vertices_capacity, shrd_alloc);
    Bitmask v1_mask(cavity.patch_info().vertices_capacity, shrd_alloc);

    Query<blockThreads> query(context, cavity.patch_id());
    query.prologue<Op::EVDiamond>(block, shrd_alloc);
    block.sync();

    // 1. mark edges to flip
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        const VertexIterator iter =
            query.template get_iterator<VertexIterator>(eh.local_id());

        if (edge_status(eh) == UNSEEN && iter[1].is_valid() && iter[3].is_valid()) {

            // FEATURE: don't flip feature edges
            if (edge_is_feature(eh) == 1) {
                edge_status(eh) = SKIP;
                return;
            }

            if (iter[0] == iter[1] || iter[0] == iter[2] || iter[0] == iter[3] ||
                iter[1] == iter[2] || iter[1] == iter[3] || iter[2] == iter[3])
                return;

            constexpr int target_valence = 6;

            const int va = v_valence(iter[0]);
            const int vb = v_valence(iter[2]);
            const int vc = v_valence(iter[1]);
            const int vd = v_valence(iter[3]);

            const int dev_pre =
                (va - target_valence)*(va - target_valence) +
                (vb - target_valence)*(vb - target_valence) +
                (vc - target_valence)*(vc - target_valence) +
                (vd - target_valence)*(vd - target_valence);

            const int dev_post =
                (va-1 - target_valence)*(va-1 - target_valence) +
                (vb-1 - target_valence)*(vb-1 - target_valence) +
                (vc+1 - target_valence)*(vc+1 - target_valence) +
                (vd+1 - target_valence)*(vd+1 - target_valence);

            if (dev_pre > dev_post) {
                edge_mask.set(eh.local_id(), true);
            }
        }
    });
    block.sync();

    // 2. link condition
    link_condition(block, cavity.patch_info(), query, edge_mask, v0_mask, v1_mask, 0, 2);
    block.sync();

    // 3. create cavities
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (edge_mask(eh.local_id())) {
            cavity.create(eh);
        } else {
            edge_status(eh) = SKIP;
        }
    });
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

    if (cavity.prologue(block, shrd_alloc, coords, edge_status)) {
        edge_mask.reset(block);
        block.sync();

        cavity.for_each_cavity(block, [&](uint16_t c, uint16_t size) {
            assert(size == 4);
            DEdgeHandle new_edge = cavity.add_edge(
                cavity.get_cavity_vertex(c, 1), cavity.get_cavity_vertex(c, 3));

            if (new_edge.is_valid()) {
                edge_mask.set(new_edge.local_id(), true);
                cavity.add_face(cavity.get_cavity_edge(c, 0),
                                new_edge, cavity.get_cavity_edge(c, 3));
                cavity.add_face(cavity.get_cavity_edge(c, 1),
                                cavity.get_cavity_edge(c, 2),
                                new_edge.get_flip_dedge());
            }
        });
    }
    block.sync();

    cavity.epilogue(block);
    block.sync();

    if (cavity.is_successful()) {
        for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
            if (edge_mask(eh.local_id())) {
                edge_status(eh) = ADDED;
            }
        });
    }
}


// =========================================================================
// Adaptive sizing: compute per-vertex quality → sizing multiplier
// Matches VCG's computeQualityDistFromRadii + ClampedLerp(0.3, 3.0)
// =========================================================================

// Per-face: compute triangle quality = 1 - inradius/circumradius
// Scatter to vertices via atomicAdd (we accumulate quality + count separately)
template <typename T, uint32_t blockThreads>
__global__ static void compute_face_quality_kernel(
    rxmesh::Context              context,
    const rxmesh::VertexAttribute<T> coords,
    rxmesh::VertexAttribute<T>   v_quality_sum,
    rxmesh::VertexAttribute<int> v_quality_count)
{
    auto compute = [&](rxmesh::FaceHandle fh, rxmesh::VertexIterator& fv) {
        using namespace rxmesh;
        vec3<T> p0 = coords.to_glm<3>(fv[0]);
        vec3<T> p1 = coords.to_glm<3>(fv[1]);
        vec3<T> p2 = coords.to_glm<3>(fv[2]);

        // Edge lengths
        T a = glm::length(p1 - p0);
        T b = glm::length(p2 - p1);
        T c = glm::length(p0 - p2);

        // Semi-perimeter and area
        T s = (a + b + c) * T(0.5);
        T area = glm::length(glm::cross(p1 - p0, p2 - p0)) * T(0.5);

        // Inradius = area / s, Circumradius = (a*b*c) / (4*area)
        // QualityRadii = inradius / circumradius = 4*area²*s / (a*b*c*s) ... simplified
        // VCG: QualityRadii = 8 * area² / (a*b*c * (a+b+c))  [= 2*inradius/circumradius actually]
        // But let's use the VCG formula directly:
        T quality_radii = T(0);
        if (a > T(1e-20) && b > T(1e-20) && c > T(1e-20) && area > T(1e-20)) {
            // inradius/circumradius ratio (0 = degenerate, 1 = equilateral)
            quality_radii = T(8) * area * area / (a * b * c * (a + b + c));
            quality_radii = min(quality_radii, T(1));
        }

        // face quality = 1 - quality_radii (1 = bad, 0 = good equilateral)
        T face_q = T(1) - quality_radii;

        // Scatter to all 3 vertices
        for (int i = 0; i < 3; i++) {
            atomicAdd(&v_quality_sum(fv[i], 0), face_q);
            atomicAdd(&v_quality_count(fv[i], 0), 1);
        }
    };

    auto block = cooperative_groups::this_thread_block();
    rxmesh::Query<blockThreads> query(context);
    rxmesh::ShmemAllocator shrd_alloc;
    query.dispatch<rxmesh::Op::FV>(block, shrd_alloc, compute);
}

// Laplacian smooth of vertex quality (1-ring average), 1 iteration
template <typename T, uint32_t blockThreads>
__global__ static void smooth_quality_kernel(
    rxmesh::Context              context,
    const rxmesh::VertexAttribute<T> quality_in,
    rxmesh::VertexAttribute<T>   quality_out)
{
    auto compute = [&](const rxmesh::VertexHandle& vh, const rxmesh::VertexIterator& iter) {
        T sum = quality_in(vh, 0);
        int count = 1;
        for (int i = 0; i < static_cast<int>(iter.size()); i++) {
            sum += quality_in(iter[i], 0);
            count++;
        }
        quality_out(vh, 0) = sum / T(count);
    };

    auto block = cooperative_groups::this_thread_block();
    rxmesh::Query<blockThreads> query(context);
    rxmesh::ShmemAllocator shrd_alloc;
    query.dispatch<rxmesh::Op::VV>(block, shrd_alloc, compute);
}

// Compute adaptive sizing field: quality → sizing multiplier in [minMult, maxMult]
inline void compute_adaptive_sizing(
    rxmesh::RXMeshDynamic&         rx,
    rxmesh::VertexAttribute<float>* coords,
    rxmesh::VertexAttribute<float>* sizing,
    float min_mult = 0.3f,
    float max_mult = 3.0f)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    auto v_qsum = rx.add_vertex_attribute<float>("_qsum", 1, DEVICE);
    auto v_qcnt = rx.add_vertex_attribute<int>("_qcnt", 1, DEVICE);
    auto v_qtmp = rx.add_vertex_attribute<float>("_qtmp", 1, DEVICE);

    v_qsum->reset(0.0f, DEVICE);
    v_qcnt->reset(0, DEVICE);
    // Reset sizing to neutral quality (0.5) before computation.
    // Ribbon vertices (not visited by for_each_vertex) keep this default.
    sizing->reset(0.5f, DEVICE);

    // Step 1: scatter face quality to vertices
    LaunchBox<blockThreads> lb;
    rx.update_launch_box({Op::FV}, lb,
        (void*)compute_face_quality_kernel<float, blockThreads>);
    fprintf(stderr, "  [sizing] FV kernel: lb.blocks=%d, lb.threads=%d\n",
            (int)lb.blocks, (int)lb.num_threads);
    compute_face_quality_kernel<float, blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), *coords, *v_qsum, *v_qcnt);
    RXMESH_CUDA_CHECK(cudaDeviceSynchronize());

    // DEBUG: check FV kernel output
    {
        rx.update_host();
        v_qsum->move(DEVICE, HOST);
        v_qcnt->move(DEVICE, HOST);
        float sum_q = 0; int sum_c = 0, nonzero_c = 0;
        rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
            sum_q += (*v_qsum)(vh, 0);
            sum_c += (*v_qcnt)(vh, 0);
            if ((*v_qcnt)(vh, 0) > 0) nonzero_c++;
        });
        fprintf(stderr, "  [sizing] qsum_total=%.4f, qcnt_total=%d, verts_with_count>0=%d/%d\n",
                sum_q, sum_c, nonzero_c, (int)rx.get_num_vertices());
        v_qsum->move(HOST, DEVICE);
        v_qcnt->move(HOST, DEVICE);
    }

    // Average: quality = sum / count. Set ribbon vertices (cnt=0) to 0.5 (neutral)
    rx.for_each_vertex(DEVICE,
        [v_qsum = *v_qsum, v_qcnt = *v_qcnt, sizing = *sizing]
        __device__(const VertexHandle vh) mutable {
            int cnt = v_qcnt(vh, 0);
            sizing(vh, 0) = (cnt > 0) ? v_qsum(vh, 0) / float(cnt) : 0.5f;
        });
    RXMESH_CUDA_CHECK(cudaDeviceSynchronize());

    // DEBUG: check after averaging
    {
        rx.update_host();
        sizing->move(DEVICE, HOST);
        float smin=999, smax=-999, savg=0; int scnt=0;
        rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
            float s = (*sizing)(vh, 0);
            smin = std::min(smin, s); smax = std::max(smax, s); savg += s; scnt++;
        });
        fprintf(stderr, "  [sizing] after avg: min=%.4f avg=%.4f max=%.4f (n=%d)\n",
                smin, savg/scnt, smax, scnt);
        sizing->move(HOST, DEVICE);
    }

    // Step 2: Laplacian smooth (2 iterations, matching VCG)
    rx.update_launch_box({Op::VV}, lb,
        (void*)smooth_quality_kernel<float, blockThreads>);

    for (int s = 0; s < 2; s++) {
        v_qtmp->copy_from(*sizing, DEVICE, DEVICE);
        smooth_quality_kernel<float, blockThreads>
            <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                rx.get_context(), *v_qtmp, *sizing);
        RXMESH_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Step 3: map quality [0,1] → sizing multiplier [minMult, maxMult]
    rx.for_each_vertex(DEVICE,
        [sizing = *sizing, min_mult, max_mult]
        __device__(const VertexHandle vh) mutable {
            float q = sizing(vh, 0);
            q = fminf(fmaxf(q, 0.0f), 1.0f);
            sizing(vh, 0) = min_mult + q * (max_mult - min_mult);
        });
    RXMESH_CUDA_CHECK(cudaDeviceSynchronize());

    // DEBUG: check after lerp
    {
        sizing->move(DEVICE, HOST);
        float smin=999, smax=-999, savg=0; int scnt=0;
        rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
            float s = (*sizing)(vh, 0);
            smin = std::min(smin, s); smax = std::max(smax, s); savg += s; scnt++;
        });
        fprintf(stderr, "  [sizing] after lerp: min=%.4f avg=%.4f max=%.4f (n=%d)\n",
                smin, savg/scnt, smax, scnt);
        sizing->move(HOST, DEVICE);
    }

    // Cleanup temp attributes
    rx.remove_attribute("_qsum");
    rx.remove_attribute("_qcnt");
    rx.remove_attribute("_qtmp");

    // DEBUG: check after remove_attribute (H4 test)
    {
        sizing->move(DEVICE, HOST);
        float smin=999, smax=-999; int scnt=0;
        rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
            float s = (*sizing)(vh, 0);
            smin = std::min(smin, s); smax = std::max(smax, s); scnt++;
        });
        fprintf(stderr, "  [sizing] after remove_attr: min=%.4f max=%.4f (n=%d)\n",
                smin, smax, scnt);
        sizing->move(HOST, DEVICE);
    }
}

// =========================================================================
// Mark feature vertices (any vertex touching a feature edge) via EV query
// =========================================================================

template <uint32_t blockThreads>
__global__ static void mark_feature_verts_kernel(
    rxmesh::Context                    context,
    const rxmesh::EdgeAttribute<int>   edge_feature,
    rxmesh::VertexAttribute<int>       vertex_feature)
{
    auto compute = [&](rxmesh::EdgeHandle& eh, const rxmesh::VertexIterator& iter) {
        if (edge_feature(eh) == 1) {
            if (iter[0].is_valid()) atomicMax(&vertex_feature(iter[0]), 1);
            if (iter[1].is_valid()) atomicMax(&vertex_feature(iter[1]), 1);
        }
    };
    auto block = cooperative_groups::this_thread_block();
    rxmesh::Query<blockThreads> query(context);
    rxmesh::ShmemAllocator shrd_alloc;
    query.dispatch<rxmesh::Op::EV>(block, shrd_alloc, compute);
}

inline void mark_feature_vertices(
    rxmesh::RXMeshDynamic&           rx,
    rxmesh::EdgeAttribute<int>*      edge_feature,
    rxmesh::VertexAttribute<int>*    vertex_feature)
{
    constexpr uint32_t blockThreads = 256;
    vertex_feature->reset(0, rxmesh::DEVICE);

    rxmesh::LaunchBox<blockThreads> lb;
    rx.update_launch_box({rxmesh::Op::EV}, lb,
        (void*)mark_feature_verts_kernel<blockThreads>);

    mark_feature_verts_kernel<blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), *edge_feature, *vertex_feature);
    RXMESH_CUDA_CHECK(cudaDeviceSynchronize());
}

// =========================================================================
// Erode/dilate feature edges (GPU, matching QuadWild's ErodeDilate)
// =========================================================================

// Count feature edges per vertex (feature valence)
template <uint32_t blockThreads>
__global__ static void feature_edge_valence_kernel(
    rxmesh::Context                    context,
    const rxmesh::EdgeAttribute<int>   edge_feature,
    rxmesh::VertexAttribute<int>       feat_valence)
{
    auto compute = [&](rxmesh::EdgeHandle& eh, const rxmesh::VertexIterator& iter) {
        if (edge_feature(eh) == 1) {
            if (iter[0].is_valid()) atomicAdd(&feat_valence(iter[0]), 1);
            if (iter[1].is_valid()) atomicAdd(&feat_valence(iter[1]), 1);
        }
    };
    auto block = cooperative_groups::this_thread_block();
    rxmesh::Query<blockThreads> query(context);
    rxmesh::ShmemAllocator shrd_alloc;
    query.dispatch<rxmesh::Op::EV>(block, shrd_alloc, compute);
}

// Erode: clear short dead-end feature edges (valence-2 endpoint)
template <typename T, uint32_t blockThreads>
__global__ static void feature_erode_kernel(
    rxmesh::Context                        context,
    const rxmesh::VertexAttribute<T>       coords,
    rxmesh::EdgeAttribute<int>             edge_feature,
    const rxmesh::VertexAttribute<int>     feat_valence,
    const rxmesh::VertexAttribute<int>     v_boundary,
    const T                                max_len)
{
    auto compute = [&](rxmesh::EdgeHandle& eh, const rxmesh::VertexIterator& iter) {
        if (edge_feature(eh) != 1) return;
        auto v0 = iter[0], v1 = iter[1];
        if (v_boundary(v0) && v_boundary(v1)) return;
        T len = glm::length(coords.to_glm<3>(v1) - coords.to_glm<3>(v0));
        if (len > max_len) return;
        if (feat_valence(v0) == 1 || feat_valence(v1) == 1)
            edge_feature(eh) = 0;
    };
    auto block = cooperative_groups::this_thread_block();
    rxmesh::Query<blockThreads> query(context);
    rxmesh::ShmemAllocator shrd_alloc;
    query.dispatch<rxmesh::Op::EV>(block, shrd_alloc, compute);
}

// Dilate: restore originally-feature edges at valence-2 non-junction vertices
template <uint32_t blockThreads>
__global__ static void feature_dilate_kernel(
    rxmesh::Context                        context,
    rxmesh::EdgeAttribute<int>             edge_feature,
    const rxmesh::EdgeAttribute<int>       edge_feature_orig,
    const rxmesh::VertexAttribute<int>     feat_valence,
    const rxmesh::VertexAttribute<int>     v_high_val)
{
    auto compute = [&](rxmesh::EdgeHandle& eh, const rxmesh::VertexIterator& iter) {
        if (edge_feature(eh) == 1) return;
        if (edge_feature_orig(eh) != 1) return;
        auto v0 = iter[0], v1 = iter[1];
        if ((feat_valence(v0) == 1 && !v_high_val(v0)) ||
            (feat_valence(v1) == 1 && !v_high_val(v1)))
            edge_feature(eh) = 1;
    };
    auto block = cooperative_groups::this_thread_block();
    rxmesh::Query<blockThreads> query(context);
    rxmesh::ShmemAllocator shrd_alloc;
    query.dispatch<rxmesh::Op::EV>(block, shrd_alloc, compute);
}

// Wrapper: run erode/dilate on RXMeshDynamic
inline void erode_dilate_features(
    rxmesh::RXMeshDynamic&            rx,
    rxmesh::VertexAttribute<float>*   coords,
    rxmesh::EdgeAttribute<int>*       edge_feature,
    int                               steps,
    float                             bbox_diag)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;
    float max_erode_len = bbox_diag * 0.05f;

    // Temp attributes
    auto feat_valence = rx.add_vertex_attribute<int>("_fv", 1, DEVICE);
    auto v_bd_bool = rx.add_vertex_attribute<bool>("_fvbd", 1);
    auto v_boundary = rx.add_vertex_attribute<int>("_fvbdi", 1, DEVICE);
    auto v_high_val = rx.add_vertex_attribute<int>("_fvhv", 1, DEVICE);
    auto edge_orig = rx.add_edge_attribute<int>("_feor", 1, DEVICE);

    // Save original features for dilate
    edge_orig->copy_from(*edge_feature, DEVICE, DEVICE);

    // Boundary detection
    rx.get_boundary_vertices(*v_bd_bool);
    rx.for_each_vertex(DEVICE,
        [vb = *v_bd_bool, vbi = *v_boundary] __device__(const VertexHandle vh) mutable {
            vbi(vh) = vb(vh) ? 1 : 0;
        });
    RXMESH_CUDA_CHECK(cudaDeviceSynchronize());

    // Prepare launch boxes
    LaunchBox<blockThreads> lb_val, lb_erode, lb_dilate;
    rx.update_launch_box({Op::EV}, lb_val, (void*)feature_edge_valence_kernel<blockThreads>);
    rx.update_launch_box({Op::EV}, lb_erode, (void*)feature_erode_kernel<float, blockThreads>);
    rx.update_launch_box({Op::EV}, lb_dilate, (void*)feature_dilate_kernel<blockThreads>);

    // Initial feature valence + mark high-valence junctions
    feat_valence->reset(0, DEVICE);
    feature_edge_valence_kernel<blockThreads>
        <<<lb_val.blocks, lb_val.num_threads, lb_val.smem_bytes_dyn>>>(
            rx.get_context(), *edge_feature, *feat_valence);
    RXMESH_CUDA_CHECK(cudaDeviceSynchronize());

    rx.for_each_vertex(DEVICE,
        [fv = *feat_valence, vb = *v_boundary, vh_out = *v_high_val]
        __device__(const VertexHandle vh) mutable {
            vh_out(vh) = (fv(vh) > 2 || (vb(vh) && fv(vh) > 1)) ? 1 : 0;
        });
    RXMESH_CUDA_CHECK(cudaDeviceSynchronize());

    // Erode steps
    for (int s = 0; s < steps; s++) {
        feat_valence->reset(0, DEVICE);
        feature_edge_valence_kernel<blockThreads>
            <<<lb_val.blocks, lb_val.num_threads, lb_val.smem_bytes_dyn>>>(
                rx.get_context(), *edge_feature, *feat_valence);
        RXMESH_CUDA_CHECK(cudaDeviceSynchronize());

        feature_erode_kernel<float, blockThreads>
            <<<lb_erode.blocks, lb_erode.num_threads, lb_erode.smem_bytes_dyn>>>(
                rx.get_context(), *coords, *edge_feature, *feat_valence,
                *v_boundary, max_erode_len);
        RXMESH_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Dilate steps
    for (int s = 0; s < steps; s++) {
        feat_valence->reset(0, DEVICE);
        feature_edge_valence_kernel<blockThreads>
            <<<lb_val.blocks, lb_val.num_threads, lb_val.smem_bytes_dyn>>>(
                rx.get_context(), *edge_feature, *feat_valence);
        RXMESH_CUDA_CHECK(cudaDeviceSynchronize());

        feature_dilate_kernel<blockThreads>
            <<<lb_dilate.blocks, lb_dilate.num_threads, lb_dilate.smem_bytes_dyn>>>(
                rx.get_context(), *edge_feature, *edge_orig,
                *feat_valence, *v_high_val);
        RXMESH_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Cleanup
    rx.remove_attribute("_fv");
    rx.remove_attribute("_fvbd");
    rx.remove_attribute("_fvbdi");
    rx.remove_attribute("_fvhv");
    rx.remove_attribute("_feor");
}

// =========================================================================
// Pre-mark feature edges as SKIP so is_done() doesn't count them
// =========================================================================

inline void pre_skip_feature_edges(
    rxmesh::RXMeshDynamic&             rx,
    rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
    rxmesh::EdgeAttribute<int>*        edge_is_feature)
{
    rx.for_each_edge(rxmesh::DEVICE,
        [edge_status = *edge_status, edge_is_feature = *edge_is_feature]
        __device__(const rxmesh::EdgeHandle eh) mutable {
            if (edge_is_feature(eh) == 1) {
                edge_status(eh) = SKIP;
            }
        });
    RXMESH_CUDA_CHECK(cudaDeviceSynchronize());
}

// =========================================================================
// Cross collapse: TODO — implement GPU kernel for collapsing low-valence
// interior vertices (valence 3 or 4). For now, this is handled on CPU
// via vcg_collapse_crosses() called from op_feature_remesh.cu between
// iterations (the mesh is already on host for BVH projection).
// =========================================================================

// =========================================================================
// Feature-aware wrapper functions (matching split_long_edges etc.)
// =========================================================================

template <typename T>
inline void feature_split_long_edges(
    rxmesh::RXMeshDynamic&             rx,
    rxmesh::VertexAttribute<T>*        coords,
    rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
    rxmesh::VertexAttribute<bool>*     v_boundary,
    const T                            high_edge_len_sq,
    const T                            low_edge_len_sq,
    rxmesh::Timers<rxmesh::GPUTimer>&  timers,
    int*                               d_buffer)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    edge_status->reset(UNSEEN, DEVICE);
    // Don't skip feature edges for split — they SHOULD be split (matching CPU).
    // Feature flag inheritance happens in the cavity fill.
    int prv_remaining_work = rx.get_num_edges();

    LaunchBox<blockThreads> lb;
    rx.update_launch_box({Op::EVDiamond}, lb,
        (void*)feature_edge_split<T, blockThreads>,
        true, false, false, false,
        [&](uint32_t v, uint32_t e, uint32_t f) {
            return detail::mask_num_bytes(e) + ShmemAllocator::default_alignment;
        });

    timers.start("SplitTotal");
    int split_outer = 0;
    while (true) {
        split_outer++;
        rx.reset_scheduler();
        int split_inner = 0;
        while (!rx.is_queue_empty()) {
            split_inner++;

            timers.start("Split");
            feature_edge_split<T, blockThreads>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    rx.get_context(), *coords, *edge_status, *v_boundary,
                    high_edge_len_sq, low_edge_len_sq, 0);
            timers.stop("Split");

            timers.start("SplitCleanup");
            rx.cleanup();
            timers.stop("SplitCleanup");
            timers.start("SplitSlice");
            // Use only 3 attrs (the official RXMesh-tested combo).
            // Extra attrs (edge_is_feature, sizing, vertex_is_feature)
            // are refreshed via pre_skip_feature_edges at start of each op.
            rx.slice_patches(*coords, *edge_status, *v_boundary);
            timers.stop("SplitSlice");
            timers.start("SplitCleanup");
            rx.cleanup();
            timers.stop("SplitCleanup");
        }
        int remaining = is_done(rx, edge_status, d_buffer);
        fprintf(stderr, "    [split] outer=%d inner=%d remaining=%d/%d\n",
                split_outer, split_inner, remaining, prv_remaining_work);
        if (remaining == 0 || prv_remaining_work == remaining) break;
        prv_remaining_work = remaining;
    }
    timers.stop("SplitTotal");
}

template <typename T>
inline void feature_collapse_short_edges(
    rxmesh::RXMeshDynamic&             rx,
    rxmesh::VertexAttribute<T>*        coords,
    rxmesh::EdgeAttribute<EdgeStatus>* edge_status,
    rxmesh::VertexAttribute<bool>*     v_boundary,
    rxmesh::EdgeAttribute<int>*        edge_is_feature,
    rxmesh::VertexAttribute<int>*      vertex_is_feature,
    const T                            low_edge_len_sq,
    const T                            high_edge_len_sq,
    rxmesh::Timers<rxmesh::GPUTimer>&  timers,
    int*                               d_buffer)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    edge_status->reset(UNSEEN, DEVICE);
    pre_skip_feature_edges(rx, edge_status, edge_is_feature);
    int prv_remaining_work = rx.get_num_edges();

    LaunchBox<blockThreads> lb;
    rx.update_launch_box({Op::EVDiamond, Op::VV}, lb,
        (void*)feature_edge_collapse<T, blockThreads>,
        true, false, false, false,
        [&](uint32_t v, uint32_t e, uint32_t f) {
            return detail::mask_num_bytes(e) +
                   2 * v * sizeof(uint16_t) +
                   2 * ShmemAllocator::default_alignment;
        });

    timers.start("CollapseTotal");
    int col_outer = 0;
    while (true) {
        col_outer++;
        rx.reset_scheduler();
        int col_inner = 0;
        while (!rx.is_queue_empty()) {
            col_inner++;
            timers.start("Collapse");
            feature_edge_collapse<T, blockThreads>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    rx.get_context(), *coords, *edge_status, *v_boundary,
                    *vertex_is_feature, low_edge_len_sq, high_edge_len_sq);
            timers.stop("Collapse");

            timers.start("CollapseCleanup");
            rx.cleanup();
            timers.stop("CollapseCleanup");
            timers.start("CollapseSlice");
            // Use only 3 attrs (the official RXMesh-tested combo).
            // Extra attrs (edge_is_feature, sizing, vertex_is_feature)
            // are refreshed via pre_skip_feature_edges at start of each op.
            rx.slice_patches(*coords, *edge_status, *v_boundary);
            timers.stop("CollapseSlice");
            timers.start("CollapseCleanup");
            rx.cleanup();
            timers.stop("CollapseCleanup");
        }
        int remaining = is_done(rx, edge_status, d_buffer);
        fprintf(stderr, "    [collapse] outer=%d inner=%d remaining=%d/%d\n",
                col_outer, col_inner, remaining, prv_remaining_work);
        if (remaining == 0 || prv_remaining_work == remaining) break;
        prv_remaining_work = remaining;
    }
    timers.stop("CollapseTotal");
}

template <typename T>
inline void feature_equalize_valences(
    rxmesh::RXMeshDynamic&                 rx,
    rxmesh::VertexAttribute<T>*            coords,
    rxmesh::VertexAttribute<uint8_t>*      v_valence,
    rxmesh::EdgeAttribute<EdgeStatus>*     edge_status,
    rxmesh::EdgeAttribute<int8_t>*         edge_link,
    rxmesh::VertexAttribute<bool>*         v_boundary,
    rxmesh::EdgeAttribute<int>*            edge_is_feature,
    rxmesh::VertexAttribute<int>*          vertex_is_feature,
    rxmesh::VertexAttribute<T>*            sizing,
    rxmesh::Timers<rxmesh::GPUTimer>&      timers,
    int*                                   d_buffer)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    edge_status->reset(UNSEEN, DEVICE);
    pre_skip_feature_edges(rx, edge_status, edge_is_feature);
    int prv_remaining_work = rx.get_num_edges();

    LaunchBox<blockThreads> lb_valence;
    rx.update_launch_box({Op::VV}, lb_valence,
        (void*)feature_compute_valence<blockThreads>);

    LaunchBox<blockThreads> lb;
    rx.update_launch_box({Op::EVDiamond}, lb,
        (void*)feature_edge_flip<T, blockThreads>,
        true, false, false, false,
        [&](uint32_t v, uint32_t e, uint32_t f) {
            return 2 * detail::mask_num_bytes(e) +
                   2 * v * sizeof(uint16_t) +
                   4 * ShmemAllocator::default_alignment;
        });

    timers.start("FlipTotal");
    int flip_outer = 0;
    while (true) {
        flip_outer++;
        rx.reset_scheduler();

        feature_compute_valence<blockThreads>
            <<<lb_valence.blocks, lb_valence.num_threads, lb_valence.smem_bytes_dyn>>>(
                rx.get_context(), *v_valence);
        RXMESH_CUDA_CHECK(cudaDeviceSynchronize());

        int flip_inner = 0;
        while (!rx.is_queue_empty()) {
            flip_inner++;
            timers.start("Flip");
            feature_edge_flip<T, blockThreads>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    rx.get_context(), *coords, *v_valence, *edge_status,
                    *edge_is_feature, d_buffer);
            timers.stop("Flip");

            timers.start("FlipCleanup");
            rx.cleanup();
            timers.stop("FlipCleanup");
            timers.start("FlipSlice");
            // Use only 3 attrs (the official RXMesh-tested combo).
            // Extra attrs (edge_is_feature, sizing, vertex_is_feature)
            // are refreshed via pre_skip_feature_edges at start of each op.
            rx.slice_patches(*coords, *edge_status, *v_boundary);
            timers.stop("FlipSlice");
            timers.start("FlipCleanup");
            rx.cleanup();
            timers.stop("FlipCleanup");
        }
        int remaining = is_done(rx, edge_status, d_buffer);
        //fprintf(stderr, "    [flip] outer=%d inner=%d remaining=%d/%d\n",
        //        flip_outer, flip_inner, remaining, prv_remaining_work);
        if (remaining == 0 || prv_remaining_work == remaining) break;
        prv_remaining_work = remaining;
    }
    timers.stop("FlipTotal");
}
