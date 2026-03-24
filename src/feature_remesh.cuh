// Feature-aware remeshing kernels — copies of RXMesh's split/collapse/flip
// with added EdgeAttribute<int> edge_is_feature to skip feature edges.
//
// Changes from originals marked with "// FEATURE:"

#pragma once
#include <cuda_profiler_api.h>

#include "rxmesh/cavity_manager.cuh"
#include "rxmesh/query.h"
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

template <typename T, uint32_t blockThreads>
__global__ static void feature_edge_split(
    rxmesh::Context                       context,
    const rxmesh::VertexAttribute<T>      coords,
    rxmesh::EdgeAttribute<EdgeStatus>     edge_status,
    rxmesh::VertexAttribute<bool>         v_boundary,
    const rxmesh::EdgeAttribute<int>      edge_is_feature,
    const rxmesh::VertexAttribute<T>      sizing,  // per-vertex sizing multiplier (1.0 = non-adaptive)
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

            // FEATURE: don't split feature edges
            if (edge_is_feature(eh) == 1) {
                edge_status(eh) = SKIP;
                return;
            }

            const VertexHandle va = iter[0];
            const VertexHandle vb = iter[2];
            const VertexHandle vc = iter[1];
            const VertexHandle vd = iter[3];

            if (!vc.is_valid() || !vd.is_valid() || !va.is_valid() || !vb.is_valid()) {
                edge_status(eh) = SKIP;
                return;
            }

            if (v_boundary(va) || v_boundary(vb) || v_boundary(vc) || v_boundary(vd))
                return;

            if (va == vb || vb == vc || vc == va || va == vd || vb == vd || vc == vd) {
                edge_status(eh) = SKIP;
                return;
            }

            const vec3<T> pa = coords.to_glm<3>(va);
            const vec3<T> pb = coords.to_glm<3>(vb);
            const T edge_len = glm::distance2(pa, pb);

            // ADAPTIVE: per-vertex sizing modulates threshold
            T mult = min(sizing(va, 0), sizing(vb, 0));
            T local_high = high_edge_len_sq * mult * mult;
            T local_low = low_edge_len_sq * mult * mult;

            if (edge_len > local_high) {
                vec3<T> p_new = (pa + pb) * T(0.5);
                vec3<T> pc = coords.to_glm<3>(vc);
                vec3<T> pd = coords.to_glm<3>(vd);

                T min_new_edge_len = std::numeric_limits<T>::max();
                min_new_edge_len = std::min(min_new_edge_len, glm::distance2(p_new, pa));
                min_new_edge_len = std::min(min_new_edge_len, glm::distance2(p_new, pb));
                min_new_edge_len = std::min(min_new_edge_len, glm::distance2(p_new, pc));
                min_new_edge_len = std::min(min_new_edge_len, glm::distance2(p_new, pd));

                if (min_new_edge_len >= local_low) {
                    cavity.create(eh);
                } else {
                    edge_status(eh) = SKIP;
                }
            } else {
                edge_status(eh) = SKIP;
            }
        }
    };

    Query<blockThreads> query(context, cavity.patch_id());
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, should_split);
    block.sync();

    shrd_alloc.dealloc(shrd_alloc.get_allocated_size_bytes() - shmem_before);

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
// Feature-aware edge collapse
// =========================================================================

template <typename T, uint32_t blockThreads>
__global__ static void __launch_bounds__(blockThreads)
    feature_edge_collapse(
        rxmesh::Context                       context,
        const rxmesh::VertexAttribute<T>      coords,
        rxmesh::EdgeAttribute<EdgeStatus>     edge_status,
        const rxmesh::EdgeAttribute<int>      edge_is_feature,
        const rxmesh::VertexAttribute<int>    vertex_is_feature,
        const rxmesh::VertexAttribute<T>      sizing,  // per-vertex sizing multiplier
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

    Bitmask edge_mask(cavity.patch_info().edges_capacity, shrd_alloc);
    edge_mask.reset(block);

    uint32_t shmem_before = shrd_alloc.get_allocated_size_bytes();

    // Use vertices_capacity (not num_vertices[0]) to avoid shared memory
    // overflow when vertex local IDs exceed live count after compaction.
    Bitmask v0_mask(cavity.patch_info().vertices_capacity, shrd_alloc);
    Bitmask v1_mask(cavity.patch_info().vertices_capacity, shrd_alloc);

    Query<blockThreads> query(context, pid);
    query.prologue<Op::EVDiamond>(block, shrd_alloc);
    block.sync();

    // 1. mark edges to collapse
    for_each_edge(cavity.patch_info(), [&](EdgeHandle eh) {
        if (edge_status(eh) == UNSEEN) {

            // FEATURE: don't collapse feature edges
            if (edge_is_feature(eh) == 1) {
                edge_status(eh) = SKIP;
                return;
            }

            const VertexIterator iter =
                query.template get_iterator<VertexIterator>(eh.local_id());
            assert(iter.size() == 4);

            const VertexHandle v0 = iter[0], v1 = iter[2];
            const VertexHandle v2 = iter[1], v3 = iter[3];

            // FEATURE: don't collapse edges touching feature vertices
            // Matches CPU behavior: feature vertices should not move
            if (vertex_is_feature(v0) || vertex_is_feature(v1)) {
                edge_status(eh) = SKIP;
                return;
            }

            if (v2.is_valid() && v3.is_valid()) {
                if (v0 == v1 || v0 == v2 || v0 == v3 || v1 == v2 || v1 == v3 || v2 == v3)
                    return;

                const T edge_len_sq = glm::distance2(
                    coords.to_glm<3>(v0), coords.to_glm<3>(v1));

                // ADAPTIVE: per-vertex sizing
                T mult = min(sizing(v0, 0), sizing(v1, 0));
                T local_low = low_edge_len_sq * mult * mult;

                if (edge_len_sq < local_low) {
                    edge_mask.set(eh.local_id(), true);
                }
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
            const EdgeHandle src = cavity.template get_creator<EdgeHandle>(c);
            VertexHandle v0, v1;
            cavity.get_vertices(src, v0, v1);

            const vec3<T> p0 = coords.to_glm<3>(v0);
            const vec3<T> p1 = coords.to_glm<3>(v1);

            // FEATURE FIX 1: collapse toward feature vertex, not midpoint
            // If one endpoint is a feature vertex, keep it pinned
            vec3<T> new_p;
            if (vertex_is_feature(v0) && !vertex_is_feature(v1)) {
                new_p = p0;  // keep feature vertex v0
            } else if (vertex_is_feature(v1) && !vertex_is_feature(v0)) {
                new_p = p1;  // keep feature vertex v1
            } else {
                new_p = (p0 + p1) * T(0.5);  // midpoint for non-feature or both-feature
            }

            // ADAPTIVE: use sizing of collapsed edge endpoints
            T cmult = min(sizing(v0, 0), sizing(v1, 0));
            T cavity_low = low_edge_len_sq * cmult * cmult;

            bool long_edge = false;
            for (uint16_t i = 0; i < size; ++i) {
                const T d = glm::distance2(coords.to_glm<3>(cavity.get_cavity_vertex(c, i)), new_p);
                if (d >= cavity_low) { long_edge = true; break; }
            }

            if (long_edge) {
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
                        edge_mask.set(e0.local_id(), true);
                        const DEdgeHandle e_init = e0;

                        for (uint16_t i = 0; i < size; ++i) {
                            const DEdgeHandle e = cavity.get_cavity_edge(c, i);
                            const DEdgeHandle e1 =
                                (i == size - 1) ?
                                    e_init.get_flip_dedge() :
                                    cavity.add_edge(cavity.get_cavity_vertex(c, (i+1) % size), new_v);
                            if (!e1.is_valid()) break;
                            if (i != size - 1) edge_mask.set(e1.local_id(), true);
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
            if (edge_mask(eh.local_id()) || cavity.is_recovered(eh)) {
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
    CUDA_ERROR(cudaDeviceSynchronize());

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
    CUDA_ERROR(cudaDeviceSynchronize());

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
        CUDA_ERROR(cudaDeviceSynchronize());
    }

    // Step 3: map quality [0,1] → sizing multiplier [minMult, maxMult]
    rx.for_each_vertex(DEVICE,
        [sizing = *sizing, min_mult, max_mult]
        __device__(const VertexHandle vh) mutable {
            float q = sizing(vh, 0);
            q = fminf(fmaxf(q, 0.0f), 1.0f);
            sizing(vh, 0) = min_mult + q * (max_mult - min_mult);
        });
    CUDA_ERROR(cudaDeviceSynchronize());

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
    CUDA_ERROR(cudaDeviceSynchronize());
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
    CUDA_ERROR(cudaDeviceSynchronize());

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
    CUDA_ERROR(cudaDeviceSynchronize());

    rx.for_each_vertex(DEVICE,
        [fv = *feat_valence, vb = *v_boundary, vh_out = *v_high_val]
        __device__(const VertexHandle vh) mutable {
            vh_out(vh) = (fv(vh) > 2 || (vb(vh) && fv(vh) > 1)) ? 1 : 0;
        });
    CUDA_ERROR(cudaDeviceSynchronize());

    // Erode steps
    for (int s = 0; s < steps; s++) {
        feat_valence->reset(0, DEVICE);
        feature_edge_valence_kernel<blockThreads>
            <<<lb_val.blocks, lb_val.num_threads, lb_val.smem_bytes_dyn>>>(
                rx.get_context(), *edge_feature, *feat_valence);
        CUDA_ERROR(cudaDeviceSynchronize());

        feature_erode_kernel<float, blockThreads>
            <<<lb_erode.blocks, lb_erode.num_threads, lb_erode.smem_bytes_dyn>>>(
                rx.get_context(), *coords, *edge_feature, *feat_valence,
                *v_boundary, max_erode_len);
        CUDA_ERROR(cudaDeviceSynchronize());
    }

    // Dilate steps
    for (int s = 0; s < steps; s++) {
        feat_valence->reset(0, DEVICE);
        feature_edge_valence_kernel<blockThreads>
            <<<lb_val.blocks, lb_val.num_threads, lb_val.smem_bytes_dyn>>>(
                rx.get_context(), *edge_feature, *feat_valence);
        CUDA_ERROR(cudaDeviceSynchronize());

        feature_dilate_kernel<blockThreads>
            <<<lb_dilate.blocks, lb_dilate.num_threads, lb_dilate.smem_bytes_dyn>>>(
                rx.get_context(), *edge_feature, *edge_orig,
                *feat_valence, *v_high_val);
        CUDA_ERROR(cudaDeviceSynchronize());
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
    CUDA_ERROR(cudaDeviceSynchronize());
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
    rxmesh::EdgeAttribute<int>*        edge_is_feature,
    rxmesh::VertexAttribute<int>*      vertex_is_feature,
    rxmesh::VertexAttribute<T>*        sizing,
    const T                            high_edge_len_sq,
    const T                            low_edge_len_sq,
    rxmesh::Timers<rxmesh::GPUTimer>&  timers,
    int*                               d_buffer)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    edge_status->reset(UNSEEN, DEVICE);
    pre_skip_feature_edges(rx, edge_status, edge_is_feature);
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
                    *edge_is_feature, *sizing, high_edge_len_sq, low_edge_len_sq, 0);
            timers.stop("Split");

            timers.start("SplitCleanup");
            rx.cleanup();
            timers.stop("SplitCleanup");
            timers.start("SplitSlice");
            {
                uint32_t pre_p = rx.get_num_patches();
                rx.slice_patches(*coords, *v_boundary,
                                 *edge_is_feature, *vertex_is_feature, *sizing);
                uint32_t post_p = rx.get_num_patches();
                if (post_p != pre_p)
                    fprintf(stderr, "        [SLICE] %u → %u patches\n", pre_p, post_p);
            }
            timers.stop("SplitSlice");
            timers.start("SplitCleanup");
            rx.cleanup();
            timers.stop("SplitCleanup");
        }
        int remaining = is_done(rx, edge_status, d_buffer);
        //fprintf(stderr, "    [split] outer=%d inner=%d remaining=%d/%d\n",
        //        split_outer, split_inner, remaining, prv_remaining_work);
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
    rxmesh::EdgeAttribute<int8_t>*     edge_link,
    rxmesh::VertexAttribute<bool>*     v_boundary,
    rxmesh::EdgeAttribute<int>*        edge_is_feature,
    rxmesh::VertexAttribute<int>*      vertex_is_feature,
    rxmesh::VertexAttribute<T>*        sizing,
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
    rx.update_launch_box({Op::EVDiamond}, lb,
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
                    rx.get_context(), *coords, *edge_status,
                    *edge_is_feature, *vertex_is_feature, *sizing,
                    low_edge_len_sq, high_edge_len_sq);
            timers.stop("Collapse");

            timers.start("CollapseCleanup");
            rx.cleanup();
            timers.stop("CollapseCleanup");
            timers.start("CollapseSlice");
            {
                uint32_t pre_p = rx.get_num_patches();
                rx.slice_patches(*coords, *v_boundary,
                                 *edge_is_feature, *vertex_is_feature, *sizing);
                uint32_t post_p = rx.get_num_patches();
                if (post_p != pre_p)
                    fprintf(stderr, "        [SLICE] %u → %u patches\n", pre_p, post_p);
            }
            timers.stop("CollapseSlice");
            timers.start("CollapseCleanup");
            rx.cleanup();
            timers.stop("CollapseCleanup");
        }
        int remaining = is_done(rx, edge_status, d_buffer);
        //fprintf(stderr, "    [collapse] outer=%d inner=%d remaining=%d/%d\n",
        //        col_outer, col_inner, remaining, prv_remaining_work);
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
        CUDA_ERROR(cudaDeviceSynchronize());

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
            {
                uint32_t pre_p = rx.get_num_patches();
                rx.slice_patches(*coords, *v_boundary,
                                 *edge_is_feature, *vertex_is_feature, *sizing);
                uint32_t post_p = rx.get_num_patches();
                if (post_p != pre_p)
                    fprintf(stderr, "        [SLICE] %u → %u patches\n", pre_p, post_p);
            }
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
