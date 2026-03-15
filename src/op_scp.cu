// Spectral Conformal Parameterization wrapper.
// Self-contained — no global Arg struct needed.

#include "pipeline.h"
#include <filesystem>
#include <cstdio>
#include <limits>

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/attribute.h"
#include "rxmesh/query.h"
#include "rxmesh/reduce_handle.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/cholesky_solver.h"
#include "rxmesh/util/import_obj.h"

#include "glm_compat.h"

using namespace rxmesh;

// ---------------------------------------------------------------------------
// SCP kernels (from apps/SCP/scp.cu)
// ---------------------------------------------------------------------------

template <uint32_t blockThreads>
__global__ static void scp_area_term(
    const Context              context,
    const VertexAttribute<int> v_bd,
    SparseMatrix<cuComplex>    E)
{
    auto compute = [&](FaceHandle& face_id, const VertexIterator& iter) {
        assert(iter.size() == 3);
        for (int i = 0; i < 3; ++i) {
            int j = (i + 1) % 3;
            if (v_bd(iter[i]) == 1 && v_bd(iter[j]) == 1) {
                ::atomicAdd(&E(iter[i], iter[j]).y, 0.25f);
                ::atomicAdd(&E(iter[j], iter[i]).y, -0.25f);
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, compute);
}

template <uint32_t blockThreads>
__global__ static void scp_conformal_energy(
    const Context                context,
    const VertexAttribute<float> coord,
    SparseMatrix<cuComplex>      E)
{
    auto compute = [&](EdgeHandle& p0, const VertexIterator& iter) {
        auto weight = [&](const vec3<float>& P, const vec3<float>& Q,
                          const vec3<float>& O) {
            const vec3<float> l1 = O - Q;
            const vec3<float> l2 = O - P;
            float w = glm::dot(l1, l2) / glm::length(glm::cross(l1, l2));
            return fmaxf(0.f, w);
        };

        VertexHandle p = iter[0];
        VertexHandle q = iter[2];
        VertexHandle o0 = iter[1];
        VertexHandle o1 = iter[3];

        assert(p.is_valid() && q.is_valid());
        assert(o0.is_valid() || o1.is_valid());

        const vec3<float> P = coord.to_glm<3>(p);
        const vec3<float> Q = coord.to_glm<3>(q);

        float coef = 0;
        if (o0.is_valid()) {
            const vec3<float> O0 = coord.to_glm<3>(o0);
            coef += weight(P, Q, O0);
        }
        if (o1.is_valid()) {
            const vec3<float> O1 = coord.to_glm<3>(o1);
            coef += weight(P, Q, O1);
        }
        coef *= 0.25f;

        E(p, q).x = -coef;
        E(q, p).x = -coef;
        ::atomicAdd(&E(p, p).x, coef);
        ::atomicAdd(&E(q, q).x, coef);
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator shrd_alloc;
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, compute);
}

// ---------------------------------------------------------------------------
// pipeline_scp
// ---------------------------------------------------------------------------

AttributeResult pipeline_scp(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int iterations)
{
    // Write to temp OBJ for file-based construction
    auto tmp = std::filesystem::temp_directory_path() / "pyrxmesh_scp_in.obj";
    FILE* fp = fopen(tmp.string().c_str(), "w");
    for (int i = 0; i < num_vertices; ++i)
        fprintf(fp, "v %f %f %f\n", vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
    for (int i = 0; i < num_faces; ++i)
        fprintf(fp, "f %d %d %d\n", faces[i*3]+1, faces[i*3+1]+1, faces[i*3+2]+1);
    fclose(fp);

    RXMeshStatic rx(tmp.string());
    std::filesystem::remove(tmp);

    if (rx.is_closed())
        throw std::runtime_error("SCP requires a mesh with boundaries (not closed).");

    constexpr uint32_t CUDABlockSize = 256;

    auto v_bd = *rx.add_vertex_attribute<int>("vBoundary", 1);
    rx.get_boundary_vertices(v_bd);

    auto coords = *rx.get_input_vertex_coordinates();
    auto uv = *rx.add_vertex_attribute<float>("uv", 3);

    DenseMatrix<cuComplex> uv_mat(rx, rx.get_num_vertices(), 1, LOCATION_ALL);

    // Count boundary vertices
    ReduceHandle rh(v_bd);
    int num_bd_vertices = rh.reduce(v_bd, cub::Sum(), 0);

    // Compute conformal energy matrix Lc
    SparseMatrix<cuComplex> Lc(rx);
    Lc.reset(make_cuComplex(0.f, 0.f), LOCATION_ALL);

    LaunchBox<CUDABlockSize> lb;
    rx.prepare_launch_box({Op::EVDiamond}, lb, (void*)scp_conformal_energy<CUDABlockSize>);
    scp_conformal_energy<CUDABlockSize>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), coords, Lc);

    // Area term
    rx.prepare_launch_box({Op::FV}, lb, (void*)scp_area_term<CUDABlockSize>);
    scp_area_term<CUDABlockSize>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            rx.get_context(), v_bd, Lc);

    // Compute B and eb
    DenseMatrix<cuComplex> eb(rx, rx.get_num_vertices(), 1, LOCATION_ALL);
    eb.reset(make_cuComplex(0.f, 0.f), LOCATION_ALL);

    SparseMatrix<cuComplex> B(rx);
    B.reset(make_cuComplex(0.f, 0.f), LOCATION_ALL);

    float nb = 1.f / std::sqrt(float(num_bd_vertices));
    rx.for_each_vertex(DEVICE,
        [B, eb, v_bd, nb] __device__(const VertexHandle vh) mutable {
            eb(vh, 0) = make_cuComplex((float)v_bd(vh, 0) * nb, 0.0f);
            B(vh, vh) = make_cuComplex((float)v_bd(vh, 0), 0.0f);
        });

    DenseMatrix<cuComplex> T1(rx, rx.get_num_vertices(), 1, LOCATION_ALL);
    T1.reset(make_cuComplex(0.f, 0.f), LOCATION_ALL);

    // Random init
    uv_mat.fill_random();

    // Factorize
    CholeskySolver solver(&Lc);
    solver.pre_solve(rx);

    float prv_norm = std::numeric_limits<float>::max();

    // Power method iterations
    for (int i = 0; i < iterations; i++) {
        cuComplex T2 = eb.dot(uv_mat);
        rx.for_each_vertex(DEVICE,
            [eb, T2, T1, uv_mat, B] __device__(const VertexHandle vh) mutable {
                T1(vh, 0) = cuCsubf(cuCmulf(B(vh, vh), uv_mat(vh, 0)),
                                    cuCmulf(eb(vh, 0), T2));
            });

        solver.solve(T1, uv_mat);

        float norm = uv_mat.norm2();
        uv_mat.multiply(1.0f / norm);

        if (std::abs(prv_norm - norm) < 0.0001f) break;
        prv_norm = norm;
    }

    // Convert to UV attribute
    rx.for_each_vertex(DEVICE,
        [uv_mat, uv] __device__(const VertexHandle vh) mutable {
            uv(vh, 0) = uv_mat(vh, 0).x;
            uv(vh, 1) = uv_mat(vh, 0).y;
        });

    uv.move(DEVICE, HOST);

    // Normalize UV coordinates
    ReduceHandle rrh(uv);
    float lower0 = rrh.reduce(uv, cub::Min(), std::numeric_limits<float>::max(), 0);
    float lower1 = rrh.reduce(uv, cub::Min(), std::numeric_limits<float>::max(), 1);
    float upper0 = rrh.reduce(uv, cub::Max(), std::numeric_limits<float>::min(), 0);
    float upper1 = rrh.reduce(uv, cub::Max(), std::numeric_limits<float>::min(), 1);

    float range0 = upper0 - lower0;
    float range1 = upper1 - lower1;
    float s = std::max(range0, range1);

    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        uv(vh, 0) = (uv(vh, 0) - lower0) / s;
        uv(vh, 1) = (uv(vh, 1) - lower1) / s;
    });

    // Return UV as (N, 2) result
    AttributeResult result;
    result.num_elements = static_cast<int>(rx.get_num_vertices());
    result.num_cols = 2;
    result.data.resize(result.num_elements * 2);

    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rx.map_to_global(vh);
        result.data[v_id * 2 + 0] = static_cast<double>(uv(vh, 0));
        result.data[v_id * 2 + 1] = static_cast<double>(uv(vh, 1));
    });

    uv_mat.release();
    Lc.release();
    eb.release();
    B.release();
    T1.release();

    return result;
}
