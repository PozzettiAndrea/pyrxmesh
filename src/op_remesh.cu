// Isotropic remeshing wrapper.

#include "pipeline.h"
#include <filesystem>
#include <chrono>
#include <cstdio>

#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"
#include "rxmesh/query.h"
#include "rxmesh/launch_box.h"

#include "glm_compat.h"

using namespace rxmesh;

static char* s_rm_argv[] = {(char*)"pyrxmesh", nullptr};
static struct arg {
    std::string obj_file_name;
    std::string output_folder   = "/tmp";
    uint32_t    nx              = 66;
    uint32_t    ny              = 66;
    float       relative_len    = 1.0f;
    int         num_smooth_iters = 5;
    uint32_t    num_iter        = 3;
    uint32_t    device_id       = 0;
    char**      argv            = s_rm_argv;
    int         argc            = 1;
} Arg;

#include "Remesh/remesh_rxmesh.cuh"

// Uses write_temp_obj_fast from pipeline.h

// GPU readback: extract verts + faces entirely on GPU, single cudaMemcpy back.

template <uint32_t blockThreads>
__global__ static void extract_vertices_kernel(
    const Context                context,
    const VertexAttribute<float> coords,
    float*                       d_verts,
    uint32_t*                    d_vert_count)
{
    auto extract = [&](const VertexHandle vh, const VertexIterator& iter) {
        // iter gives VV neighbors but we just need the vertex itself
        uint32_t vid = context.linear_id(vh);
        d_verts[vid * 3 + 0] = coords(vh, 0);
        d_verts[vid * 3 + 1] = coords(vh, 1);
        d_verts[vid * 3 + 2] = coords(vh, 2);
        atomicAdd(d_vert_count, 1u);
    };
    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, extract);
}

template <uint32_t blockThreads>
__global__ static void extract_faces_kernel(
    const Context context,
    int*          d_faces,
    uint32_t*     d_face_count)
{
    auto extract = [&](const FaceHandle fh, const VertexIterator& iter) {
        uint32_t fid = atomicAdd(d_face_count, 1u);
        d_faces[fid * 3 + 0] = static_cast<int>(context.linear_id(iter[0]));
        d_faces[fid * 3 + 1] = static_cast<int>(context.linear_id(iter[1]));
        d_faces[fid * 3 + 2] = static_cast<int>(context.linear_id(iter[2]));
    };
    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, extract);
}

static MeshResult extract_mesh_gpu(RXMeshDynamic& rx)
{
    constexpr uint32_t blockThreads = 256;

    rx.update_host();
    auto coords = rx.get_input_vertex_coordinates();

    uint32_t num_verts = rx.get_num_vertices();
    uint32_t num_faces = rx.get_num_faces();

    // Allocate device output arrays
    float* d_verts;
    int*   d_faces;
    uint32_t* d_vert_count;
    uint32_t* d_face_count;
    CUDA_ERROR(cudaMalloc(&d_verts, num_verts * 3 * sizeof(float)));
    CUDA_ERROR(cudaMalloc(&d_faces, num_faces * 3 * sizeof(int)));
    CUDA_ERROR(cudaMalloc(&d_vert_count, sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_face_count, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_vert_count, 0, sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_face_count, 0, sizeof(uint32_t)));

    // Launch vertex extraction
    LaunchBox<blockThreads> lb_v;
    rx.update_launch_box({Op::VV}, lb_v,
                         (void*)extract_vertices_kernel<blockThreads>,
                         false, false);
    extract_vertices_kernel<blockThreads>
        <<<lb_v.blocks, lb_v.num_threads, lb_v.smem_bytes_dyn>>>(
            rx.get_context(), *coords, d_verts, d_vert_count);

    // Launch face extraction
    LaunchBox<blockThreads> lb_f;
    rx.update_launch_box({Op::FV}, lb_f,
                         (void*)extract_faces_kernel<blockThreads>,
                         false, false);
    extract_faces_kernel<blockThreads>
        <<<lb_f.blocks, lb_f.num_threads, lb_f.smem_bytes_dyn>>>(
            rx.get_context(), d_faces, d_face_count);

    CUDA_ERROR(cudaDeviceSynchronize());

    // Get actual counts
    uint32_t h_nv, h_nf;
    CUDA_ERROR(cudaMemcpy(&h_nv, d_vert_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(&h_nf, d_face_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Copy results back
    MeshResult r;
    r.num_vertices = static_cast<int>(h_nv);
    r.num_faces = static_cast<int>(h_nf);

    std::vector<float> h_verts(h_nv * 3);
    std::vector<int>   h_faces(h_nf * 3);
    CUDA_ERROR(cudaMemcpy(h_verts.data(), d_verts, h_nv * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(h_faces.data(), d_faces, h_nf * 3 * sizeof(int), cudaMemcpyDeviceToHost));

    r.vertices.resize(h_nv * 3);
    r.faces.resize(h_nf * 3);
    for (uint32_t i = 0; i < h_nv; ++i) {
        r.vertices[i*3+0] = h_verts[i*3+0];
        r.vertices[i*3+1] = h_verts[i*3+1];
        r.vertices[i*3+2] = h_verts[i*3+2];
    }
    for (uint32_t i = 0; i < h_nf; ++i) {
        r.faces[i*3+0] = h_faces[i*3+0];
        r.faces[i*3+1] = h_faces[i*3+1];
        r.faces[i*3+2] = h_faces[i*3+2];
    }

    CUDA_ERROR(cudaFree(d_verts));
    CUDA_ERROR(cudaFree(d_faces));
    CUDA_ERROR(cudaFree(d_vert_count));
    CUDA_ERROR(cudaFree(d_face_count));

    return r;
}

MeshResult pipeline_remesh(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double relative_len,
    int iterations,
    int smooth_iterations,
    bool verbose)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();
    if (verbose)
        fprintf(stderr, "[pyrxmesh] remesh: input %d verts, %d faces, relative_len=%.2f\n",
                num_vertices, num_faces, relative_len);

    auto tp = clk::now();
    // Cast int* → uint32_t* and double* → float* (flat, no heap allocs)
    std::vector<uint32_t> flat_faces_u32(num_faces * 3);
    for (int i = 0; i < num_faces * 3; ++i)
        flat_faces_u32[i] = static_cast<uint32_t>(faces[i]);
    std::vector<float> flat_verts_f32(num_vertices * 3);
    for (int i = 0; i < num_vertices * 3; ++i)
        flat_verts_f32[i] = static_cast<float>(vertices[i]);
    double t_prep = ms_since(tp);

    Arg.obj_file_name = "pyrxmesh_remesh";
    Arg.relative_len = static_cast<float>(relative_len);
    Arg.num_iter = static_cast<uint32_t>(iterations);
    Arg.num_smooth_iters = smooth_iterations;

    tp = clk::now();
    RXMeshDynamic rx(flat_faces_u32.data(), num_faces, "", 512, 2.0f, 2);
    rx.add_vertex_coordinates_flat(flat_verts_f32.data(), num_vertices);
    double t_build = ms_since(tp);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("Remesh requires an edge-manifold mesh");

    tp = clk::now();
    remesh_rxmesh(rx);
    CUDA_ERROR(cudaDeviceSynchronize());
    double t_gpu = ms_since(tp);

    tp = clk::now();
    auto result = extract_mesh_gpu(rx);
    double t_readback = ms_since(tp);
    if (verbose) {
        fprintf(stderr, "[pyrxmesh] remesh: prep=%.1fms, mesh_build=%.1fms, gpu=%.1fms, readback=%.1fms\n",
                t_prep, t_build, t_gpu, t_readback);
        fprintf(stderr, "[pyrxmesh] remesh: output %d verts, %d faces, total %.1f ms\n",
                result.num_vertices, result.num_faces, ms_since(t0));
    }
    return result;
}
