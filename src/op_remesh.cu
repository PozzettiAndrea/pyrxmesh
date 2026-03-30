// Isotropic remeshing wrapper.

#include "pipeline.h"
#include <filesystem>
#include <chrono>
#include <cstdio>

#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

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

// Extract mesh directly from RXMeshDynamic — no OBJ round-trip.
static MeshResult extract_mesh_direct(RXMeshDynamic& rx)
{
    rx.update_host();
    auto coords = rx.get_input_vertex_coordinates();
    coords->move(DEVICE, HOST);

    std::vector<glm::vec3> v_list;
    rx.create_vertex_list(v_list, *coords);

    std::vector<glm::uvec3> f_list;
    rx.create_face_list(f_list);

    MeshResult r;
    r.num_vertices = static_cast<int>(v_list.size());
    r.num_faces = static_cast<int>(f_list.size());
    r.vertices.resize(r.num_vertices * 3);
    r.faces.resize(r.num_faces * 3);
    for (int i = 0; i < r.num_vertices; ++i) {
        r.vertices[i*3+0] = v_list[i][0];
        r.vertices[i*3+1] = v_list[i][1];
        r.vertices[i*3+2] = v_list[i][2];
    }
    for (int i = 0; i < r.num_faces; ++i) {
        r.faces[i*3+0] = f_list[i][0];
        r.faces[i*3+1] = f_list[i][1];
        r.faces[i*3+2] = f_list[i][2];
    }
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
    auto fv = flat_faces_to_fv(faces, num_faces);
    auto vv = flat_verts_to_vv(vertices, num_vertices);
    double t_prep = ms_since(tp);

    Arg.obj_file_name = "pyrxmesh_remesh";
    Arg.relative_len = static_cast<float>(relative_len);
    Arg.num_iter = static_cast<uint32_t>(iterations);
    Arg.num_smooth_iters = smooth_iterations;

    tp = clk::now();
    RXMeshDynamic rx(fv, "", 512, 2.0f, 2);
    rx.add_vertex_coordinates(vv);
    double t_build = ms_since(tp);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("Remesh requires an edge-manifold mesh");

    tp = clk::now();
    remesh_rxmesh(rx);
    CUDA_ERROR(cudaDeviceSynchronize());
    double t_gpu = ms_since(tp);

    tp = clk::now();
    auto result = extract_mesh_direct(rx);
    double t_readback = ms_since(tp);
    if (verbose) {
        fprintf(stderr, "[pyrxmesh] remesh: prep=%.1fms, mesh_build=%.1fms, gpu=%.1fms, readback=%.1fms\n",
                t_prep, t_build, t_gpu, t_readback);
        fprintf(stderr, "[pyrxmesh] remesh: output %d verts, %d faces, total %.1f ms\n",
                result.num_vertices, result.num_faces, ms_since(t0));
    }
    return result;
}
