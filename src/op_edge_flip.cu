// Standalone edge flip wrapper — flips edges to equalize vertex valences.

#include "pipeline.h"
#include <filesystem>
#include <cstdio>
#include <chrono>

#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include "glm_compat.h"

using namespace rxmesh;

static char* s_ef_argv[] = {(char*)"pyrxmesh", nullptr};
static struct arg {
    std::string obj_file_name;
    std::string output_folder   = "/tmp";
    uint32_t    nx              = 66;
    uint32_t    ny              = 66;
    float       relative_len    = 1.0f;
    int         num_smooth_iters = 0;
    uint32_t    num_iter        = 1;
    uint32_t    device_id       = 0;
    char**      argv            = s_ef_argv;
    int         argc            = 1;
} Arg;

#include "Remesh/flip.cuh"

// Uses write_temp_obj_fast from pipeline.h

static MeshResult read_obj_ef(const std::string& path)
{
    std::vector<std::vector<float>>    verts;
    std::vector<std::vector<uint32_t>> faces;
    import_obj(path, verts, faces);

    MeshResult r;
    r.num_vertices = verts.size();
    r.num_faces = faces.size();
    r.vertices.resize(r.num_vertices * 3);
    r.faces.resize(r.num_faces * 3);
    for (int i = 0; i < r.num_vertices; ++i)
        for (int j = 0; j < 3; ++j)
            r.vertices[i*3+j] = verts[i][j];
    for (int i = 0; i < r.num_faces; ++i)
        for (int j = 0; j < 3; ++j)
            r.faces[i*3+j] = faces[i][j];
    return r;
}

MeshResult pipeline_edge_flip(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int iterations,
    bool verbose)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();
    if (verbose)
        fprintf(stderr, "[pyrxmesh] edge_flip: input %d verts, %d faces\n",
                num_vertices, num_faces);

    auto tp = clk::now();
    auto fv = flat_faces_to_fv(faces, num_faces);
    auto vv = flat_verts_to_vv(vertices, num_vertices);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_eflip_out.obj").string();
    double t_prep = ms_since(tp);

    Arg.obj_file_name = "pyrxmesh_eflip";

    tp = clk::now();
    RXMeshDynamic rx(fv, "", 512, 2.0f, 2);
    rx.add_vertex_coordinates(vv);
    double t_build = ms_since(tp);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("edge_flip requires an edge-manifold mesh");

    auto coords = rx.get_input_vertex_coordinates();
    auto v_valence = rx.add_vertex_attribute<uint8_t>("Valence", 1);
    auto edge_status = rx.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);
    auto edge_link = rx.add_edge_attribute<int8_t>("edgeLink", 1);
    auto v_boundary = rx.add_vertex_attribute<bool>("BoundaryV", 1);

    int* d_buffer;
    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer, sizeof(int)));
    rx.get_boundary_vertices(*v_boundary);

    Timers<GPUTimer> timers;
    timers.add("FlipTotal");
    timers.add("Flip");
    timers.add("FlipCleanup");
    timers.add("FlipSlice");

    tp = clk::now();
    for (int iter = 0; iter < iterations; ++iter) {
        equalize_valences(rx, coords.get(), v_valence.get(), edge_status.get(),
                          edge_link.get(), v_boundary.get(), timers, d_buffer);
    }
    CUDA_ERROR(cudaDeviceSynchronize());
    double t_gpu = ms_since(tp);

    tp = clk::now();
    rx.update_host();
    coords->move(DEVICE, HOST);
    rx.export_obj(out_path, *coords);
    auto result = read_obj_ef(out_path);
    double t_readback = ms_since(tp);

    std::filesystem::remove(out_path);
    CUDA_ERROR(cudaFree(d_buffer));

    if (verbose) {
        fprintf(stderr, "[pyrxmesh] edge_flip: prep=%.1fms, mesh_build=%.1fms, gpu=%.1fms, readback=%.1fms\n",
                t_prep, t_build, t_gpu, t_readback);
        fprintf(stderr, "[pyrxmesh] edge_flip: output %d verts, %d faces, total %.1f ms\n",
                result.num_vertices, result.num_faces, ms_since(t0));
    }
    return result;
}
