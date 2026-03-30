// SEC (Shortest Edge Collapse) wrapper.

#include "pipeline.h"
#include <filesystem>
#include <chrono>
#include <cstdio>

#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include "glm_compat.h"

using namespace rxmesh;

static char* s_sec_argv[] = {(char*)"pyrxmesh", nullptr};
static struct arg {
    std::string obj_file_name;
    std::string output_folder = "/tmp";
    float       target        = 0.1f;
    float       reduce_ratio  = 0.1f;
    uint32_t    device_id     = 0;
    char**      argv          = s_sec_argv;
    int         argc          = 1;
} Arg;

#include "SECHistogram/sec_rxmesh.cuh"

// Uses write_temp_obj_fast from pipeline.h

static MeshResult read_obj_sec(const std::string& path)
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

MeshResult pipeline_sec(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double target_ratio,
    bool verbose)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();
    if (verbose)
        fprintf(stderr, "[pyrxmesh] sec: input %d verts, %d faces, target_ratio=%.2f\n",
                num_vertices, num_faces, target_ratio);

    auto tp = clk::now();
    auto fv = flat_faces_to_fv(faces, num_faces);
    auto vv = flat_verts_to_vv(vertices, num_vertices);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_sec_out.obj").string();
    double t_prep = ms_since(tp);

    Arg.obj_file_name = "pyrxmesh_sec";
    Arg.target = static_cast<float>(target_ratio);
    Arg.reduce_ratio = 0.1f;

    tp = clk::now();
    RXMeshDynamic rx(fv, "", 256, 3.5f, 1.5f);
    rx.add_vertex_coordinates(vv);
    double t_build = ms_since(tp);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("SEC requires an edge-manifold mesh");
    if (!rx.is_closed())
        throw std::runtime_error("SEC requires a closed mesh");

    uint32_t final_num_vertices = static_cast<uint32_t>(
        target_ratio * rx.get_num_vertices());
    if (final_num_vertices < 4) final_num_vertices = 4;

    tp = clk::now();
    sec_rxmesh(rx, final_num_vertices);
    CUDA_ERROR(cudaDeviceSynchronize());
    double t_gpu = ms_since(tp);

    tp = clk::now();
    auto coords = rx.get_input_vertex_coordinates();
    rx.export_obj(out_path, *coords);
    auto result = read_obj_sec(out_path);
    double t_readback = ms_since(tp);

    std::filesystem::remove(out_path);
    if (verbose) {
        fprintf(stderr, "[pyrxmesh] sec: prep=%.1fms, mesh_build=%.1fms, gpu=%.1fms, readback=%.1fms\n",
                t_prep, t_build, t_gpu, t_readback);
        fprintf(stderr, "[pyrxmesh] sec: output %d verts, %d faces, total %.1f ms\n",
                result.num_vertices, result.num_faces, ms_since(t0));
    }
    return result;
}
