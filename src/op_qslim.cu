// QSlim decimation wrapper — compiled as separate TU with its own Arg struct.

#include "pipeline.h"
#include <filesystem>
#include <chrono>
#include <cstdio>

#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include "glm_compat.h"

using namespace rxmesh;

// Arg struct expected by qslim_rxmesh.cuh
static char* s_qs_argv[] = {(char*)"pyrxmesh", nullptr};
static struct arg {
    std::string obj_file_name;
    std::string output_folder = "/tmp";
    float       target        = 0.1f;
    float       reduce_ratio  = 0.1f;
    uint32_t    device_id     = 0;
    char**      argv          = s_qs_argv;
    int         argc          = 1;
} Arg;

#include "QSlim/qslim_rxmesh.cuh"

static std::string write_temp_obj_qs(
    const double* vertices, int nv, const int* faces, int nf)
{
    auto tmp = std::filesystem::temp_directory_path() / "pyrxmesh_qslim_in.obj";
    FILE* f = fopen(tmp.string().c_str(), "w");
    for (int i = 0; i < nv; ++i)
        fprintf(f, "v %f %f %f\n", vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
    for (int i = 0; i < nf; ++i)
        fprintf(f, "f %d %d %d\n", faces[i*3]+1, faces[i*3+1]+1, faces[i*3+2]+1);
    fclose(f);
    return tmp.string();
}

static MeshResult read_obj_qs(const std::string& path)
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

MeshResult pipeline_qslim(
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
        fprintf(stderr, "[pyrxmesh] qslim: input %d verts, %d faces, target_ratio=%.2f\n",
                num_vertices, num_faces, target_ratio);

    auto tp = clk::now();
    auto in_path = write_temp_obj_qs(vertices, num_vertices, faces, num_faces);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_qslim_out.obj").string();
    double t_write = ms_since(tp);

    Arg.obj_file_name = in_path;
    Arg.target = static_cast<float>(target_ratio);
    Arg.reduce_ratio = 0.1f;

    tp = clk::now();
    RXMeshDynamic rx(in_path, "", 256, 3.5f, 1.5f);
    double t_build = ms_since(tp);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("QSlim requires an edge-manifold mesh");
    if (!rx.is_closed())
        throw std::runtime_error("QSlim requires a closed mesh");

    uint32_t final_num_vertices = static_cast<uint32_t>(
        target_ratio * rx.get_num_vertices());
    if (final_num_vertices < 4) final_num_vertices = 4;

    tp = clk::now();
    qslim_rxmesh(rx, final_num_vertices);
    CUDA_ERROR(cudaDeviceSynchronize());
    double t_gpu = ms_since(tp);

    tp = clk::now();
    auto coords = rx.get_input_vertex_coordinates();
    rx.export_obj(out_path, *coords);
    auto result = read_obj_qs(out_path);
    double t_readback = ms_since(tp);

    std::filesystem::remove(in_path);
    std::filesystem::remove(out_path);
    if (verbose) {
        fprintf(stderr, "[pyrxmesh] qslim: obj_write=%.1fms, mesh_build=%.1fms, gpu=%.1fms, readback=%.1fms\n",
                t_write, t_build, t_gpu, t_readback);
        fprintf(stderr, "[pyrxmesh] qslim: output %d verts, %d faces, total %.1f ms\n",
                result.num_vertices, result.num_faces, ms_since(t0));
    }
    return result;
}
