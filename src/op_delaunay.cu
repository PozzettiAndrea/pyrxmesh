// Delaunay edge flip wrapper.

#include "pipeline.h"
#include <filesystem>
#include <chrono>
#include <cstdio>

#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include "glm_compat.h"

using namespace rxmesh;

static char* s_del_argv[] = {(char*)"pyrxmesh", nullptr};
static struct arg {
    std::string obj_file_name;
    std::string output_folder = "/tmp";
    bool        verify        = false;
    bool        skip_mcf      = true;
    uint32_t    device_id     = 0;
    char**      argv          = s_del_argv;
    int         argc          = 1;
} Arg;

#include "Delaunay/delaunay_rxmesh.cuh"

static std::string write_temp_obj_del(
    const double* vertices, int nv, const int* faces, int nf)
{
    auto tmp = std::filesystem::temp_directory_path() / "pyrxmesh_del_in.obj";
    FILE* f = fopen(tmp.string().c_str(), "w");
    for (int i = 0; i < nv; ++i)
        fprintf(f, "v %f %f %f\n", vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
    for (int i = 0; i < nf; ++i)
        fprintf(f, "f %d %d %d\n", faces[i*3]+1, faces[i*3+1]+1, faces[i*3+2]+1);
    fclose(f);
    return tmp.string();
}

static MeshResult read_obj_del(const std::string& path)
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

MeshResult pipeline_delaunay(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    bool verbose)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();
    if (verbose)
        fprintf(stderr, "[pyrxmesh] delaunay: input %d verts, %d faces\n", num_vertices, num_faces);

    auto tp = clk::now();
    auto in_path = write_temp_obj_del(vertices, num_vertices, faces, num_faces);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_del_out.obj").string();
    double t_write = ms_since(tp);

    Arg.obj_file_name = in_path;

    tp = clk::now();
    RXMeshDynamic rx(in_path, "", 512, 2.0f, 2);
    double t_build = ms_since(tp);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("Delaunay requires an edge-manifold mesh");

    tp = clk::now();
    delaunay_rxmesh(rx, false, true);
    CUDA_ERROR(cudaDeviceSynchronize());
    double t_gpu = ms_since(tp);

    tp = clk::now();
    auto coords = rx.get_input_vertex_coordinates();
    rx.export_obj(out_path, *coords);
    auto result = read_obj_del(out_path);
    double t_readback = ms_since(tp);

    std::filesystem::remove(in_path);
    std::filesystem::remove(out_path);
    if (verbose) {
        fprintf(stderr, "[pyrxmesh] delaunay: obj_write=%.1fms, mesh_build=%.1fms, gpu=%.1fms, readback=%.1fms\n",
                t_write, t_build, t_gpu, t_readback);
        fprintf(stderr, "[pyrxmesh] delaunay: output %d verts, %d faces, total %.1f ms\n",
                result.num_vertices, result.num_faces, ms_since(t0));
    }
    return result;
}
