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

// Uses write_temp_obj_fast from pipeline.h

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
    auto fv = flat_faces_to_fv(faces, num_faces);
    auto vv = flat_verts_to_vv(vertices, num_vertices);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_del_out.obj").string();
    double t_prep = ms_since(tp);

    Arg.obj_file_name = "pyrxmesh_del";

    tp = clk::now();
    RXMeshDynamic rx(fv, "", 512, 2.0f, 2);
    rx.add_vertex_coordinates(vv);
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

    std::filesystem::remove(out_path);
    if (verbose) {
        fprintf(stderr, "[pyrxmesh] delaunay: prep=%.1fms, mesh_build=%.1fms, gpu=%.1fms, readback=%.1fms\n",
                t_prep, t_build, t_gpu, t_readback);
        fprintf(stderr, "[pyrxmesh] delaunay: output %d verts, %d faces, total %.1f ms\n",
                result.num_vertices, result.num_faces, ms_since(t0));
    }
    return result;
}
