// Isotropic remeshing wrapper.

#include "pipeline.h"
#include <filesystem>
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

static std::string write_temp_obj_rm(
    const double* vertices, int nv, const int* faces, int nf)
{
    auto tmp = std::filesystem::temp_directory_path() / "pyrxmesh_remesh_in.obj";
    FILE* f = fopen(tmp.string().c_str(), "w");
    for (int i = 0; i < nv; ++i)
        fprintf(f, "v %f %f %f\n", vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
    for (int i = 0; i < nf; ++i)
        fprintf(f, "f %d %d %d\n", faces[i*3]+1, faces[i*3+1]+1, faces[i*3+2]+1);
    fclose(f);
    return tmp.string();
}

static MeshResult read_obj_rm(const std::string& path)
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

MeshResult pipeline_remesh(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double relative_len,
    int iterations,
    int smooth_iterations)
{
    auto in_path = write_temp_obj_rm(vertices, num_vertices, faces, num_faces);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_remesh_out.obj").string();

    Arg.obj_file_name = in_path;
    Arg.relative_len = static_cast<float>(relative_len);
    Arg.num_iter = static_cast<uint32_t>(iterations);
    Arg.num_smooth_iters = smooth_iterations;

    RXMeshDynamic rx(in_path, "", 512, 2.0f, 2);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("Remesh requires an edge-manifold mesh");

    remesh_rxmesh(rx);

    auto coords = rx.get_input_vertex_coordinates();
    rx.export_obj(out_path, *coords);

    auto result = read_obj_rm(out_path);
    std::filesystem::remove(in_path);
    std::filesystem::remove(out_path);
    return result;
}
