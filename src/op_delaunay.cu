// Delaunay edge flip wrapper.

#include "pipeline.h"
#include <filesystem>
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
    const int* faces, int num_faces)
{
    auto in_path = write_temp_obj_del(vertices, num_vertices, faces, num_faces);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_del_out.obj").string();

    Arg.obj_file_name = in_path;

    RXMeshDynamic rx(in_path, "", 512, 2.0f, 2.0f);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("Delaunay requires an edge-manifold mesh");

    delaunay_rxmesh(rx, false, true);

    auto coords = rx.get_input_vertex_coordinates();
    rx.export_obj(out_path, *coords);

    auto result = read_obj_del(out_path);
    std::filesystem::remove(in_path);
    std::filesystem::remove(out_path);
    return result;
}
