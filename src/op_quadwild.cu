// QuadWild preprocessing: GPU remesh with auto target edge length.
// Float input (QuadWild native format), auto edge length from mesh area.

#include "pipeline.h"
#include <filesystem>
#include <chrono>
#include <cstdio>
#include <cmath>

#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include "glm_compat.h"

using namespace rxmesh;

static char* s_qw_argv[] = {(char*)"pyrxmesh", nullptr};
static struct arg {
    std::string obj_file_name;
    std::string output_folder   = "/tmp";
    uint32_t    nx              = 66;
    uint32_t    ny              = 66;
    float       relative_len    = 1.0f;
    int         num_smooth_iters = 5;
    uint32_t    num_iter        = 3;
    uint32_t    device_id       = 0;
    char**      argv            = s_qw_argv;
    int         argc            = 1;
} Arg;

#include "Remesh/remesh_rxmesh.cuh"

static std::string write_temp_obj_qw(
    const float* vertices, int nv, const int* faces, int nf)
{
    auto tmp = std::filesystem::temp_directory_path() / "pyrxmesh_qw_in.obj";
    FILE* f = fopen(tmp.string().c_str(), "w");
    for (int i = 0; i < nv; ++i)
        fprintf(f, "v %.15g %.15g %.15g\n",
                vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
    for (int i = 0; i < nf; ++i)
        fprintf(f, "f %d %d %d\n",
                faces[i*3]+1, faces[i*3+1]+1, faces[i*3+2]+1);
    fclose(f);
    return tmp.string();
}

static MeshResult read_obj_qw(const std::string& path)
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

MeshResult pipeline_quadwild_preprocess(
    const float* vertices, int num_vertices,
    const int* faces, int num_faces,
    const QuadwildParams& params,
    bool verbose)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();

    // Compute mesh stats: area, volume, avg edge length
    double total_edge_len = 0;
    double total_area = 0;
    double total_volume = 0;
    int n_edges = 0;
    for (int f = 0; f < num_faces; f++) {
        int i0 = faces[f*3], i1 = faces[f*3+1], i2 = faces[f*3+2];
        float v0x = vertices[i0*3], v0y = vertices[i0*3+1], v0z = vertices[i0*3+2];
        float v1x = vertices[i1*3], v1y = vertices[i1*3+1], v1z = vertices[i1*3+2];
        float v2x = vertices[i2*3], v2y = vertices[i2*3+1], v2z = vertices[i2*3+2];
        float ax = v1x-v0x, ay = v1y-v0y, az = v1z-v0z;
        float bx = v2x-v0x, by = v2y-v0y, bz = v2z-v0z;
        float cx = ay*bz - az*by, cy = az*bx - ax*bz, cz = ax*by - ay*bx;
        total_area += 0.5 * sqrt(cx*cx + cy*cy + cz*cz);
        // Signed volume contribution: v0 . (v1 x v2) / 6
        total_volume += (v0x*(v1y*v2z - v1z*v2y) +
                         v0y*(v1z*v2x - v1x*v2z) +
                         v0z*(v1x*v2y - v1y*v2x)) / 6.0;
        total_edge_len += sqrt(ax*ax + ay*ay + az*az);
        float ex = v2x-v1x, ey = v2y-v1y, ez = v2z-v1z;
        total_edge_len += sqrt(ex*ex + ey*ey + ez*ez);
        total_edge_len += sqrt(bx*bx + by*by + bz*bz);
        n_edges += 3;
    }
    float avg_edge = static_cast<float>(total_edge_len / n_edges);
    double vol = fabs(total_volume);

    // ExpectedEdgeL — QuadWild's sphericity-based target edge length
    int target_faces_count = params.target_faces > 0 ? params.target_faces : 10000;
    float target_edge = params.target_edge_length;
    if (target_edge <= 0) {
        double A = total_area;
        double Sphericity = (pow(M_PI, 1.0/3.0) * pow(6.0 * vol, 2.0/3.0)) / A;
        double KScale = pow(Sphericity, 2);
        double FaceA = A / 2000.0;  // TargetSph = 2000
        double IdealA = FaceA * KScale;
        double IdealL0 = sqrt(IdealA * 2.309);
        double IdealL1 = sqrt((A * 2.309) / target_faces_count);
        target_edge = static_cast<float>(std::min(IdealL0, IdealL1));
    }
    float relative_len = target_edge / avg_edge;

    double Sphericity = (pow(M_PI, 1.0/3.0) * pow(6.0 * vol, 2.0/3.0)) / total_area;
    if (verbose)
        fprintf(stderr, "[pyrxmesh] quadwild_preprocess: input %d verts, %d faces, "
                "area=%.4f, vol=%.4f, sphericity=%.4f, target_edge=%.6f, relative_len=%.3f\n",
                num_vertices, num_faces, total_area, vol, Sphericity, target_edge, relative_len);

    auto tp = clk::now();
    auto in_path = write_temp_obj_qw(vertices, num_vertices, faces, num_faces);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_qw_out.obj").string();
    double t_write = ms_since(tp);

    Arg.obj_file_name = in_path;
    Arg.relative_len = relative_len;
    Arg.num_iter = static_cast<uint32_t>(params.num_iterations > 0 ? params.num_iterations : 3);
    Arg.num_smooth_iters = params.num_smooth_iters > 0 ? params.num_smooth_iters : 5;

    tp = clk::now();
    RXMeshDynamic rx(in_path, "", 512, 3.5f, 5);
    double t_build = ms_since(tp);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("quadwild_preprocess: mesh is not edge-manifold");

    tp = clk::now();
    remesh_rxmesh(rx);
    CUDA_ERROR(cudaDeviceSynchronize());
    double t_gpu = ms_since(tp);

    tp = clk::now();
    auto coords = rx.get_input_vertex_coordinates();
    rx.export_obj(out_path, *coords);
    auto result = read_obj_qw(out_path);
    double t_readback = ms_since(tp);

    std::filesystem::remove(in_path);
    std::filesystem::remove(out_path);

    if (verbose) {
        fprintf(stderr, "[pyrxmesh] quadwild_preprocess: obj_write=%.1fms, mesh_build=%.1fms, "
                "gpu=%.1fms, readback=%.1fms\n",
                t_write, t_build, t_gpu, t_readback);
        fprintf(stderr, "[pyrxmesh] quadwild_preprocess: output %d verts, %d faces, total %.1f ms\n",
                result.num_vertices, result.num_faces, ms_since(t0));
    }

    return result;
}
