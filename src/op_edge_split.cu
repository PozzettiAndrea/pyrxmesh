// Standalone edge split wrapper — splits edges longer than threshold.

#include "pipeline.h"
#include <filesystem>
#include <cstdio>
#include <chrono>

#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include "glm_compat.h"

using namespace rxmesh;

static char* s_es_argv[] = {(char*)"pyrxmesh", nullptr};
static struct arg {
    std::string obj_file_name;
    std::string output_folder   = "/tmp";
    uint32_t    nx              = 66;
    uint32_t    ny              = 66;
    float       relative_len    = 1.0f;
    int         num_smooth_iters = 0;
    uint32_t    num_iter        = 1;
    uint32_t    device_id       = 0;
    char**      argv            = s_es_argv;
    int         argc            = 1;
} Arg;

#include "Remesh/split.cuh"

// Stats struct and compute_stats from remesh_rxmesh.cuh (needed for edge length thresholds)
struct Stats {
    float avg_edge_len, max_edge_len, min_edge_len, avg_vertex_valence;
    int   max_vertex_valence, min_vertex_valence;
};

inline void compute_stats(rxmesh::RXMeshDynamic&                rx,
                          const rxmesh::VertexAttribute<float>* coords,
                          rxmesh::EdgeAttribute<float>*         edge_len,
                          rxmesh::VertexAttribute<int>*         vertex_valence,
                          Stats&                                stats)
{
    using namespace rxmesh;
    constexpr uint32_t blockThreads = 256;

    edge_len->reset(DEVICE, 0);
    vertex_valence->reset(DEVICE, 0);

    stats.avg_edge_len       = 0;
    stats.max_edge_len       = std::numeric_limits<float>::min();
    stats.min_edge_len       = std::numeric_limits<float>::max();
    stats.avg_vertex_valence = 0;
    stats.max_vertex_valence = std::numeric_limits<int>::min();
    stats.min_vertex_valence = std::numeric_limits<int>::max();

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box({Op::EV},
                         launch_box,
                         (void*)stats_kernel<float, blockThreads>,
                         false, false, true);

    stats_kernel<float, blockThreads><<<launch_box.blocks,
                                        launch_box.num_threads,
                                        launch_box.smem_bytes_dyn>>>(
        rx.get_context(), *coords, *edge_len, *vertex_valence);
    CUDA_ERROR(cudaDeviceSynchronize());

    vertex_valence->move(DEVICE, HOST);
    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        int val = (*vertex_valence)(vh);
        stats.avg_vertex_valence += val;
        stats.max_vertex_valence = std::max(stats.max_vertex_valence, val);
        stats.min_vertex_valence = std::min(stats.min_vertex_valence, val);
    }, NULL, false);
    stats.avg_vertex_valence /= rx.get_num_vertices();

    edge_len->move(DEVICE, HOST);
    rx.for_each_edge(HOST, [&](const EdgeHandle eh) {
        float len = (*edge_len)(eh);
        stats.avg_edge_len += len;
        stats.max_edge_len = std::max(stats.max_edge_len, len);
        if (len > std::numeric_limits<float>::epsilon())
            stats.min_edge_len = std::min(stats.min_edge_len, len);
    }, NULL, false);
    stats.avg_edge_len /= rx.get_num_edges();
}

static std::string write_temp_obj_es(
    const double* vertices, int nv, const int* faces, int nf)
{
    auto tmp = std::filesystem::temp_directory_path() / "pyrxmesh_esplit_in.obj";
    FILE* f = fopen(tmp.string().c_str(), "w");
    for (int i = 0; i < nv; ++i)
        fprintf(f, "v %f %f %f\n", vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
    for (int i = 0; i < nf; ++i)
        fprintf(f, "f %d %d %d\n", faces[i*3]+1, faces[i*3+1]+1, faces[i*3+2]+1);
    fclose(f);
    return tmp.string();
}

static MeshResult read_obj_es(const std::string& path)
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

MeshResult pipeline_edge_split(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double relative_len,
    int iterations,
    bool verbose)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();
    if (verbose)
        fprintf(stderr, "[pyrxmesh] edge_split: input %d verts, %d faces, relative_len=%.2f\n",
                num_vertices, num_faces, relative_len);

    auto tp = clk::now();
    auto in_path = write_temp_obj_es(vertices, num_vertices, faces, num_faces);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_esplit_out.obj").string();
    double t_write = ms_since(tp);

    Arg.obj_file_name = in_path;
    Arg.relative_len = static_cast<float>(relative_len);

    tp = clk::now();
    RXMeshDynamic rx(in_path, "", 512, 2.0f, 2);
    double t_build = ms_since(tp);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("edge_split requires an edge-manifold mesh");

    auto coords = rx.get_input_vertex_coordinates();
    auto edge_status = rx.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);
    auto v_boundary = rx.add_vertex_attribute<bool>("BoundaryV", 1);
    auto edge_len = rx.add_edge_attribute<float>("edgeLen", 1);
    auto vertex_valence = rx.add_vertex_attribute<int>("vertexValence", 1);

    int* d_buffer;
    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer, sizeof(int)));
    rx.get_boundary_vertices(*v_boundary);

    Stats stats;
    compute_stats(rx, coords.get(), edge_len.get(), vertex_valence.get(), stats);

    const float low_edge_len = (4.f / 5.f) * Arg.relative_len * stats.avg_edge_len;
    const float low_edge_len_sq = low_edge_len * low_edge_len;
    const float high_edge_len = (4.f / 3.f) * Arg.relative_len * stats.avg_edge_len;
    const float high_edge_len_sq = high_edge_len * high_edge_len;

    Timers<GPUTimer> timers;
    timers.add("SplitTotal");
    timers.add("Split");
    timers.add("SplitCleanup");
    timers.add("SplitSlice");

    tp = clk::now();
    for (int iter = 0; iter < iterations; ++iter) {
        split_long_edges(rx, coords.get(), edge_status.get(), v_boundary.get(),
                         high_edge_len_sq, low_edge_len_sq, timers, d_buffer);
    }
    CUDA_ERROR(cudaDeviceSynchronize());
    double t_gpu = ms_since(tp);

    tp = clk::now();
    rx.update_host();
    coords->move(DEVICE, HOST);
    rx.export_obj(out_path, *coords);
    auto result = read_obj_es(out_path);
    double t_readback = ms_since(tp);

    std::filesystem::remove(in_path);
    std::filesystem::remove(out_path);
    CUDA_ERROR(cudaFree(d_buffer));

    if (verbose) {
        fprintf(stderr, "[pyrxmesh] edge_split: avg_edge=%.4f, threshold=%.4f\n",
                stats.avg_edge_len, high_edge_len);
        fprintf(stderr, "[pyrxmesh] edge_split: obj_write=%.1fms, mesh_build=%.1fms, gpu=%.1fms, readback=%.1fms\n",
                t_write, t_build, t_gpu, t_readback);
        fprintf(stderr, "[pyrxmesh] edge_split: output %d verts, %d faces, total %.1f ms\n",
                result.num_vertices, result.num_faces, ms_since(t0));
    }
    return result;
}
