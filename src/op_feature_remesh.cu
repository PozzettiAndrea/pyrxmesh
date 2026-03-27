// Feature-aware GPU remeshing — uses feature_remesh.cuh kernels that
// skip feature edges during split/collapse/flip.

#include "pipeline.h"
#include <filesystem>
#include <chrono>
#include <cstdio>

#include "rxmesh/util/log.h"
#include "rxmesh/util/macros.h"
#include "rxmesh/util/util.h"

#include "glm_compat.h"
#include "gpu_bvh.cuh"
#include "vcg_remesh.h"  // for vcg_micro_collapse, vcg_clean_mesh

using namespace rxmesh;
using namespace pyrxmesh_bvh;

static char* s_fr_argv[] = {(char*)"pyrxmesh", nullptr};
static struct arg {
    std::string obj_file_name;
    std::string output_folder   = "/tmp";
    uint32_t    nx              = 66;
    uint32_t    ny              = 66;
    float       relative_len    = 1.0f;
    int         num_smooth_iters = 5;
    uint32_t    num_iter        = 15;
    uint32_t    device_id       = 0;
    char**      argv            = s_fr_argv;
    int         argc            = 1;
} Arg;

// Include the smoothing kernel from Remesh (tangential relaxation)
#include "Remesh/smoothing.cuh"

// Our feature-aware kernels
#include "feature_remesh.cuh"

// Stats (from remesh_rxmesh.cuh)
struct Stats {
    float avg_edge_len, max_edge_len, min_edge_len, avg_vertex_valence;
    int   max_vertex_valence, min_vertex_valence;
};

inline void compute_stats(RXMeshDynamic&                rx,
                          const VertexAttribute<float>* coords,
                          EdgeAttribute<float>*         edge_len,
                          VertexAttribute<int>*         vertex_valence,
                          Stats&                        stats)
{
    constexpr uint32_t blockThreads = 256;
    edge_len->reset(DEVICE, 0);
    vertex_valence->reset(DEVICE, 0);

    stats.avg_edge_len = 0;
    stats.max_edge_len = std::numeric_limits<float>::min();
    stats.min_edge_len = std::numeric_limits<float>::max();
    stats.avg_vertex_valence = 0;
    stats.max_vertex_valence = std::numeric_limits<int>::min();
    stats.min_vertex_valence = std::numeric_limits<int>::max();

    LaunchBox<blockThreads> launch_box;
    rx.update_launch_box({Op::EV}, launch_box,
        (void*)stats_kernel<float, blockThreads>, false, false, true);

    stats_kernel<float, blockThreads><<<launch_box.blocks,
        launch_box.num_threads, launch_box.smem_bytes_dyn>>>(
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

// Detect features on RXMeshDynamic (dihedral angle + boundary)
template <typename T, uint32_t blockThreads>
__global__ static void detect_features_dynamic_kernel(
    const Context            context,
    const VertexAttribute<T> coords,
    EdgeAttribute<int>       edge_feature,
    const T                  cos_threshold)
{
    auto compute = [&](EdgeHandle& eh, const VertexIterator& iter) {
        VertexHandle p  = iter[0], o0 = iter[1], q = iter[2], o1 = iter[3];
        if (!o0.is_valid() || !o1.is_valid()) { edge_feature(eh) = 1; return; }

        vec3<T> vp = coords.to_glm<3>(p), vq = coords.to_glm<3>(q);
        vec3<T> vo0 = coords.to_glm<3>(o0), vo1 = coords.to_glm<3>(o1);

        vec3<T> n0 = glm::cross(vq - vp, vo0 - vp);
        vec3<T> n1 = glm::cross(vo1 - vp, vq - vp);
        T len0 = glm::length(n0), len1 = glm::length(n1);
        if (len0 < T(1e-10) || len1 < T(1e-10)) { edge_feature(eh) = 1; return; }

        T cos_angle = glm::dot(n0 / len0, n1 / len1);
        edge_feature(eh) = (cos_angle < cos_threshold) ? 1 : 0;
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator shrd_alloc;
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, compute);
}

static std::string write_temp_obj_fr(
    const double* vertices, int nv, const int* faces, int nf)
{
    auto tmp = std::filesystem::temp_directory_path() / "pyrxmesh_fremesh_in.obj";
    FILE* f = fopen(tmp.string().c_str(), "w");
    for (int i = 0; i < nv; ++i)
        fprintf(f, "v %f %f %f\n", vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
    for (int i = 0; i < nf; ++i)
        fprintf(f, "f %d %d %d\n", faces[i*3]+1, faces[i*3+1]+1, faces[i*3+2]+1);
    fclose(f);
    return tmp.string();
}

static MeshResult read_obj_fr(const std::string& path)
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

MeshResult pipeline_feature_remesh(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double relative_len,
    int iterations,
    int smooth_iterations,
    float crease_angle_deg,
    int max_passes,
    bool verbose)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();

    if (verbose)
        fprintf(stderr, "[pyrxmesh] feature_remesh: input %d verts, %d faces, "
                "relative_len=%.3f, crease=%.1f°\n",
                num_vertices, num_faces, relative_len, crease_angle_deg);

    auto tp = clk::now();
    auto in_path = write_temp_obj_fr(vertices, num_vertices, faces, num_faces);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_fremesh_out.obj").string();
    double t_write = ms_since(tp);

    Arg.obj_file_name = in_path;
    Arg.relative_len = static_cast<float>(relative_len);
    Arg.num_iter = static_cast<uint32_t>(iterations);
    Arg.num_smooth_iters = smooth_iterations;

    tp = clk::now();
    RXMeshDynamic rx(in_path, "", 512, 3.5f, 5);
    double t_build = ms_since(tp);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("feature_remesh: mesh is not edge-manifold");

    constexpr uint32_t blockThreads = 256;

    // Set up attributes
    auto coords = rx.get_input_vertex_coordinates();
    auto new_coords = rx.add_vertex_attribute<float>("newCoords", 3);
    new_coords->reset(LOCATION_ALL, 0);
    auto edge_status = rx.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);
    auto v_valence = rx.add_vertex_attribute<uint8_t>("Valence", 1);
    auto v_boundary = rx.add_vertex_attribute<bool>("BoundaryV", 1);
    auto edge_len = rx.add_edge_attribute<float>("edgeLen", 1);
    auto vertex_valence_attr = rx.add_vertex_attribute<int>("vertexValence", 1);
    auto edge_link = rx.add_edge_attribute<int8_t>("edgeLink", 1);

    // FEATURE: edge and vertex feature attributes
    auto edge_feature = rx.add_edge_attribute<int>("edgeFeature", 1, LOCATION_ALL);
    edge_feature->reset(0, DEVICE);
    auto vertex_feature = rx.add_vertex_attribute<int>("vertFeature", 1, LOCATION_ALL);
    vertex_feature->reset(0, DEVICE);

    int* d_buffer;
    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer, sizeof(int)));

    rx.get_boundary_vertices(*v_boundary);

    // ── Feature detection + erode/dilate (all GPU) ─────────────────
    tp = clk::now();
    float cos_threshold = std::cos(crease_angle_deg * M_PI / 180.0f);

    // Step 1: detect features by dihedral angle (EVDiamond query)
    {
        LaunchBox<blockThreads> lb_feat;
        rx.update_launch_box({Op::EVDiamond}, lb_feat,
            (void*)detect_features_dynamic_kernel<float, blockThreads>);
        detect_features_dynamic_kernel<float, blockThreads>
            <<<lb_feat.blocks, lb_feat.num_threads, lb_feat.smem_bytes_dyn>>>(
                rx.get_context(), *coords, *edge_feature, cos_threshold);
        CUDA_ERROR(cudaDeviceSynchronize());
    }

    // Step 2: compute bbox diagonal for erode threshold
    float bbox_diag;
    {
        float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
        for (int i = 0; i < num_vertices; i++) {
            for (int j = 0; j < 3; j++) {
                float c = static_cast<float>(vertices[i*3+j]);
                bmin[j] = std::min(bmin[j], c);
                bmax[j] = std::max(bmax[j], c);
            }
        }
        bbox_diag = std::sqrt((bmax[0]-bmin[0])*(bmax[0]-bmin[0]) +
                               (bmax[1]-bmin[1])*(bmax[1]-bmin[1]) +
                               (bmax[2]-bmin[2])*(bmax[2]-bmin[2]));
    }

    // Step 3: erode/dilate (4 steps, matching QuadWild)
    erode_dilate_features(rx, coords.get(), edge_feature.get(), 4, bbox_diag);

    // Step 4: mark feature vertices from cleaned edge features
    mark_feature_vertices(rx, edge_feature.get(), vertex_feature.get());

    double t_features = ms_since(tp);

    if (verbose) {
        int nfeat = 0;
        edge_feature->move(DEVICE, HOST);
        rx.update_host();
        rx.for_each_edge(HOST, [&](const EdgeHandle& eh) {
            if ((*edge_feature)(eh)) nfeat++;
        });
        edge_feature->move(HOST, DEVICE);
        fprintf(stderr, "[pyrxmesh] feature_remesh: %d feature edges (after erode/dilate), %.1f ms\n",
                nfeat, t_features);
    }

    // Compute edge length stats
    Stats stats;
    compute_stats(rx, coords.get(), edge_len.get(), vertex_valence_attr.get(), stats);

    const float low_edge_len = (4.f / 5.f) * Arg.relative_len * stats.avg_edge_len;
    const float low_edge_len_sq = low_edge_len * low_edge_len;
    const float high_edge_len = (4.f / 3.f) * Arg.relative_len * stats.avg_edge_len;
    const float high_edge_len_sq = high_edge_len * high_edge_len;

    if (verbose)
        fprintf(stderr, "[pyrxmesh] thresholds: split_above=%.6f collapse_below=%.6f "
                "relative_len=%.4f avg_edge=%.6f\n",
                high_edge_len, low_edge_len, Arg.relative_len, stats.avg_edge_len);

    // Adaptive sizing attribute (1.0 = uniform for pass 1)
    auto sizing = rx.add_vertex_attribute<float>("sizing", 1, LOCATION_ALL);
    sizing->reset(1.0f, DEVICE);

    Timers<GPUTimer> timers;
    timers.add("SplitTotal"); timers.add("Split");
    timers.add("SplitCleanup"); timers.add("SplitSlice");
    timers.add("CollapseTotal"); timers.add("Collapse");
    timers.add("CollapseCleanup"); timers.add("CollapseSlice");
    timers.add("FlipTotal"); timers.add("Flip");
    timers.add("FlipCleanup"); timers.add("FlipSlice");
    timers.add("SmoothTotal");

    // ── BVH setup: build on original mesh for surface projection ────
    tp = clk::now();

    // Upload original mesh to flat GPU arrays
    float* d_ref_V = nullptr;
    int* d_ref_F = nullptr;
    CUDA_ERROR(cudaMalloc(&d_ref_V, num_vertices * 3 * sizeof(float)));
    CUDA_ERROR(cudaMalloc(&d_ref_F, num_faces * 3 * sizeof(int)));
    {
        std::vector<float> ref_v(num_vertices * 3);
        for (int i = 0; i < num_vertices * 3; i++)
            ref_v[i] = static_cast<float>(vertices[i]);
        CUDA_ERROR(cudaMemcpy(d_ref_V, ref_v.data(),
            num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_ref_F, faces,
            num_faces * 3 * sizeof(int), cudaMemcpyHostToDevice));
    }

    GpuBVH bvh;
    gpu_bvh_build(bvh, d_ref_V, d_ref_F, num_faces, num_vertices);
    double t_bvh_build = ms_since(tp);

    if (verbose)
        fprintf(stderr, "[pyrxmesh] feature_remesh: BVH built on %d faces, %.1f ms\n",
                num_faces, t_bvh_build);

    // Pre-compute global_id attribute (maps VertexHandle → global index on device)
    auto global_id = rx.add_vertex_attribute<uint32_t>("globalId", 1);
    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        (*global_id)(vh) = rx.map_to_global(vh);
    });
    global_id->move(HOST, DEVICE);

    // Allocate flat GPU arrays for BVH queries
    // These are re-allocated if vertex count changes (after split/collapse)
    uint32_t flat_capacity = rx.get_num_vertices() * 2;  // room for splits
    float* d_flat_V = nullptr;
    NearestResult* d_bvh_results = nullptr;
    CUDA_ERROR(cudaMalloc(&d_flat_V, flat_capacity * 3 * sizeof(float)));
    CUDA_ERROR(cudaMalloc(&d_bvh_results, flat_capacity * sizeof(NearestResult)));

    double t_proj_total = 0;

    // ── Main iteration loop ─────────────────────────────────────────
    tp = clk::now();
    for (uint32_t iter = 0; iter < Arg.num_iter; ++iter) {
        auto t_iter = clk::now();

        if (verbose) {
            uint32_t lv = rx.get_num_vertices(true);
            uint32_t le = rx.get_num_edges(true);
            uint32_t lf = rx.get_num_faces(true);
            fprintf(stderr, "  [gpu iter %u/%u] V=%u E=%u F=%u\n",
                    iter, Arg.num_iter, lv, le, lf);
        }

        feature_split_long_edges(rx, coords.get(),
            edge_status.get(), v_boundary.get(),
            high_edge_len_sq, low_edge_len_sq, timers, d_buffer);

        // Re-detect features after split via dihedral angle.
        // Mathematically equivalent to inheritance (see boffins session).
        {
            edge_feature->reset(0, DEVICE);
            LaunchBox<blockThreads> lb_fd;
            rx.update_launch_box({Op::EVDiamond}, lb_fd,
                (void*)detect_features_dynamic_kernel<float, blockThreads>);
            detect_features_dynamic_kernel<float, blockThreads>
                <<<lb_fd.blocks, lb_fd.num_threads, lb_fd.smem_bytes_dyn>>>(
                    rx.get_context(), *coords, *edge_feature, cos_threshold);
            CUDA_ERROR(cudaDeviceSynchronize());
            mark_feature_vertices(rx, edge_feature.get(), vertex_feature.get());
        }

        // Count how many edges are below collapse threshold
        if (verbose) {
            Stats post_stats;
            compute_stats(rx, coords.get(), edge_len.get(), vertex_valence_attr.get(), post_stats);
            fprintf(stderr, "    [gpu] edge stats: avg=%.6f min=%.6f max=%.6f\n",
                    post_stats.avg_edge_len, post_stats.min_edge_len, post_stats.max_edge_len);
        }

        if (verbose)
            fprintf(stderr, "    [gpu] after split:    V=%u E=%u F=%u\n",
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));

        // Run collapse multiple times to approximate CPU's sequential cascade.
        // Each call re-marks edges from scratch against the updated mesh.
        for (int col_pass = 0; col_pass < 1; col_pass++) {
            uint32_t pre_v = rx.get_num_vertices(true);
            feature_collapse_short_edges(rx, coords.get(),
                edge_status.get(), v_boundary.get(),
                edge_feature.get(), vertex_feature.get(),
                low_edge_len_sq, high_edge_len_sq, timers, d_buffer);
            uint32_t post_v = rx.get_num_vertices(true);
            if (verbose)
                fprintf(stderr, "    [gpu] collapse pass %d: V %u → %u (-%u)\n",
                        col_pass, pre_v, post_v, pre_v - post_v);
            if (pre_v == post_v) break;  // no progress, stop
        }

        // Re-detect features after collapse
        {
            edge_feature->reset(0, DEVICE);
            LaunchBox<blockThreads> lb_fd;
            rx.update_launch_box({Op::EVDiamond}, lb_fd,
                (void*)detect_features_dynamic_kernel<float, blockThreads>);
            detect_features_dynamic_kernel<float, blockThreads>
                <<<lb_fd.blocks, lb_fd.num_threads, lb_fd.smem_bytes_dyn>>>(
                    rx.get_context(), *coords, *edge_feature, cos_threshold);
            CUDA_ERROR(cudaDeviceSynchronize());
            mark_feature_vertices(rx, edge_feature.get(), vertex_feature.get());
        }

        if (verbose)
            fprintf(stderr, "    [gpu] after collapse: V=%u E=%u F=%u\n",
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));

        feature_equalize_valences(rx, coords.get(),
            v_valence.get(), edge_status.get(), edge_link.get(),
            v_boundary.get(), edge_feature.get(), vertex_feature.get(),
            sizing.get(), timers, d_buffer);

        if (verbose)
            fprintf(stderr, "    [gpu] after flip:     V=%u E=%u F=%u\n",
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));

        tangential_relaxation(rx, coords.get(), new_coords.get(),
            v_boundary.get(), Arg.num_smooth_iters, timers);
        std::swap(new_coords, coords);

        if (verbose)
            fprintf(stderr, "    [gpu] after smooth:   V=%u E=%u F=%u\n",
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));

        // ── BVH surface projection after smooth ─────────────────
        auto t_proj = clk::now();

        // Update global_id if topology changed (split/collapse add/remove verts)
        // Re-map on host (cheap, <1ms)
        uint32_t nv_current = rx.get_num_vertices();
        global_id->reset(UINT32_MAX, HOST);
        rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
            (*global_id)(vh) = rx.map_to_global(vh);
        });
        global_id->move(HOST, DEVICE);

        // Reallocate if needed
        if (nv_current > flat_capacity) {
            flat_capacity = nv_current * 2;
            CUDA_ERROR(cudaFree(d_flat_V));
            CUDA_ERROR(cudaFree(d_bvh_results));
            CUDA_ERROR(cudaMalloc(&d_flat_V, flat_capacity * 3 * sizeof(float)));
            CUDA_ERROR(cudaMalloc(&d_bvh_results, flat_capacity * sizeof(NearestResult)));
        }

        // Flatten: RXMesh coords → flat GPU array (via host)
        coords->move(DEVICE, HOST);
        {
            std::vector<float> h_flat(nv_current * 3, 0.0f);
            rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
                uint32_t gid = rx.map_to_global(vh);
                h_flat[gid*3+0] = (*coords)(vh, 0);
                h_flat[gid*3+1] = (*coords)(vh, 1);
                h_flat[gid*3+2] = (*coords)(vh, 2);
            });
            CUDA_ERROR(cudaMemcpy(d_flat_V, h_flat.data(),
                nv_current * 3 * sizeof(float), cudaMemcpyHostToDevice));
        }

        // BVH nearest-point query
        gpu_bvh_nearest(bvh, d_flat_V, nv_current, d_bvh_results);

        // Scatter back: write projected positions to RXMesh coords
        {
            std::vector<NearestResult> h_results(nv_current);
            CUDA_ERROR(cudaMemcpy(h_results.data(), d_bvh_results,
                nv_current * sizeof(NearestResult), cudaMemcpyDeviceToHost));

            rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
                uint32_t gid = rx.map_to_global(vh);
                // Project ALL vertices to surface (matching CPU behavior)
                const auto& r = h_results[gid];
                (*coords)(vh, 0) = r.nearest_x;
                (*coords)(vh, 1) = r.nearest_y;
                (*coords)(vh, 2) = r.nearest_z;
            });
            coords->move(HOST, DEVICE);
        }

        double proj_ms = ms_since(t_proj);
        t_proj_total += proj_ms;

        if (verbose) {
            fprintf(stderr, "    [gpu] after project:  V=%u E=%u F=%u\n",
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));
            fprintf(stderr, "  [iter %u] done in %.1f ms (proj %.1f ms)\n",
                    iter, ms_since(t_iter), proj_ms);
        }
    }
    CUDA_ERROR(cudaDeviceSynchronize());
    double t_pass1 = ms_since(tp);

    // ── Checkpoint pass 1 result via OBJ round-trip ────────────────
    tp = clk::now();
    rx.update_host();
    coords->move(DEVICE, HOST);
    auto mid_path = (std::filesystem::temp_directory_path() / "pyrxmesh_fremesh_mid.obj").string();
    rx.export_obj(mid_path, *coords);
    auto mid_result = read_obj_fr(mid_path);
    double t_checkpoint = ms_since(tp);

    // Free pass 1 GPU resources
    gpu_bvh_free(bvh);
    CUDA_ERROR(cudaFree(d_ref_V));
    CUDA_ERROR(cudaFree(d_ref_F));
    CUDA_ERROR(cudaFree(d_flat_V));
    CUDA_ERROR(cudaFree(d_bvh_results));
    CUDA_ERROR(cudaFree(d_buffer));

    if (verbose)
        fprintf(stderr, "[pyrxmesh] feature_remesh: pass1 done, %d verts, %d faces, "
                "checkpoint %.1f ms\n",
                mid_result.num_vertices, mid_result.num_faces, t_checkpoint);

    if (max_passes <= 1) {
        std::filesystem::remove(in_path);
        std::filesystem::remove(mid_path);
        if (verbose)
            fprintf(stderr, "[pyrxmesh] feature_remesh: returning after pass 1 (max_passes=%d)\n", max_passes);
        return mid_result;
    }

    // ── Micro-edge collapse (CPU, exact QuadWild match) ──────────────
    // Collapses shortest edge of triangles with QualityRadii <= 0.01
    tp = clk::now();
    auto micro_result = vcg_micro_collapse(
        mid_result.vertices.data(), mid_result.num_vertices,
        mid_result.faces.data(), mid_result.num_faces,
        0.01f, 2, verbose);
    double t_micro = ms_since(tp);

    // ── VCG cleanup: SolveGeometricArtifacts ─────────────────────────
    tp = clk::now();
    auto cleaned = vcg_clean_mesh(
        micro_result.vertices.data(), micro_result.num_vertices,
        micro_result.faces.data(), micro_result.num_faces, verbose);
    double t_cleanup = ms_since(tp);

    // ── Adaptive pass 2: fresh RXMeshDynamic from cleaned checkpoint ──
    tp = clk::now();

    auto pass2_path = (std::filesystem::temp_directory_path() / "pyrxmesh_fremesh_p2.obj").string();
    {
        FILE* f = fopen(pass2_path.c_str(), "w");
        for (int i = 0; i < cleaned.num_vertices; i++)
            fprintf(f, "v %.8g %.8g %.8g\n",
                    cleaned.vertices[i*3], cleaned.vertices[i*3+1], cleaned.vertices[i*3+2]);
        for (int i = 0; i < cleaned.num_faces; i++)
            fprintf(f, "f %d %d %d\n",
                    cleaned.faces[i*3]+1, cleaned.faces[i*3+1]+1, cleaned.faces[i*3+2]+1);
        fclose(f);
    }

    Arg.obj_file_name = pass2_path;
    RXMeshDynamic rx2(pass2_path, "", 512, 3.5f, 5);
    double t_build2 = ms_since(tp);

    if (verbose)
        fprintf(stderr, "[pyrxmesh] feature_remesh: pass2 mesh built, %u V, %u F, %.1f ms\n",
                rx2.get_num_vertices(), rx2.get_num_faces(), t_build2);

    // Set up pass 2 attributes
    auto coords2 = rx2.get_input_vertex_coordinates();
    auto new_coords2 = rx2.add_vertex_attribute<float>("newCoords", 3);
    new_coords2->reset(LOCATION_ALL, 0);
    auto edge_status2 = rx2.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);
    auto v_valence2 = rx2.add_vertex_attribute<uint8_t>("Valence", 1);
    auto v_boundary2 = rx2.add_vertex_attribute<bool>("BoundaryV", 1);
    auto edge_link2 = rx2.add_edge_attribute<int8_t>("edgeLink", 1);
    auto edge_feature2 = rx2.add_edge_attribute<int>("edgeFeature", 1, LOCATION_ALL);
    edge_feature2->reset(0, DEVICE);
    auto vertex_feature2 = rx2.add_vertex_attribute<int>("vertFeature", 1, LOCATION_ALL);
    vertex_feature2->reset(0, DEVICE);
    auto sizing2 = rx2.add_vertex_attribute<float>("sizing", 1, LOCATION_ALL);
    sizing2->reset(1.0f, DEVICE);

    int* d_buffer2;
    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer2, sizeof(int)));

    rx2.get_boundary_vertices(*v_boundary2);

    // Re-detect features on the remeshed mesh
    {
        LaunchBox<blockThreads> lb2;
        rx2.update_launch_box({Op::EVDiamond}, lb2,
            (void*)detect_features_dynamic_kernel<float, blockThreads>);
        detect_features_dynamic_kernel<float, blockThreads>
            <<<lb2.blocks, lb2.num_threads, lb2.smem_bytes_dyn>>>(
                rx2.get_context(), *coords2, *edge_feature2, cos_threshold);
        CUDA_ERROR(cudaDeviceSynchronize());
        mark_feature_vertices(rx2, edge_feature2.get(), vertex_feature2.get());
    }

    // Compute adaptive sizing
    compute_adaptive_sizing(rx2, coords2.get(), sizing2.get(), 0.3f, 3.0f);
    CUDA_ERROR(cudaDeviceSynchronize());

    if (verbose) {
        rx2.update_host();
        sizing2->move(DEVICE, HOST);
        float min_s = 999.f, max_s = 0.f, sum_s = 0.f;
        int cnt_s = 0;
        rx2.for_each_vertex(HOST, [&](const VertexHandle& vh) {
            float s = (*sizing2)(vh, 0);
            min_s = std::min(min_s, s); max_s = std::max(max_s, s);
            sum_s += s; cnt_s++;
        });
        sizing2->move(HOST, DEVICE);
        fprintf(stderr, "[pyrxmesh] feature_remesh: pass2 sizing min=%.2f avg=%.2f max=%.2f\n",
                min_s, cnt_s > 0 ? sum_s / cnt_s : 0, max_s);
    }

    // Compute pass 2 edge length stats
    auto edge_len2 = rx2.add_edge_attribute<float>("edgeLen", 1);
    auto vv_attr2 = rx2.add_vertex_attribute<int>("vertexValence", 1);
    Stats stats2;
    compute_stats(rx2, coords2.get(), edge_len2.get(), vv_attr2.get(), stats2);

    const float low2 = (4.f / 5.f) * Arg.relative_len * stats2.avg_edge_len;
    const float low2_sq = low2 * low2;
    const float high2 = (4.f / 3.f) * Arg.relative_len * stats2.avg_edge_len;
    const float high2_sq = high2 * high2;

    // TODO: BVH projection in pass 2 causes segfault after dynamic topology changes.
    // Needs investigation — likely stale device pointers after cavity ops.
    // VCG's adaptive pass uses maxSurfDist constraint inside remeshing, not post-hoc projection.
#if 0
    // Build BVH on ORIGINAL mesh for pass 2 projection
    float* d_ref_V2 = nullptr;
    int* d_ref_F2 = nullptr;
    CUDA_ERROR(cudaMalloc(&d_ref_V2, num_vertices * 3 * sizeof(float)));
    CUDA_ERROR(cudaMalloc(&d_ref_F2, num_faces * 3 * sizeof(int)));
    {
        std::vector<float> ref_v(num_vertices * 3);
        for (int i = 0; i < num_vertices * 3; i++)
            ref_v[i] = static_cast<float>(vertices[i]);
        CUDA_ERROR(cudaMemcpy(d_ref_V2, ref_v.data(),
            num_vertices * 3 * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_ref_F2, faces,
            num_faces * 3 * sizeof(int), cudaMemcpyHostToDevice));
    }
    GpuBVH bvh2;
    gpu_bvh_build(bvh2, d_ref_V2, d_ref_F2, num_faces, num_vertices);

    uint32_t flat_capacity2 = rx2.get_num_vertices() * 2;
    float* d_flat_V2 = nullptr;
    NearestResult* d_bvh_results2 = nullptr;
    CUDA_ERROR(cudaMalloc(&d_flat_V2, flat_capacity2 * 3 * sizeof(float)));
    CUDA_ERROR(cudaMalloc(&d_bvh_results2, flat_capacity2 * sizeof(NearestResult)));
#endif

    Timers<GPUTimer> timers2;
    timers2.add("SplitTotal"); timers2.add("Split");
    timers2.add("SplitCleanup"); timers2.add("SplitSlice");
    timers2.add("CollapseTotal"); timers2.add("Collapse");
    timers2.add("CollapseCleanup"); timers2.add("CollapseSlice");
    timers2.add("FlipTotal"); timers2.add("Flip");
    timers2.add("FlipCleanup"); timers2.add("FlipSlice");
    timers2.add("SmoothTotal");

    tp = clk::now();
    for (uint32_t iter = 0; iter < Arg.num_iter; ++iter) {
        auto t_iter = clk::now();

        if (verbose)
            fprintf(stderr, "  [pass2 iter %u/%u] V=%u E=%u F=%u P=%u\n",
                    iter, Arg.num_iter, rx2.get_num_vertices(),
                    rx2.get_num_edges(), rx2.get_num_faces(), rx2.get_num_patches());

        feature_split_long_edges(rx2, coords2.get(),
            edge_status2.get(), v_boundary2.get(),
            high2_sq, low2_sq, timers2, d_buffer2);

        feature_collapse_short_edges(rx2, coords2.get(),
            edge_status2.get(), v_boundary2.get(),
            edge_feature2.get(), vertex_feature2.get(),
            low2_sq, high2_sq, timers2, d_buffer2);

        feature_equalize_valences(rx2, coords2.get(),
            v_valence2.get(), edge_status2.get(), edge_link2.get(),
            v_boundary2.get(), edge_feature2.get(), vertex_feature2.get(),
            sizing2.get(), timers2, d_buffer2);

        tangential_relaxation(rx2, coords2.get(), new_coords2.get(),
            v_boundary2.get(), Arg.num_smooth_iters, timers2);
        std::swap(new_coords2, coords2);

        // BVH projection disabled in pass 2 — see TODO above
#if 0
        {
            uint32_t nv2 = rx2.get_num_vertices();
            if (nv2 > flat_capacity2) {
                flat_capacity2 = nv2 * 2;
                CUDA_ERROR(cudaFree(d_flat_V2));
                CUDA_ERROR(cudaFree(d_bvh_results2));
                CUDA_ERROR(cudaMalloc(&d_flat_V2, flat_capacity2 * 3 * sizeof(float)));
                CUDA_ERROR(cudaMalloc(&d_bvh_results2, flat_capacity2 * sizeof(NearestResult)));
            }

            coords2->move(DEVICE, HOST);
            rx2.update_host();
            std::vector<float> h_flat(nv2 * 3, 0.0f);
            rx2.for_each_vertex(HOST, [&](const VertexHandle& vh) {
                uint32_t gid = rx2.map_to_global(vh);
                h_flat[gid*3+0] = (*coords2)(vh, 0);
                h_flat[gid*3+1] = (*coords2)(vh, 1);
                h_flat[gid*3+2] = (*coords2)(vh, 2);
            });
            CUDA_ERROR(cudaMemcpy(d_flat_V2, h_flat.data(),
                nv2 * 3 * sizeof(float), cudaMemcpyHostToDevice));
            gpu_bvh_nearest(bvh2, d_flat_V2, nv2, d_bvh_results2);

            std::vector<NearestResult> h_results(nv2);
            CUDA_ERROR(cudaMemcpy(h_results.data(), d_bvh_results2,
                nv2 * sizeof(NearestResult), cudaMemcpyDeviceToHost));
            rx2.for_each_vertex(HOST, [&](const VertexHandle& vh) {
                uint32_t gid = rx2.map_to_global(vh);
                if ((*vertex_feature2)(vh) > 0) return;
                (*coords2)(vh, 0) = h_results[gid].nearest_x;
                (*coords2)(vh, 1) = h_results[gid].nearest_y;
                (*coords2)(vh, 2) = h_results[gid].nearest_z;
            });
            coords2->move(HOST, DEVICE);
        }
#endif

        if (verbose)
            fprintf(stderr, "  [pass2 iter %u] done in %.1f ms\n", iter, ms_since(t_iter));
    }
    CUDA_ERROR(cudaDeviceSynchronize());
    double t_pass2 = ms_since(tp);
    double t_gpu = t_pass1 + t_pass2;

    // Free pass 2 resources
#if 0
    gpu_bvh_free(bvh2);
    CUDA_ERROR(cudaFree(d_ref_V2));
    CUDA_ERROR(cudaFree(d_ref_F2));
    CUDA_ERROR(cudaFree(d_flat_V2));
    CUDA_ERROR(cudaFree(d_bvh_results2));
#endif
    CUDA_ERROR(cudaFree(d_buffer2));

    // Export final result
    tp = clk::now();
    rx2.update_host();
    coords2->move(DEVICE, HOST);
    rx2.export_obj(out_path, *coords2);
    auto result = read_obj_fr(out_path);
    double t_readback = ms_since(tp);

    std::filesystem::remove(in_path);
    std::filesystem::remove(mid_path);
    std::filesystem::remove(pass2_path);
    std::filesystem::remove(out_path);

    if (verbose) {
        fprintf(stderr, "[pyrxmesh] feature_remesh: obj_write=%.1fms, mesh_build=%.1fms, "
                "features=%.1fms, pass1=%.1fms, micro=%.1fms, cleanup=%.1fms, pass2=%.1fms, readback=%.1fms\n",
                t_write, t_build, t_features, t_pass1, t_micro, t_cleanup, t_pass2, t_readback);
        fprintf(stderr, "[pyrxmesh] feature_remesh: bvh_build=%.1fms, projection=%.1fms\n",
                t_bvh_build, t_proj_total);
        fprintf(stderr, "[pyrxmesh] feature_remesh: output %d verts, %d faces, total %.1f ms\n",
                result.num_vertices, result.num_faces, ms_since(t0));
    }

    return result;
}
