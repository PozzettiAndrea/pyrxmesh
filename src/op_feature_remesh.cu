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
#include "op_cross_collapse.cuh"

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

// CPU-side patch validation — copies device data to host, no GPU kernels.
// Checks all invariants the CavityManager assumes.
inline bool validate_patches_cpu(RXMeshDynamic& rx, const char* label) {
    CUDA_ERROR(cudaDeviceSynchronize());

    uint32_t num_patches = rx.get_num_patches();
    int total_bad = 0;
    int first_bad_patch = -1;
    uint16_t max_v_cap = 0, max_e_cap = 0, max_f_cap = 0;

    auto report = [&](uint32_t p, const char* msg) {
        if (total_bad < 5)
            fprintf(stderr, "    [validate] patch %u: %s\n", p, msg);
        total_bad++;
        if (first_bad_patch < 0) first_bad_patch = p;
    };

    for (uint32_t p = 0; p < num_patches; p++) {
        const auto& pi = rx.get_patch(p);

        uint16_t v_cap = pi.vertices_capacity;
        uint16_t e_cap = pi.edges_capacity;
        uint16_t f_cap = pi.faces_capacity;
        max_v_cap = std::max(max_v_cap, v_cap);
        max_e_cap = std::max(max_e_cap, e_cap);
        max_f_cap = std::max(max_f_cap, f_cap);

        // Copy counts from device
        uint16_t num_v = 0, num_e = 0, num_f = 0;
        if (pi.num_vertices)
            cudaMemcpy(&num_v, pi.num_vertices, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        if (pi.num_edges)
            cudaMemcpy(&num_e, pi.num_edges, sizeof(uint16_t), cudaMemcpyDeviceToHost);
        if (pi.num_faces)
            cudaMemcpy(&num_f, pi.num_faces, sizeof(uint16_t), cudaMemcpyDeviceToHost);

        // 1. Counts within capacity
        if (num_v > v_cap) { char buf[128]; snprintf(buf, 128, "num_v=%u > v_cap=%u", num_v, v_cap); report(p, buf); continue; }
        if (num_e > e_cap) { char buf[128]; snprintf(buf, 128, "num_e=%u > e_cap=%u", num_e, e_cap); report(p, buf); continue; }
        if (num_f > f_cap) { char buf[128]; snprintf(buf, 128, "num_f=%u > f_cap=%u", num_f, f_cap); report(p, buf); continue; }

        // Copy EV, FE, active bitmasks from device
        std::vector<uint16_t> h_ev(2 * e_cap, INVALID16);
        std::vector<uint16_t> h_fe(3 * f_cap, INVALID16);
        if (pi.ev && e_cap > 0)
            cudaMemcpy(h_ev.data(), pi.ev, 2 * e_cap * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        if (pi.fe && f_cap > 0)
            cudaMemcpy(h_fe.data(), pi.fe, 3 * f_cap * sizeof(uint16_t), cudaMemcpyDeviceToHost);

        // Copy active bitmasks
        uint32_t v_mask_words = (v_cap + 31) / 32;
        uint32_t e_mask_words = (e_cap + 31) / 32;
        uint32_t f_mask_words = (f_cap + 31) / 32;
        std::vector<uint32_t> h_active_v(v_mask_words, 0);
        std::vector<uint32_t> h_active_e(e_mask_words, 0);
        std::vector<uint32_t> h_active_f(f_mask_words, 0);
        if (pi.active_mask_v)
            cudaMemcpy(h_active_v.data(), pi.active_mask_v, v_mask_words * 4, cudaMemcpyDeviceToHost);
        if (pi.active_mask_e)
            cudaMemcpy(h_active_e.data(), pi.active_mask_e, e_mask_words * 4, cudaMemcpyDeviceToHost);
        if (pi.active_mask_f)
            cudaMemcpy(h_active_f.data(), pi.active_mask_f, f_mask_words * 4, cudaMemcpyDeviceToHost);

        auto is_active = [](const std::vector<uint32_t>& mask, uint16_t idx) -> bool {
            return (mask[idx / 32] >> (idx % 32)) & 1;
        };

        // 2. EV bounds check (all edges, not just active)
        for (uint16_t e = 0; e < num_e; e++) {
            uint16_t va = h_ev[2 * e];
            uint16_t vb = h_ev[2 * e + 1];
            if (va >= v_cap || vb >= v_cap) {
                char buf[128]; snprintf(buf, 128, "ev[%u]=(%u,%u) > v_cap=%u", e, va, vb, v_cap);
                report(p, buf); break;
            }
        }

        // 3. Active edges must have active vertices
        int bad_active = 0;
        for (uint16_t e = 0; e < num_e; e++) {
            if (!is_active(h_active_e, e)) continue;
            uint16_t va = h_ev[2 * e];
            uint16_t vb = h_ev[2 * e + 1];
            if (va < v_cap && !is_active(h_active_v, va)) {
                if (bad_active < 2) {
                    char buf[128]; snprintf(buf, 128, "active edge %u has inactive vertex %u", e, va);
                    report(p, buf);
                }
                bad_active++;
            }
            if (vb < v_cap && !is_active(h_active_v, vb)) {
                if (bad_active < 2) {
                    char buf[128]; snprintf(buf, 128, "active edge %u has inactive vertex %u", e, vb);
                    report(p, buf);
                }
                bad_active++;
            }
        }

        // 4. FE bounds + active faces must have active edges
        for (uint16_t f = 0; f < num_f; f++) {
            for (int j = 0; j < 3; j++) {
                uint16_t eid = h_fe[3 * f + j];
                if (eid >= e_cap) {
                    char buf[128]; snprintf(buf, 128, "fe[%u][%d]=%u > e_cap=%u", f, j, eid, e_cap);
                    report(p, buf); goto next_patch;
                }
            }
            if (is_active(h_active_f, f)) {
                for (int j = 0; j < 3; j++) {
                    uint16_t eid = h_fe[3 * f + j];
                    if (eid < e_cap && !is_active(h_active_e, eid)) {
                        char buf[128]; snprintf(buf, 128, "active face %u has inactive edge %u", f, eid);
                        report(p, buf); break;
                    }
                }
            }
        }
        next_patch:;
    }

    // Compute shared memory budget
    // CavityManager needs: cavity_id_v/e/f + ev + fe + bitmasks + user shmem
    uint32_t cm_shmem =
        max_v_cap * 2 + max_e_cap * 2 + max_f_cap * 2 +  // cavity_id_v/e/f
        2 * max_e_cap * 2 +                                 // ev
        3 * max_f_cap * 2 +                                 // fe
        7 * ((max_v_cap + 31) / 32) * 4 +                   // vertex bitmasks
        4 * ((max_e_cap + 31) / 32) * 4 +                   // edge bitmasks
        4 * ((max_f_cap + 31) / 32) * 4;                    // face bitmasks

    fprintf(stderr, "    [validate] %s: %s (%u patches, max cap V=%u E=%u F=%u, "
            "est_shmem=%uKB)\n",
            label, total_bad == 0 ? "OK" : "FAILED",
            num_patches, max_v_cap, max_e_cap, max_f_cap,
            cm_shmem / 1024);
    if (total_bad > 0)
        fprintf(stderr, "    [validate] %d issues (first bad: patch %d)\n",
                total_bad, first_bad_patch);

    return total_bad == 0;
}

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

// Uses write_temp_obj_fast from pipeline.h

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
    float flip_normal_thr,
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
    auto fv = flat_faces_to_fv(faces, num_faces);
    auto vv = flat_verts_to_vv(vertices, num_vertices);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_fremesh_out.obj").string();
    double t_write = ms_since(tp);

    Arg.obj_file_name = "pyrxmesh_fremesh";
    Arg.relative_len = static_cast<float>(relative_len);
    Arg.num_iter = static_cast<uint32_t>(iterations);
    Arg.num_smooth_iters = smooth_iterations;

    tp = clk::now();
    RXMeshDynamic rx(fv, "", 512, 2.0f, 2);
    rx.add_vertex_coordinates(vv);
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

    // Hausdorff surface distance threshold (matching VCG: bbox.Diag() / 2500)
    const float max_surf_dist = bbox_diag / 2500.0f;

    if (verbose)
        fprintf(stderr, "[pyrxmesh] thresholds: split_above=%.6f collapse_below=%.6f "
                "relative_len=%.4f avg_edge=%.6f max_surf_dist=%.6f\n",
                high_edge_len, low_edge_len, Arg.relative_len, stats.avg_edge_len,
                max_surf_dist);

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

    // Device-side BVH context for on-GPU projection (no host round-trip)
    auto bvh_ctx = gpu_bvh_device_context(bvh);

    double t_proj_total = 0;

    // Helper: compute Q max on host (for debugging sliver creation)
    auto compute_qmax = [&](const char* label) {
        rx.update_host();
        coords->move(DEVICE, HOST);
        float qmax = 0;
        int bad_count = 0;
        rx.for_each_face(HOST, [&](const FaceHandle& fh) {
            VertexHandle v0, v1, v2;
            // Get face vertices via face-vertex query isn't trivial on host...
            // Use the export approach instead
        });
        // Simple: just export and compute
        auto tmp = (std::filesystem::temp_directory_path() / "pyrxmesh_qcheck.obj").string();
        rx.export_obj(tmp, *coords);
        auto mr = read_obj_fr(tmp);
        std::filesystem::remove(tmp);
        for (int i = 0; i < mr.num_faces; i++) {
            int i0 = mr.faces[i*3], i1 = mr.faces[i*3+1], i2 = mr.faces[i*3+2];
            double p0x=mr.vertices[i0*3], p0y=mr.vertices[i0*3+1], p0z=mr.vertices[i0*3+2];
            double p1x=mr.vertices[i1*3], p1y=mr.vertices[i1*3+1], p1z=mr.vertices[i1*3+2];
            double p2x=mr.vertices[i2*3], p2y=mr.vertices[i2*3+1], p2z=mr.vertices[i2*3+2];
            // Edge lengths
            double ea = std::sqrt((p1x-p0x)*(p1x-p0x)+(p1y-p0y)*(p1y-p0y)+(p1z-p0z)*(p1z-p0z));
            double eb = std::sqrt((p2x-p1x)*(p2x-p1x)+(p2y-p1y)*(p2y-p1y)+(p2z-p1z)*(p2z-p1z));
            double ec = std::sqrt((p0x-p2x)*(p0x-p2x)+(p0y-p2y)*(p0y-p2y)+(p0z-p2z)*(p0z-p2z));
            double longest = std::max({ea, eb, ec});
            // Semi-perimeter and area
            double s = (ea+eb+ec)*0.5;
            double area_sq = s*(s-ea)*(s-eb)*(s-ec);
            double area = (area_sq > 0) ? std::sqrt(area_sq) : 0;
            // Inradius = area / s
            double inradius = (s > 1e-20) ? area / s : 0;
            // VTK aspect ratio = longest_edge / (2 * inradius * sqrt(3))
            double ar = (inradius > 1e-20) ? longest / (2.0 * inradius * std::sqrt(3.0)) : 1e10;
            if (ar > qmax) qmax = ar;
            if (ar > 100) bad_count++;
        }
        coords->move(HOST, DEVICE);
        fprintf(stderr, "      [qcheck] %s: Q_max=%.0f (%d faces with Q>100)\n",
                label, qmax, bad_count);
    };

    // ── Main iteration loop ─────────────────────────────────────────
    tp = clk::now();
    for (uint32_t iter = 0; iter < Arg.num_iter; ++iter) {
        auto t_iter = clk::now();

        if (verbose)
            fprintf(stderr, "  [gpu iter %u/%u] V=%u E=%u F=%u\n",
                    iter, Arg.num_iter,
                    rx.get_num_vertices(true),
                    rx.get_num_edges(true),
                    rx.get_num_faces(true));

        if (verbose) compute_qmax("before split");

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

        if (verbose) {
            fprintf(stderr, "    [gpu] after split:    V=%u E=%u F=%u\n",
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));
            compute_qmax("after split");
        }

        // Run collapse
        for (int col_pass = 0; col_pass < 1; col_pass++) {
            uint32_t pre_v = rx.get_num_vertices(true);
            feature_collapse_short_edges(rx, coords.get(),
                edge_status.get(), v_boundary.get(),
                edge_feature.get(), vertex_feature.get(),
                low_edge_len_sq, high_edge_len_sq, timers, d_buffer,
                bvh_ctx, max_surf_dist);
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

        if (verbose) {
            fprintf(stderr, "    [gpu] after collapse: V=%u E=%u F=%u\n",
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));
            compute_qmax("after collapse");
        }

        // CollapseCrosses: remove valence 3-4 interior vertices (matches CPU)
        // Set PYRXMESH_NO_CROSSES=1 to disable for debugging.
        if (!getenv("PYRXMESH_NO_CROSSES"))
            collapse_crosses(rx, coords.get(), edge_status.get(), v_boundary.get(),
                vertex_feature.get(), high_edge_len_sq, timers, d_buffer, verbose);

        // Re-detect features after crosses
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

        if (verbose) {
            fprintf(stderr, "    [gpu] after crosses:  V=%u E=%u F=%u\n",
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));
            compute_qmax("after crosses");
        }

        feature_equalize_valences(rx, coords.get(),
            v_valence.get(), edge_status.get(), edge_link.get(),
            v_boundary.get(), edge_feature.get(), vertex_feature.get(),
            sizing.get(), timers, d_buffer, flip_normal_thr);

        if (verbose) {
            fprintf(stderr, "    [gpu] after flip:     V=%u E=%u F=%u\n",
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));
            compute_qmax("after flip");
        }

        tangential_relaxation(rx, coords.get(), new_coords.get(),
            v_boundary.get(), Arg.num_smooth_iters, timers);
        std::swap(new_coords, coords);

        if (verbose) {
            fprintf(stderr, "    [gpu] after smooth:   V=%u E=%u F=%u\n",
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));
            compute_qmax("after smooth");
        }

        // ── BVH surface projection (entirely on device) ─────────
        auto t_proj = clk::now();

        rx.for_each_vertex(DEVICE,
            [coords = *coords, bvh_ctx] __device__(const VertexHandle vh) mutable {
                float px = coords(vh, 0);
                float py = coords(vh, 1);
                float pz = coords(vh, 2);
                auto r = pyrxmesh_bvh::gpu_bvh_query_point(bvh_ctx, px, py, pz);
                coords(vh, 0) = r.nearest_x;
                coords(vh, 1) = r.nearest_y;
                coords(vh, 2) = r.nearest_z;
            });
        CUDA_ERROR(cudaDeviceSynchronize());

        double proj_ms = ms_since(t_proj);
        t_proj_total += proj_ms;

        if (verbose) {
            fprintf(stderr, "    [gpu] after project:  V=%u E=%u F=%u\n",
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));
            compute_qmax("after project");
        }

        // NOTE: GPU micro-collapse (quality-based) was here but causes crashes
        // on iteration 2+ because slice_patches inside micro-collapse corrupts
        // patch state. Micro-collapse is now done at the pipeline level (Python)
        // between calls to feature_remesh, matching QuadWild's architecture.

        if (verbose)
            fprintf(stderr, "  [iter %u] done in %.1f ms (proj %.1f ms)\n",
                    iter, ms_since(t_iter), proj_ms);
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

    // Free GPU resources
    gpu_bvh_free(bvh);
    CUDA_ERROR(cudaFree(d_ref_V));
    CUDA_ERROR(cudaFree(d_ref_F));
    CUDA_ERROR(cudaFree(d_buffer));

    if (verbose)
        fprintf(stderr, "[pyrxmesh] feature_remesh: pass1 done, %d verts, %d faces, "
                "checkpoint %.1f ms\n",
                mid_result.num_vertices, mid_result.num_faces, t_checkpoint);

    // no input file to clean up (direct array construction)
    std::filesystem::remove(mid_path);

    // Ensure all GPU work is complete before returning.
    // Without this, subsequent CUDA operations (e.g. detect_features) can hang
    // due to unfenced async ops from RXMeshDynamic destruction.
    CUDA_ERROR(cudaDeviceSynchronize());

    if (verbose) {
        fprintf(stderr, "[pyrxmesh] feature_remesh: obj_write=%.1fms, mesh_build=%.1fms, "
                "features=%.1fms, remesh=%.1fms, readback=%.1fms\n",
                t_write, t_build, t_features, t_pass1, t_checkpoint);
        fprintf(stderr, "[pyrxmesh] feature_remesh: bvh_build=%.1fms, projection=%.1fms\n",
                t_bvh_build, t_proj_total);
        fprintf(stderr, "[pyrxmesh] feature_remesh: output %d verts, %d faces, total %.1f ms\n",
                mid_result.num_vertices, mid_result.num_faces, ms_since(t0));
    }

    return mid_result;
}


// =========================================================================
// Full QuadWild GPU remesh pipeline — single RXMeshDynamic, no round-trips.
// isotropic (N iters) → micro-collapse → re-detect → adaptive (N iters)
// =========================================================================

QuadwildRemeshResult pipeline_quadwild_remesh(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double relative_len,
    int isotropic_iterations,
    int adaptive_iterations,
    int smooth_iterations,
    float crease_angle_deg,
    float micro_quality_thr,
    bool verbose)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();

    if (verbose)
        fprintf(stderr, "[pyrxmesh] quadwild_remesh: input %d verts, %d faces, "
                "iso_iters=%d, adapt_iters=%d, relative_len=%.3f\n",
                num_vertices, num_faces, isotropic_iterations,
                adaptive_iterations, relative_len);

    // ── Build RXMeshDynamic directly from arrays ──────────────────────
    auto tp = clk::now();
    auto fv = flat_faces_to_fv(faces, num_faces);
    auto vv = flat_verts_to_vv(vertices, num_vertices);
    auto out_path = (std::filesystem::temp_directory_path() / "pyrxmesh_qw_out.obj").string();

    Arg.obj_file_name = "pyrxmesh_fremesh";
    Arg.relative_len = static_cast<float>(relative_len);
    Arg.num_smooth_iters = smooth_iterations;

    RXMeshDynamic rx(fv, "", 512, 2.0f, 2);
    rx.add_vertex_coordinates(vv);
    double t_build = ms_since(tp);

    if (!rx.is_edge_manifold())
        throw std::runtime_error("quadwild_remesh: mesh is not edge-manifold");

    constexpr uint32_t blockThreads = 256;

    // ── Set up attributes ─────────────────────────────────────────────
    auto coords = rx.get_input_vertex_coordinates();
    auto new_coords = rx.add_vertex_attribute<float>("newCoords", 3);
    new_coords->reset(LOCATION_ALL, 0);
    auto edge_status = rx.add_edge_attribute<EdgeStatus>("EdgeStatus", 1);
    auto v_valence = rx.add_vertex_attribute<uint8_t>("Valence", 1);
    auto v_boundary = rx.add_vertex_attribute<bool>("BoundaryV", 1);
    auto edge_len = rx.add_edge_attribute<float>("edgeLen", 1);
    auto vertex_valence_attr = rx.add_vertex_attribute<int>("vertexValence", 1);
    auto edge_link = rx.add_edge_attribute<int8_t>("edgeLink", 1);
    auto edge_feature = rx.add_edge_attribute<int>("edgeFeature", 1, LOCATION_ALL);
    edge_feature->reset(0, DEVICE);
    auto vertex_feature = rx.add_vertex_attribute<int>("vertFeature", 1, LOCATION_ALL);
    vertex_feature->reset(0, DEVICE);
    auto sizing = rx.add_vertex_attribute<float>("sizing", 1, LOCATION_ALL);
    sizing->reset(1.0f, DEVICE);

    int* d_buffer;
    CUDA_ERROR(cudaMallocManaged((void**)&d_buffer, sizeof(int)));
    rx.get_boundary_vertices(*v_boundary);

    // ── Feature detection + erode/dilate ──────────────────────────────
    tp = clk::now();
    float cos_threshold = std::cos(crease_angle_deg * M_PI / 180.0f);
    {
        LaunchBox<blockThreads> lb_feat;
        rx.update_launch_box({Op::EVDiamond}, lb_feat,
            (void*)detect_features_dynamic_kernel<float, blockThreads>);
        detect_features_dynamic_kernel<float, blockThreads>
            <<<lb_feat.blocks, lb_feat.num_threads, lb_feat.smem_bytes_dyn>>>(
                rx.get_context(), *coords, *edge_feature, cos_threshold);
        CUDA_ERROR(cudaDeviceSynchronize());
    }
    float bbox_diag;
    {
        float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
        for (int i = 0; i < num_vertices; i++)
            for (int j = 0; j < 3; j++) {
                float c = static_cast<float>(vertices[i*3+j]);
                bmin[j] = std::min(bmin[j], c); bmax[j] = std::max(bmax[j], c);
            }
        bbox_diag = std::sqrt((bmax[0]-bmin[0])*(bmax[0]-bmin[0]) +
                               (bmax[1]-bmin[1])*(bmax[1]-bmin[1]) +
                               (bmax[2]-bmin[2])*(bmax[2]-bmin[2]));
    }
    erode_dilate_features(rx, coords.get(), edge_feature.get(), 4, bbox_diag);
    mark_feature_vertices(rx, edge_feature.get(), vertex_feature.get());
    double t_features = ms_since(tp);

    const float max_surf_dist = bbox_diag / 2500.0f;

    if (verbose)
        fprintf(stderr, "[pyrxmesh] quadwild_remesh: features done, %.1f ms, "
                "max_surf_dist=%.6f\n", t_features, max_surf_dist);

    // ── BVH for surface projection ────────────────────────────────────
    tp = clk::now();
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
    double t_bvh = ms_since(tp);

    auto bvh_ctx = gpu_bvh_device_context(bvh);

    Timers<GPUTimer> timers;
    timers.add("SplitTotal"); timers.add("Split");
    timers.add("SplitCleanup"); timers.add("SplitSlice");
    timers.add("CollapseTotal"); timers.add("Collapse");
    timers.add("CollapseCleanup"); timers.add("CollapseSlice");
    timers.add("FlipTotal"); timers.add("Flip");
    timers.add("FlipCleanup"); timers.add("FlipSlice");
    timers.add("SmoothTotal");

    // Helper lambda: run one remeshing iteration
    auto run_iteration = [&](uint32_t iter, uint32_t total_iters,
                             float& hi_sq, float& lo_sq, const char* pass_name) {
        auto t_iter = clk::now();

        // Recompute thresholds
        Stats iter_stats;
        compute_stats(rx, coords.get(), edge_len.get(),
                      vertex_valence_attr.get(), iter_stats);
        float h = (4.f / 3.f) * Arg.relative_len * iter_stats.avg_edge_len;
        float l = (4.f / 5.f) * Arg.relative_len * iter_stats.avg_edge_len;
        hi_sq = h * h; lo_sq = l * l;

        if (verbose)
            fprintf(stderr, "  [%s iter %u/%u] V=%u E=%u F=%u (split>%.4f collapse<%.4f)\n",
                    pass_name, iter, total_iters,
                    rx.get_num_vertices(true), rx.get_num_edges(true),
                    rx.get_num_faces(true), h, l);

        // Split
        feature_split_long_edges(rx, coords.get(), edge_status.get(),
            v_boundary.get(), hi_sq, lo_sq, timers, d_buffer);

        // Re-detect features after split
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
            fprintf(stderr, "    [%s] after split:    V=%u E=%u F=%u\n", pass_name,
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));

        // Collapse
        {
            uint32_t pre_v = rx.get_num_vertices(true);
            feature_collapse_short_edges(rx, coords.get(), edge_status.get(),
                v_boundary.get(), edge_feature.get(), vertex_feature.get(),
                lo_sq, hi_sq, timers, d_buffer,
                bvh_ctx, max_surf_dist);
            uint32_t post_v = rx.get_num_vertices(true);
            if (verbose)
                fprintf(stderr, "    [%s] collapse: V %u → %u (-%u)\n",
                        pass_name, pre_v, post_v, pre_v - post_v);
        }

        // Re-detect after collapse
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
            fprintf(stderr, "    [%s] after collapse: V=%u E=%u F=%u\n", pass_name,
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));

        // Cross-collapse
        if (!getenv("PYRXMESH_NO_CROSSES"))
            collapse_crosses(rx, coords.get(), edge_status.get(), v_boundary.get(),
                vertex_feature.get(), hi_sq, timers, d_buffer, verbose);

        // Re-detect after crosses
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
            fprintf(stderr, "    [%s] after crosses:  V=%u E=%u F=%u\n", pass_name,
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));

        // Flip
        feature_equalize_valences(rx, coords.get(), v_valence.get(),
            edge_status.get(), edge_link.get(), v_boundary.get(),
            edge_feature.get(), vertex_feature.get(), sizing.get(), timers, d_buffer);

        if (verbose)
            fprintf(stderr, "    [%s] after flip:     V=%u E=%u F=%u\n", pass_name,
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));

        // Smooth
        tangential_relaxation(rx, coords.get(), new_coords.get(),
            v_boundary.get(), Arg.num_smooth_iters, timers);
        std::swap(new_coords, coords);

        if (verbose)
            fprintf(stderr, "    [%s] after smooth:   V=%u E=%u F=%u\n", pass_name,
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));

        // BVH project (entirely on device)
        rx.for_each_vertex(DEVICE,
            [coords = *coords, bvh_ctx] __device__(const VertexHandle vh) mutable {
                float px = coords(vh, 0);
                float py = coords(vh, 1);
                float pz = coords(vh, 2);
                auto r = pyrxmesh_bvh::gpu_bvh_query_point(bvh_ctx, px, py, pz);
                coords(vh, 0) = r.nearest_x;
                coords(vh, 1) = r.nearest_y;
                coords(vh, 2) = r.nearest_z;
            });
        CUDA_ERROR(cudaDeviceSynchronize());

        if (verbose)
            fprintf(stderr, "    [%s] after project:  V=%u E=%u F=%u\n", pass_name,
                    rx.get_num_vertices(true), rx.get_num_edges(true), rx.get_num_faces(true));

        if (verbose)
            fprintf(stderr, "  [%s iter %u] done in %.1f ms\n",
                    pass_name, iter, ms_since(t_iter));
    };

    // Helper: export current mesh state to MeshResult
    auto export_mesh = [&]() -> MeshResult {
        rx.update_host();
        coords->move(DEVICE, HOST);
        auto tmp = (std::filesystem::temp_directory_path() / "pyrxmesh_qw_tmp.obj").string();
        rx.export_obj(tmp, *coords);
        auto result = read_obj_fr(tmp);
        std::filesystem::remove(tmp);
        return result;
    };

    // ═══ Pass 1: Isotropic ═══════════════════════════════════════════
    float hi_sq = 0, lo_sq = 0;
    tp = clk::now();
    for (uint32_t iter = 0; iter < (uint32_t)isotropic_iterations; ++iter)
        run_iteration(iter, isotropic_iterations, hi_sq, lo_sq, "iso");
    CUDA_ERROR(cudaDeviceSynchronize());
    double t_iso = ms_since(tp);

    if (verbose)
        fprintf(stderr, "[pyrxmesh] quadwild_remesh: isotropic done, V=%u F=%u, %.1f ms\n",
                rx.get_num_vertices(true), rx.get_num_faces(true), t_iso);

    QuadwildRemeshResult result;
    result.after_isotropic = export_mesh();

    // ═══ Micro-collapse ══════════════════════════════════════════════
    tp = clk::now();
    gpu_micro_collapse(rx, coords.get(), edge_status.get(), v_boundary.get(),
                       micro_quality_thr, timers, d_buffer, verbose);
    double t_micro = ms_since(tp);

    if (verbose)
        fprintf(stderr, "[pyrxmesh] quadwild_remesh: micro-collapse done, V=%u F=%u, %.1f ms\n",
                rx.get_num_vertices(true), rx.get_num_faces(true), t_micro);

    result.after_micro = export_mesh();

    // ═══ Re-detect features ══════════════════════════════════════════
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

    // ═══ Pass 2: Adaptive ════════════════════════════════════════════
    if (adaptive_iterations > 0) {
        tp = clk::now();
        for (uint32_t iter = 0; iter < (uint32_t)adaptive_iterations; ++iter)
            run_iteration(iter, adaptive_iterations, hi_sq, lo_sq, "adapt");
        CUDA_ERROR(cudaDeviceSynchronize());
        double t_adapt = ms_since(tp);

        if (verbose)
            fprintf(stderr, "[pyrxmesh] quadwild_remesh: adaptive done, V=%u F=%u, %.1f ms\n",
                    rx.get_num_vertices(true), rx.get_num_faces(true), t_adapt);
    }

    result.final_mesh = export_mesh();

    // ── Cleanup ──────────────────────────────────────────────────────
    gpu_bvh_free(bvh);
    CUDA_ERROR(cudaFree(d_ref_V));
    CUDA_ERROR(cudaFree(d_ref_F));
    CUDA_ERROR(cudaFree(d_buffer));
    // no input file to clean up (direct array construction)
    CUDA_ERROR(cudaDeviceSynchronize());

    if (verbose)
        fprintf(stderr, "[pyrxmesh] quadwild_remesh: total %.1f ms, output %d V %d F\n",
                ms_since(t0), result.final_mesh.num_vertices, result.final_mesh.num_faces);

    return result;
}
