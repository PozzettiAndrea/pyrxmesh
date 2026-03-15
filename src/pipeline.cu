// CUDA implementation of the RXMesh pipeline.
// Compiled by nvcc. Contains all RXMesh API calls and __device__ lambdas.

#include "pipeline.h"

#include <glm/gtc/constants.hpp>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/attribute.h"
#include "rxmesh/query.h"
#include "rxmesh/util/import_obj.h"
#include "rxmesh/matrix/sparse_matrix.h"
#include "rxmesh/matrix/dense_matrix.h"
#include "rxmesh/matrix/cholesky_solver.h"

#include "glm_compat.h"

// Geodesic kernel (used by persistent mesh handle)
#include "Geodesic/geodesic_kernel.cuh"

using namespace rxmesh;

// ---------------------------------------------------------------------------
// Helpers: convert between flat arrays and RXMesh's vector-of-vectors format
// ---------------------------------------------------------------------------

static std::vector<std::vector<uint32_t>> flat_faces_to_fv(
    const int* faces, int num_faces)
{
    std::vector<std::vector<uint32_t>> fv(num_faces);
    for (int i = 0; i < num_faces; ++i) {
        fv[i] = {
            static_cast<uint32_t>(faces[i * 3 + 0]),
            static_cast<uint32_t>(faces[i * 3 + 1]),
            static_cast<uint32_t>(faces[i * 3 + 2])
        };
    }
    return fv;
}

static std::vector<std::vector<float>> flat_verts_to_vv(
    const double* vertices, int num_vertices)
{
    std::vector<std::vector<float>> vv(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        vv[i] = {
            static_cast<float>(vertices[i * 3 + 0]),
            static_cast<float>(vertices[i * 3 + 1]),
            static_cast<float>(vertices[i * 3 + 2])
        };
    }
    return vv;
}

// Write a mesh to a temp OBJ file, returning the path
static std::string write_temp_obj(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    const std::string& prefix = "pyrxmesh_in")
{
    auto tmp = std::filesystem::temp_directory_path() / (prefix + ".obj");
    std::string path = tmp.string();
    FILE* f = fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("Cannot create temp OBJ: " + path);

    for (int i = 0; i < num_vertices; ++i) {
        fprintf(f, "v %f %f %f\n",
            vertices[i*3+0], vertices[i*3+1], vertices[i*3+2]);
    }
    for (int i = 0; i < num_faces; ++i) {
        fprintf(f, "f %d %d %d\n",
            faces[i*3+0]+1, faces[i*3+1]+1, faces[i*3+2]+1);
    }
    fclose(f);
    return path;
}

// Read a mesh result from an OBJ file
static MeshResult read_obj_result(const std::string& path)
{
    std::vector<std::vector<float>>    verts;
    std::vector<std::vector<uint32_t>> faces;

    if (!import_obj(path, verts, faces)) {
        throw std::runtime_error("Failed to read result OBJ: " + path);
    }

    MeshResult result;
    result.num_vertices = static_cast<int>(verts.size());
    result.num_faces    = static_cast<int>(faces.size());

    result.vertices.resize(result.num_vertices * 3);
    for (int i = 0; i < result.num_vertices; ++i) {
        result.vertices[i*3+0] = static_cast<double>(verts[i][0]);
        result.vertices[i*3+1] = static_cast<double>(verts[i][1]);
        result.vertices[i*3+2] = static_cast<double>(verts[i][2]);
    }
    result.faces.resize(result.num_faces * 3);
    for (int i = 0; i < result.num_faces; ++i) {
        result.faces[i*3+0] = static_cast<int>(faces[i][0]);
        result.faces[i*3+1] = static_cast<int>(faces[i][1]);
        result.faces[i*3+2] = static_cast<int>(faces[i][2]);
    }
    return result;
}

// Extract mesh result from RXMeshStatic + coordinates attribute
static MeshResult extract_static_mesh(
    RXMeshStatic& rx,
    VertexAttribute<float>& coords)
{
    coords.move(DEVICE, HOST);

    MeshResult result;
    result.num_vertices = static_cast<int>(rx.get_num_vertices());
    result.num_faces    = static_cast<int>(rx.get_num_faces());
    result.vertices.resize(result.num_vertices * 3);

    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rx.map_to_global(vh);
        for (uint32_t i = 0; i < 3; ++i) {
            result.vertices[v_id * 3 + i] = static_cast<double>(coords(vh, i));
        }
    });

    // Extract faces via temp OBJ export + reimport
    auto tmp_out = std::filesystem::temp_directory_path() / "pyrxmesh_faces.obj";
    std::string out_path = tmp_out.string();
    rx.export_obj(out_path, coords);
    auto face_result = read_obj_result(out_path);
    result.faces = std::move(face_result.faces);
    result.num_faces = static_cast<int>(result.faces.size() / 3);
    std::filesystem::remove(out_path);

    return result;
}

// ---------------------------------------------------------------------------
// pipeline_init
// ---------------------------------------------------------------------------

static bool g_initialized = false;

void pipeline_init(int device_id)
{
    if (!g_initialized) {
        rx_init(device_id);
        g_initialized = true;
    }
}

static void ensure_init()
{
    if (!g_initialized) {
        pipeline_init(0);
    }
}

// ---------------------------------------------------------------------------
// pipeline_mesh_info
// ---------------------------------------------------------------------------

MeshInfo pipeline_mesh_info(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces)
{
    ensure_init();

    auto fv = flat_faces_to_fv(faces, num_faces);
    RXMeshStatic rx(fv);

    MeshInfo info;
    info.num_vertices    = rx.get_num_vertices();
    info.num_edges       = rx.get_num_edges();
    info.num_faces       = rx.get_num_faces();
    info.is_edge_manifold = rx.is_edge_manifold();
    info.is_closed       = rx.is_closed();
    info.max_valence     = rx.get_input_max_valence();
    info.num_components  = rx.get_num_components();
    return info;
}

// ---------------------------------------------------------------------------
// pipeline_load_obj
// ---------------------------------------------------------------------------

MeshResult pipeline_load_obj(const std::string& path)
{
    ensure_init();

    std::vector<std::vector<float>>    verts;
    std::vector<std::vector<uint32_t>> faces;

    if (!import_obj(path, verts, faces)) {
        throw std::runtime_error("Failed to load OBJ file: " + path);
    }

    MeshResult result;
    result.num_vertices = static_cast<int>(verts.size());
    result.num_faces    = static_cast<int>(faces.size());

    result.vertices.resize(result.num_vertices * 3);
    for (int i = 0; i < result.num_vertices; ++i) {
        result.vertices[i * 3 + 0] = static_cast<double>(verts[i][0]);
        result.vertices[i * 3 + 1] = static_cast<double>(verts[i][1]);
        result.vertices[i * 3 + 2] = static_cast<double>(verts[i][2]);
    }

    result.faces.resize(result.num_faces * 3);
    for (int i = 0; i < result.num_faces; ++i) {
        result.faces[i * 3 + 0] = static_cast<int>(faces[i][0]);
        result.faces[i * 3 + 1] = static_cast<int>(faces[i][1]);
        result.faces[i * 3 + 2] = static_cast<int>(faces[i][2]);
    }

    return result;
}

// ---------------------------------------------------------------------------
// pipeline_vertex_normals
// ---------------------------------------------------------------------------

template <typename T, uint32_t blockThreads>
__global__ static void compute_vertex_normal_kernel(
    const Context          context,
    VertexAttribute<T>     coords,
    VertexAttribute<T>     normals)
{
    auto vn_lambda = [&](FaceHandle face_id, VertexIterator& fv) {
        vec3<T> c0 = coords.to_glm<3>(fv[0]);
        vec3<T> c1 = coords.to_glm<3>(fv[1]);
        vec3<T> c2 = coords.to_glm<3>(fv[2]);

        vec3<T> n = cross(c1 - c0, c2 - c0);

        vec3<T> d01 = c1 - c0, d12 = c2 - c1, d20 = c0 - c2;
        vec3<T> l(glm::dot(d01, d01), glm::dot(d12, d12), glm::dot(d20, d20));

        for (uint32_t v = 0; v < 3; ++v) {
            for (uint32_t i = 0; i < 3; ++i) {
                atomicAdd(&normals(fv[v], i), n[i] / (l[v] + l[(v + 2) % 3]));
            }
        }
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, vn_lambda);
}

AttributeResult pipeline_vertex_normals(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces)
{
    ensure_init();
    constexpr uint32_t blockThreads = 256;

    auto fv = flat_faces_to_fv(faces, num_faces);
    auto vv = flat_verts_to_vv(vertices, num_vertices);

    RXMeshStatic rx(fv);
    auto coords  = rx.add_vertex_attribute<float>(vv, "coordinates");
    auto normals = rx.add_vertex_attribute<float>("normals", 3, LOCATION_ALL);
    normals->reset(0, DEVICE);

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({Op::FV}, launch_box,
        (void*)compute_vertex_normal_kernel<float, blockThreads>);

    compute_vertex_normal_kernel<float, blockThreads>
        <<<launch_box.blocks, launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords, *normals);

    CUDA_ERROR(cudaDeviceSynchronize());
    normals->move(DEVICE, HOST);

    AttributeResult result;
    result.num_elements = static_cast<int>(rx.get_num_vertices());
    result.num_cols = 3;
    result.data.resize(result.num_elements * 3);

    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rx.map_to_global(vh);
        for (uint32_t i = 0; i < 3; ++i) {
            result.data[v_id * 3 + i] = static_cast<double>((*normals)(vh, i));
        }
    });
    return result;
}

// ---------------------------------------------------------------------------
// pipeline_smooth
// ---------------------------------------------------------------------------

template <typename T, uint32_t blockThreads>
__global__ static void laplacian_smooth_kernel(
    const Context          context,
    VertexAttribute<T>     pos_in,
    VertexAttribute<T>     pos_out,
    T                      lambda_val)
{
    auto smooth_lambda = [&](const VertexHandle& vh, const VertexIterator& iter) {
        const int k = iter.size();
        if (k == 0) {
            for (int i = 0; i < 3; ++i) pos_out(vh, i) = pos_in(vh, i);
            return;
        }
        T cx = T(0), cy = T(0), cz = T(0);
        for (int v = 0; v < k; ++v) {
            cx += pos_in(iter[v], 0);
            cy += pos_in(iter[v], 1);
            cz += pos_in(iter[v], 2);
        }
        cx /= T(k); cy /= T(k); cz /= T(k);

        pos_out(vh, 0) = pos_in(vh, 0) + lambda_val * (cx - pos_in(vh, 0));
        pos_out(vh, 1) = pos_in(vh, 1) + lambda_val * (cy - pos_in(vh, 1));
        pos_out(vh, 2) = pos_in(vh, 2) + lambda_val * (cz - pos_in(vh, 2));
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, smooth_lambda);
}

MeshResult pipeline_smooth(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int iterations, double lambda)
{
    ensure_init();
    constexpr uint32_t blockThreads = 256;

    auto fv = flat_faces_to_fv(faces, num_faces);
    auto vv = flat_verts_to_vv(vertices, num_vertices);

    RXMeshStatic rx(fv);
    rx.add_vertex_coordinates(vv);

    auto pos_a = rx.add_vertex_attribute<float>(vv, "pos_a");
    auto pos_b = rx.add_vertex_attribute<float>("pos_b", 3, DEVICE);
    float lambda_f = static_cast<float>(lambda);

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({Op::VV}, launch_box,
        (void*)laplacian_smooth_kernel<float, blockThreads>);

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            laplacian_smooth_kernel<float, blockThreads>
                <<<launch_box.blocks, launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(
                    rx.get_context(), *pos_a, *pos_b, lambda_f);
        } else {
            laplacian_smooth_kernel<float, blockThreads>
                <<<launch_box.blocks, launch_box.num_threads,
                   launch_box.smem_bytes_dyn>>>(
                    rx.get_context(), *pos_b, *pos_a, lambda_f);
        }
        CUDA_ERROR(cudaDeviceSynchronize());
    }

    auto& final_pos = (iterations % 2 == 0) ? pos_a : pos_b;
    final_pos->move(DEVICE, HOST);

    MeshResult result;
    result.num_vertices = static_cast<int>(rx.get_num_vertices());
    result.num_faces = num_faces;
    result.vertices.resize(result.num_vertices * 3);
    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rx.map_to_global(vh);
        for (uint32_t i = 0; i < 3; ++i)
            result.vertices[v_id * 3 + i] = static_cast<double>((*final_pos)(vh, i));
    });
    result.faces.resize(num_faces * 3);
    std::memcpy(result.faces.data(), faces, num_faces * 3 * sizeof(int));
    return result;
}

// ---------------------------------------------------------------------------
// pipeline_gaussian_curvature
// ---------------------------------------------------------------------------

template <typename T, uint32_t blockThreads>
__global__ static void compute_gaussian_curvature_kernel(
    const Context          context,
    VertexAttribute<T>     coords,
    VertexAttribute<T>     gcs,
    VertexAttribute<T>     amix)
{
    auto gc_lambda = [&](FaceHandle face_id, VertexIterator& fv) {
        const vec3<T> c0 = coords.to_glm<3>(fv[0]);
        const vec3<T> c1 = coords.to_glm<3>(fv[1]);
        const vec3<T> c2 = coords.to_glm<3>(fv[2]);

        vec3<T> d01 = c1 - c0, d12 = c2 - c1, d20 = c0 - c2;
        vec3<T> l(glm::dot(d01, d01), glm::dot(d12, d12), glm::dot(d20, d20));

        T s = glm::length(glm::cross(c1 - c0, c2 - c0));

        vec3<T> c(glm::dot(c1 - c0, c2 - c0),
                  glm::dot(c2 - c1, c0 - c1),
                  glm::dot(c0 - c2, c1 - c2));

        vec3<T> rads(atan2(s, c[0]), atan2(s, c[1]), atan2(s, c[2]));

        bool is_ob = false;
        for (int i = 0; i < 3; ++i) {
            if (rads[i] > glm::pi<T>() * T(0.5)) is_ob = true;
        }

        for (uint32_t v = 0; v < 3; ++v) {
            uint32_t v1 = (v + 1) % 3;
            uint32_t v2 = (v + 2) % 3;

            if (is_ob) {
                if (rads[v] > glm::pi<T>() * T(0.5)) {
                    atomicAdd(&amix(fv[v]), 0.25f * s);
                } else {
                    atomicAdd(&amix(fv[v]), 0.125f * s);
                }
            } else {
                atomicAdd(&amix(fv[v]),
                    0.125f * ((l[v2]) * (c[v1] / s) + (l[v]) * (c[v2] / s)));
            }
            atomicAdd(&gcs(fv[v]), -rads[v]);
        }
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, gc_lambda);
}

AttributeResult pipeline_gaussian_curvature(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces)
{
    ensure_init();
    constexpr uint32_t blockThreads = 256;

    auto fv = flat_faces_to_fv(faces, num_faces);
    auto vv = flat_verts_to_vv(vertices, num_vertices);

    RXMeshStatic rx(fv);
    auto coords = rx.add_vertex_attribute<float>(vv, "coordinates");
    auto v_gc   = rx.add_vertex_attribute<float>("v_gc", 1, LOCATION_ALL);
    auto v_amix = rx.add_vertex_attribute<float>("v_amix", 1, LOCATION_ALL);

    v_gc->reset(2.0f * glm::pi<float>(), DEVICE);
    v_amix->reset(0, DEVICE);

    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({Op::FV}, launch_box,
        (void*)compute_gaussian_curvature_kernel<float, blockThreads>);

    compute_gaussian_curvature_kernel<float, blockThreads>
        <<<launch_box.blocks, launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(rx.get_context(), *coords, *v_gc, *v_amix);

    // Normalize by mixed area
    rx.for_each_vertex(DEVICE,
        [v_gc = *v_gc, v_amix = *v_amix] __device__(const VertexHandle vh) {
            v_gc(vh, 0) = v_gc(vh, 0) / v_amix(vh, 0);
        });

    CUDA_ERROR(cudaDeviceSynchronize());
    v_gc->move(DEVICE, HOST);

    AttributeResult result;
    result.num_elements = static_cast<int>(rx.get_num_vertices());
    result.num_cols = 1;
    result.data.resize(result.num_elements);

    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rx.map_to_global(vh);
        result.data[v_id] = static_cast<double>((*v_gc)(vh, 0));
    });
    return result;
}

// ---------------------------------------------------------------------------
// pipeline_filter (bilateral mesh denoising)
// ---------------------------------------------------------------------------

// Simplified bilateral filter kernel (1-ring only, no multi-ring expansion)
template <typename T, uint32_t blockThreads>
__global__ static void bilateral_filter_kernel(
    const Context          context,
    VertexAttribute<T>     input_coords,
    VertexAttribute<T>     filtered_coords,
    VertexAttribute<T>     vertex_normals)
{
    auto filter_lambda = [&](const VertexHandle& vh, const VertexIterator& iter) {
        const int k = iter.size();
        if (k == 0) {
            filtered_coords(vh, 0) = input_coords(vh, 0);
            filtered_coords(vh, 1) = input_coords(vh, 1);
            filtered_coords(vh, 2) = input_coords(vh, 2);
            return;
        }

        vec3<T> v(input_coords(vh, 0), input_coords(vh, 1), input_coords(vh, 2));
        vec3<T> n(vertex_normals(vh, 0), vertex_normals(vh, 1), vertex_normals(vh, 2));
        n = glm::normalize(n);

        // Compute sigma_c as min edge length squared
        T sigma_c_sq = T(1e10);
        for (int i = 0; i < k; ++i) {
            vec3<T> q(input_coords(iter[i], 0), input_coords(iter[i], 1),
                      input_coords(iter[i], 2));
            vec3<T> d = q - v;
            T len_sq = glm::dot(d, d);
            if (len_sq < sigma_c_sq) sigma_c_sq = len_sq;
        }

        // Compute sigma_s from height variance
        T sum_h = T(0), sum_h_sq = T(0);
        for (int i = 0; i < k; ++i) {
            vec3<T> q(input_coords(iter[i], 0), input_coords(iter[i], 1),
                      input_coords(iter[i], 2));
            T h = glm::dot(q - v, n);
            T ah = (h < T(0)) ? -h : h;
            sum_h += ah;
            sum_h_sq += ah * ah;
        }
        T sigma_s_sq = (sum_h_sq / T(k)) - (sum_h * sum_h) / (T(k) * T(k));
        if (sigma_s_sq < T(1e-20)) sigma_s_sq += T(1e-20);

        // Apply bilateral filter
        T sum = T(0), normalizer = T(0);
        for (int i = 0; i < k; ++i) {
            vec3<T> q(input_coords(iter[i], 0), input_coords(iter[i], 1),
                      input_coords(iter[i], 2));
            vec3<T> d = q - v;
            T t = glm::length(d);
            T h = glm::dot(d, n);
            T wc = exp(T(-0.5) * t * t / sigma_c_sq);
            T ws = exp(T(-0.5) * h * h / sigma_s_sq);
            sum += wc * ws * h;
            normalizer += wc * ws;
        }

        vec3<T> result = v + n * (sum / normalizer);
        filtered_coords(vh, 0) = result[0];
        filtered_coords(vh, 1) = result[1];
        filtered_coords(vh, 2) = result[2];
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, filter_lambda);
}

MeshResult pipeline_filter(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int iterations)
{
    ensure_init();
    constexpr uint32_t blockThreads = 256;

    auto fv = flat_faces_to_fv(faces, num_faces);
    auto vv = flat_verts_to_vv(vertices, num_vertices);

    RXMeshStatic rx(fv);
    rx.add_vertex_coordinates(vv);

    auto coords = rx.add_vertex_attribute<float>(vv, "coords");
    auto filtered = rx.add_vertex_attribute<float>("filtered", 3, LOCATION_ALL);
    auto vnormals = rx.add_vertex_attribute<float>("vn", 3, DEVICE);

    LaunchBox<blockThreads> vn_lb;
    rx.prepare_launch_box({Op::FV}, vn_lb,
        (void*)compute_vertex_normal_kernel<float, blockThreads>);

    LaunchBox<blockThreads> filter_lb;
    rx.prepare_launch_box({Op::VV}, filter_lb,
        (void*)bilateral_filter_kernel<float, blockThreads>);

    VertexAttribute<float>* double_buffer[2] = {coords.get(), filtered.get()};
    uint32_t d = 0;

    for (int itr = 0; itr < iterations; ++itr) {
        vnormals->reset(0, DEVICE);

        compute_vertex_normal_kernel<float, blockThreads>
            <<<vn_lb.blocks, blockThreads, vn_lb.smem_bytes_dyn>>>(
                rx.get_context(), *double_buffer[d], *vnormals);

        bilateral_filter_kernel<float, blockThreads>
            <<<filter_lb.blocks, blockThreads, filter_lb.smem_bytes_dyn>>>(
                rx.get_context(), *double_buffer[d], *double_buffer[!d], *vnormals);

        d = !d;
        CUDA_ERROR(cudaDeviceSynchronize());
    }

    coords->copy_from(*double_buffer[d], DEVICE, HOST);

    MeshResult result;
    result.num_vertices = static_cast<int>(rx.get_num_vertices());
    result.num_faces = num_faces;
    result.vertices.resize(result.num_vertices * 3);
    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rx.map_to_global(vh);
        for (uint32_t i = 0; i < 3; ++i)
            result.vertices[v_id * 3 + i] = static_cast<double>((*coords)(vh, i));
    });
    result.faces.resize(num_faces * 3);
    std::memcpy(result.faces.data(), faces, num_faces * 3 * sizeof(int));
    return result;
}

// ---------------------------------------------------------------------------
// pipeline_mcf (Mean Curvature Flow)
// ---------------------------------------------------------------------------

// MCF kernel: set up right-hand side B = M * X
template <typename T, uint32_t blockThreads>
__global__ static void mcf_B_setup_kernel(
    const Context          context,
    VertexAttribute<T>     coords,
    DenseMatrix<T>         B,
    bool                   uniform)
{
    auto compute = [&](const VertexHandle& vh, const VertexIterator& iter) {
        T sum_w = T(0);
        for (int i = 0; i < static_cast<int>(iter.size()); ++i) {
            T w = T(1); // uniform weight
            if (!uniform) {
                // Cotangent weight would go here but uniform is simpler
                w = T(1);
            }
            sum_w += w;
        }
        for (int d = 0; d < 3; ++d) {
            B(vh, d) = sum_w * coords(vh, d);
        }
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, compute);
}

// MCF kernel: set up matrix A = M - dt * L
template <typename T, uint32_t blockThreads>
__global__ static void mcf_A_setup_kernel(
    const Context          context,
    VertexAttribute<T>     coords,
    SparseMatrix<T>        A,
    bool                   uniform,
    T                      dt)
{
    auto compute = [&](const VertexHandle& vh, const VertexIterator& iter) {
        T sum_w = T(0);
        for (int i = 0; i < static_cast<int>(iter.size()); ++i) {
            T w = T(1); // uniform weight
            sum_w += w;
            A(vh, iter[i]) = dt * w;  // off-diagonal: dt * w
        }
        A(vh, vh) = sum_w - dt * (-sum_w);  // diagonal: M_ii + dt * sum_w
        A(vh, vh) = sum_w + dt * sum_w;
    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, compute);
}

MeshResult pipeline_mcf(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    double time_step, bool use_uniform_laplace)
{
    ensure_init();
    constexpr uint32_t blockThreads = 256;

    // Use temp OBJ for construction (MCF needs file-based init for patches)
    auto in_path = write_temp_obj(vertices, num_vertices, faces, num_faces, "pyrxmesh_mcf_in");
    RXMeshStatic rx(in_path, "", 256);

    auto coords = rx.get_input_vertex_coordinates();
    uint32_t nv = rx.get_num_vertices();

    SparseMatrix<float> A_mat(rx);
    DenseMatrix<float>  B_mat(rx, nv, 3, LOCATION_ALL);
    DenseMatrix<float>  X_mat = *coords->to_matrix();

    float dt = static_cast<float>(time_step);

    // Set up B and A
    rx.run_kernel<blockThreads>({Op::VV},
        mcf_B_setup_kernel<float, blockThreads>, *coords, B_mat, use_uniform_laplace);

    rx.run_kernel<blockThreads>({Op::VV},
        mcf_A_setup_kernel<float, blockThreads>, *coords, A_mat, use_uniform_laplace, dt);

    // Solve A * X = B via Cholesky
    CholeskySolver solver(&A_mat, PermuteMethod::NSTDIS);
    solver.permute_alloc();
    solver.permute(rx);
    solver.premute_value_ptr();
    solver.analyze_pattern();
    solver.post_analyze_alloc();
    solver.factorize();
    solver.solve(B_mat, X_mat);

    X_mat.move(DEVICE, HOST);
    coords->from_matrix(&X_mat);
    coords->move(DEVICE, HOST);

    auto result = extract_static_mesh(rx, *coords);

    B_mat.release();
    X_mat.release();
    A_mat.release();

    std::filesystem::remove(in_path);
    return result;
}

// =========================================================================
// Persistent mesh handle implementation
// =========================================================================

struct MeshHandleImpl {
    RXMeshStatic*                            rx;
    std::shared_ptr<VertexAttribute<float>>  coords;
    std::vector<std::vector<uint32_t>>       fv;       // original faces
    int                                      num_verts;
    int                                      num_faces;
    std::string                              tmp_path; // if created from arrays

    // Cached launch boxes
    bool fv_lb_ready = false;
    bool vv_lb_ready = false;
};

MeshHandle mesh_create(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces)
{
    ensure_init();

    auto h = new MeshHandleImpl();
    h->num_verts = num_vertices;
    h->num_faces = num_faces;

    h->fv = flat_faces_to_fv(faces, num_faces);
    auto vv = flat_verts_to_vv(vertices, num_vertices);

    h->rx = new RXMeshStatic(h->fv);
    h->rx->add_vertex_coordinates(vv);
    h->coords = h->rx->add_vertex_attribute<float>(vv, "coordinates");

    return h;
}

MeshHandle mesh_create_from_obj(const std::string& path)
{
    ensure_init();

    auto h = new MeshHandleImpl();

    std::vector<std::vector<float>> verts;
    if (!import_obj(path, verts, h->fv)) {
        delete h;
        throw std::runtime_error("Failed to load OBJ: " + path);
    }

    h->num_verts = static_cast<int>(verts.size());
    h->num_faces = static_cast<int>(h->fv.size());

    h->rx = new RXMeshStatic(h->fv);
    h->rx->add_vertex_coordinates(verts);
    h->coords = h->rx->add_vertex_attribute<float>(verts, "coordinates");

    return h;
}

void mesh_destroy(MeshHandle h)
{
    if (h) {
        h->coords.reset();
        delete h->rx;
        if (!h->tmp_path.empty()) {
            std::filesystem::remove(h->tmp_path);
        }
        delete h;
    }
}

int mesh_get_num_vertices(MeshHandle h) { return h->rx->get_num_vertices(); }
int mesh_get_num_faces(MeshHandle h) { return h->rx->get_num_faces(); }

void mesh_get_vertices(MeshHandle h, double* out)
{
    h->coords->move(DEVICE, HOST);
    h->rx->for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = h->rx->map_to_global(vh);
        for (uint32_t i = 0; i < 3; ++i)
            out[v_id * 3 + i] = static_cast<double>((*h->coords)(vh, i));
    });
}

MeshInfo mesh_get_info(MeshHandle h)
{
    MeshInfo info;
    info.num_vertices     = h->rx->get_num_vertices();
    info.num_edges        = h->rx->get_num_edges();
    info.num_faces        = h->rx->get_num_faces();
    info.is_edge_manifold = h->rx->is_edge_manifold();
    info.is_closed        = h->rx->is_closed();
    info.max_valence      = h->rx->get_input_max_valence();
    info.num_components   = h->rx->get_num_components();
    return info;
}

AttributeResult mesh_vertex_normals(MeshHandle h)
{
    constexpr uint32_t blockThreads = 256;

    auto normals = h->rx->add_vertex_attribute<float>("_normals", 3, LOCATION_ALL);
    normals->reset(0, DEVICE);

    LaunchBox<blockThreads> lb;
    h->rx->prepare_launch_box({Op::FV}, lb,
        (void*)compute_vertex_normal_kernel<float, blockThreads>);

    compute_vertex_normal_kernel<float, blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            h->rx->get_context(), *h->coords, *normals);
    CUDA_ERROR(cudaDeviceSynchronize());

    normals->move(DEVICE, HOST);

    AttributeResult result;
    result.num_elements = h->rx->get_num_vertices();
    result.num_cols = 3;
    result.data.resize(result.num_elements * 3);

    h->rx->for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = h->rx->map_to_global(vh);
        for (uint32_t i = 0; i < 3; ++i)
            result.data[v_id * 3 + i] = static_cast<double>((*normals)(vh, i));
    });

    h->rx->remove_attribute("_normals");
    return result;
}

AttributeResult mesh_gaussian_curvature(MeshHandle h)
{
    constexpr uint32_t blockThreads = 256;

    auto v_gc   = h->rx->add_vertex_attribute<float>("_gc", 1, LOCATION_ALL);
    auto v_amix = h->rx->add_vertex_attribute<float>("_amix", 1, LOCATION_ALL);

    v_gc->reset(2.0f * glm::pi<float>(), DEVICE);
    v_amix->reset(0, DEVICE);

    LaunchBox<blockThreads> lb;
    h->rx->prepare_launch_box({Op::FV}, lb,
        (void*)compute_gaussian_curvature_kernel<float, blockThreads>);

    compute_gaussian_curvature_kernel<float, blockThreads>
        <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
            h->rx->get_context(), *h->coords, *v_gc, *v_amix);

    h->rx->for_each_vertex(DEVICE,
        [gc = *v_gc, amix = *v_amix] __device__(const VertexHandle vh) {
            gc(vh, 0) = gc(vh, 0) / amix(vh, 0);
        });
    CUDA_ERROR(cudaDeviceSynchronize());

    v_gc->move(DEVICE, HOST);

    AttributeResult result;
    result.num_elements = h->rx->get_num_vertices();
    result.num_cols = 1;
    result.data.resize(result.num_elements);

    h->rx->for_each_vertex(HOST, [&](const VertexHandle& vh) {
        result.data[h->rx->map_to_global(vh)] = static_cast<double>((*v_gc)(vh, 0));
    });

    h->rx->remove_attribute("_gc");
    h->rx->remove_attribute("_amix");
    return result;
}

AttributeResult mesh_geodesic(MeshHandle h, const int* seed_vertices, int num_seeds)
{
    // Delegate to the stateless version — geodesic needs toplesets which are
    // seed-dependent, so we can't cache much. But we avoid file I/O.
    // TODO: cache toplesets for repeated calls with same seeds.

    constexpr uint32_t blockThreads = 256;

    std::vector<uint32_t> seeds(num_seeds);
    for (int i = 0; i < num_seeds; ++i)
        seeds[i] = static_cast<uint32_t>(seed_vertices[i]);

    // BFS toplesets on CPU using face-vertex adjacency
    uint32_t nv = h->rx->get_num_vertices();
    std::vector<std::vector<uint32_t>> adj(nv);
    for (auto& face : h->fv) {
        for (size_t i = 0; i < face.size(); ++i)
            for (size_t j = i + 1; j < face.size(); ++j) {
                adj[face[i]].push_back(face[j]);
                adj[face[j]].push_back(face[i]);
            }
    }
    for (auto& n : adj) {
        std::sort(n.begin(), n.end());
        n.erase(std::unique(n.begin(), n.end()), n.end());
    }

    std::vector<uint32_t> toplesets(nv, UINT32_MAX);
    std::vector<uint32_t> sorted_index;
    std::vector<uint32_t> limits;
    sorted_index.reserve(nv);

    uint32_t level = 0;
    for (uint32_t s : seeds) {
        if (toplesets[s] == UINT32_MAX) {
            toplesets[s] = level;
            sorted_index.push_back(s);
        }
    }
    limits.push_back(0);
    for (size_t i = 0; i < sorted_index.size(); ++i) {
        uint32_t v = sorted_index[i];
        if (toplesets[v] > level) { level++; limits.push_back(i); }
        for (uint32_t nb : adj[v]) {
            if (toplesets[nb] == UINT32_MAX) {
                toplesets[nb] = toplesets[v] + 1;
                sorted_index.push_back(nb);
            }
        }
    }
    limits.push_back(sorted_index.size());

    auto d_toplesets = h->rx->add_vertex_attribute<uint32_t>(toplesets, "_topleset");
    auto geo_dist = h->rx->add_vertex_attribute<float>("_geo", 1);
    geo_dist->reset(std::numeric_limits<float>::infinity(), HOST);
    h->rx->for_each_vertex(HOST, [&](const VertexHandle vh) {
        uint32_t v_id = h->rx->map_to_global(vh);
        for (uint32_t s : seeds)
            if (s == v_id) { (*geo_dist)(vh) = 0; break; }
    });
    geo_dist->move(HOST, DEVICE);

    auto geo_dist_2 = h->rx->add_vertex_attribute<float>("_geo2", 1, DEVICE);
    geo_dist_2->copy_from(*geo_dist, DEVICE, DEVICE);

    uint32_t *d_error = nullptr;
    CUDA_ERROR(cudaMalloc((void**)&d_error, sizeof(uint32_t)));

    LaunchBox<blockThreads> lb;
    h->rx->prepare_launch_box({Op::VV}, lb,
        (void*)relax_ptp_rxmesh<float, blockThreads>, true);

    VertexAttribute<float>* db[2] = {geo_dist.get(), geo_dist_2.get()};
    uint32_t d = 0, ii = 1, jj = 2, iter = 0;
    uint32_t max_iter = 2 * limits.size();

    while (ii < jj && iter < max_iter) {
        iter++;
        if (ii < (jj / 2)) ii = jj / 2;

        relax_ptp_rxmesh<float, blockThreads>
            <<<lb.blocks, blockThreads, lb.smem_bytes_dyn>>>(
                h->rx->get_context(), *h->coords, *db[!d], *db[d],
                *d_toplesets, ii, jj, d_error,
                std::numeric_limits<float>::infinity(), float(1e-3));

        uint32_t h_error = 0;
        CUDA_ERROR(cudaMemcpy(&h_error, d_error, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemset(d_error, 0, sizeof(uint32_t)));

        uint32_t n_cond = limits[ii + 1] - limits[ii];
        if (n_cond == h_error) ii++;
        if (jj < limits.size() - 1) jj++;
        d = !d;
    }
    CUDA_ERROR(cudaDeviceSynchronize());

    geo_dist->copy_from(*db[d], DEVICE, HOST);

    AttributeResult result;
    result.num_elements = nv;
    result.num_cols = 1;
    result.data.resize(nv);
    h->rx->for_each_vertex(HOST, [&](const VertexHandle& vh) {
        result.data[h->rx->map_to_global(vh)] = static_cast<double>((*geo_dist)(vh, 0));
    });

    CUDA_ERROR(cudaFree(d_error));
    h->rx->remove_attribute("_topleset");
    h->rx->remove_attribute("_geo");
    h->rx->remove_attribute("_geo2");
    return result;
}

MeshResult mesh_smooth(MeshHandle h, int iterations, double lambda)
{
    constexpr uint32_t blockThreads = 256;
    float lambda_f = static_cast<float>(lambda);

    auto pos_b = h->rx->add_vertex_attribute<float>("_pos_b", 3, DEVICE);

    LaunchBox<blockThreads> lb;
    h->rx->prepare_launch_box({Op::VV}, lb,
        (void*)laplacian_smooth_kernel<float, blockThreads>);

    auto* a = h->coords.get();
    auto* b = pos_b.get();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            laplacian_smooth_kernel<float, blockThreads>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    h->rx->get_context(), *a, *b, lambda_f);
        } else {
            laplacian_smooth_kernel<float, blockThreads>
                <<<lb.blocks, lb.num_threads, lb.smem_bytes_dyn>>>(
                    h->rx->get_context(), *b, *a, lambda_f);
        }
        CUDA_ERROR(cudaDeviceSynchronize());
    }

    // If odd iterations, result is in pos_b — copy back to coords
    if (iterations > 0 && iterations % 2 != 0) {
        h->coords->copy_from(*pos_b, DEVICE, DEVICE);
    }

    h->coords->move(DEVICE, HOST);

    MeshResult result;
    result.num_vertices = h->rx->get_num_vertices();
    result.num_faces = h->num_faces;
    result.vertices.resize(result.num_vertices * 3);
    h->rx->for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = h->rx->map_to_global(vh);
        for (uint32_t i = 0; i < 3; ++i)
            result.vertices[v_id * 3 + i] = static_cast<double>((*h->coords)(vh, i));
    });

    result.faces.resize(h->num_faces * 3);
    for (int i = 0; i < h->num_faces; ++i)
        for (int j = 0; j < 3; ++j)
            result.faces[i * 3 + j] = static_cast<int>(h->fv[i][j]);

    h->rx->remove_attribute("_pos_b");
    return result;
}

MeshResult mesh_filter(MeshHandle h, int iterations)
{
    constexpr uint32_t blockThreads = 256;

    auto filtered = h->rx->add_vertex_attribute<float>("_filtered", 3, LOCATION_ALL);
    auto vnormals = h->rx->add_vertex_attribute<float>("_vn", 3, DEVICE);

    LaunchBox<blockThreads> vn_lb, filter_lb;
    h->rx->prepare_launch_box({Op::FV}, vn_lb,
        (void*)compute_vertex_normal_kernel<float, blockThreads>);
    h->rx->prepare_launch_box({Op::VV}, filter_lb,
        (void*)bilateral_filter_kernel<float, blockThreads>);

    VertexAttribute<float>* db[2] = {h->coords.get(), filtered.get()};
    uint32_t d = 0;

    for (int itr = 0; itr < iterations; ++itr) {
        vnormals->reset(0, DEVICE);
        compute_vertex_normal_kernel<float, blockThreads>
            <<<vn_lb.blocks, blockThreads, vn_lb.smem_bytes_dyn>>>(
                h->rx->get_context(), *db[d], *vnormals);
        bilateral_filter_kernel<float, blockThreads>
            <<<filter_lb.blocks, blockThreads, filter_lb.smem_bytes_dyn>>>(
                h->rx->get_context(), *db[d], *db[!d], *vnormals);
        d = !d;
        CUDA_ERROR(cudaDeviceSynchronize());
    }

    h->coords->copy_from(*db[d], DEVICE, HOST);

    MeshResult result;
    result.num_vertices = h->rx->get_num_vertices();
    result.num_faces = h->num_faces;
    result.vertices.resize(result.num_vertices * 3);
    h->rx->for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = h->rx->map_to_global(vh);
        for (uint32_t i = 0; i < 3; ++i)
            result.vertices[v_id * 3 + i] = static_cast<double>((*h->coords)(vh, i));
    });
    result.faces.resize(h->num_faces * 3);
    for (int i = 0; i < h->num_faces; ++i)
        for (int j = 0; j < 3; ++j)
            result.faces[i * 3 + j] = static_cast<int>(h->fv[i][j]);

    h->rx->remove_attribute("_filtered");
    h->rx->remove_attribute("_vn");
    return result;
}

// Dynamic mesh operations (QSlim, Remesh, SEC, Delaunay) are implemented
// in separate .cu files (op_qslim.cu, op_sec.cu, op_delaunay.cu, op_remesh.cu)
// because each app header expects its own global Arg struct.

