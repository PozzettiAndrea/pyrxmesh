// Geodesic distance computation wrapper.
// Reimplements the topeset BFS (originally OpenMesh) in plain C++,
// then uses RXMesh's GPU kernel for the parallel distance computation.

#include "pipeline.h"
#include <filesystem>
#include <chrono>
#include <cstdio>
#include <queue>
#include <limits>

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/attribute.h"
#include "rxmesh/query.h"
#include "rxmesh/util/import_obj.h"

#include "glm_compat.h"

// Include the geodesic kernel directly
#include "Geodesic/geodesic_kernel.cuh"

using namespace rxmesh;

// ---------------------------------------------------------------------------
// BFS topeset computation (replaces OpenMesh dependency)
// ---------------------------------------------------------------------------

static void compute_toplesets_from_fv(
    const std::vector<std::vector<uint32_t>>& fv,
    uint32_t num_vertices,
    const std::vector<uint32_t>& seeds,
    std::vector<uint32_t>& toplesets,
    std::vector<uint32_t>& sorted_index,
    std::vector<uint32_t>& limits)
{
    // Build adjacency list from face-vertex
    std::vector<std::vector<uint32_t>> adj(num_vertices);
    for (auto& face : fv) {
        for (size_t i = 0; i < face.size(); ++i) {
            for (size_t j = i + 1; j < face.size(); ++j) {
                adj[face[i]].push_back(face[j]);
                adj[face[j]].push_back(face[i]);
            }
        }
    }
    // Deduplicate adjacency
    for (auto& neighbors : adj) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    // BFS from seeds
    toplesets.assign(num_vertices, UINT32_MAX);
    sorted_index.clear();
    sorted_index.reserve(num_vertices);
    limits.clear();

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
        if (toplesets[v] > level) {
            level++;
            limits.push_back(static_cast<uint32_t>(i));
        }
        for (uint32_t neighbor : adj[v]) {
            if (toplesets[neighbor] == UINT32_MAX) {
                toplesets[neighbor] = toplesets[v] + 1;
                sorted_index.push_back(neighbor);
            }
        }
    }
    limits.push_back(static_cast<uint32_t>(sorted_index.size()));
}

// ---------------------------------------------------------------------------
// pipeline_geodesic
// ---------------------------------------------------------------------------

AttributeResult pipeline_geodesic(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    const int* seed_vertices, int num_seeds,
    bool verbose)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    constexpr uint32_t blockThreads = 256;

    // Build face-vertex lists
    std::vector<std::vector<uint32_t>> fv(num_faces);
    for (int i = 0; i < num_faces; ++i) {
        fv[i] = {
            static_cast<uint32_t>(faces[i*3+0]),
            static_cast<uint32_t>(faces[i*3+1]),
            static_cast<uint32_t>(faces[i*3+2])
        };
    }

    std::vector<std::vector<float>> vv(num_vertices);
    for (int i = 0; i < num_vertices; ++i) {
        vv[i] = {
            static_cast<float>(vertices[i*3+0]),
            static_cast<float>(vertices[i*3+1]),
            static_cast<float>(vertices[i*3+2])
        };
    }

    // Compute toplesets via BFS (replaces OpenMesh dependency)
    std::vector<uint32_t> seeds(num_seeds);
    for (int i = 0; i < num_seeds; ++i) {
        seeds[i] = static_cast<uint32_t>(seed_vertices[i]);
    }

    std::vector<uint32_t> toplesets, sorted_index, limits;
    compute_toplesets_from_fv(fv, num_vertices, seeds, toplesets, sorted_index, limits);

    if (limits.back() != static_cast<uint32_t>(num_vertices)) {
        throw std::runtime_error(
            "Geodesic: could not compute toplesets for all vertices. "
            "Mesh may not be manifold or may contain unreachable vertices.");
    }

    // Create RXMesh
    RXMeshStatic rx(fv);
    auto coords = rx.add_vertex_attribute<float>(vv, "coordinates");
    auto d_toplesets = rx.add_vertex_attribute<uint32_t>(toplesets, "topleset");

    // Initialize geodesic distances
    auto geo_dist = rx.add_vertex_attribute<float>("geo", 1);
    geo_dist->reset(std::numeric_limits<float>::infinity(), HOST);
    rx.for_each_vertex(HOST, [&](const VertexHandle vh) {
        uint32_t v_id = rx.map_to_global(vh);
        for (uint32_t s : seeds) {
            if (s == v_id) {
                (*geo_dist)(vh) = 0;
                break;
            }
        }
    });
    geo_dist->move(HOST, DEVICE);

    auto geo_dist_2 = rx.add_vertex_attribute<float>("geo2", 1, DEVICE);
    geo_dist_2->copy_from(*geo_dist, DEVICE, DEVICE);

    // Error tracking
    uint32_t *d_error = nullptr, h_error = 0;
    CUDA_ERROR(cudaMalloc((void**)&d_error, sizeof(uint32_t)));

    // Prepare launch box
    LaunchBox<blockThreads> launch_box;
    rx.prepare_launch_box({Op::VV}, launch_box,
        (void*)relax_ptp_rxmesh<float, blockThreads>, true);

    // Double buffer
    VertexAttribute<float>* double_buffer[2] = {geo_dist.get(), geo_dist_2.get()};

    // Run geodesic computation
    uint32_t d = 0;
    uint32_t i = 1, j = 2;
    uint32_t iter = 0;
    uint32_t max_iter = 2 * limits.size();

    while (i < j && iter < max_iter) {
        iter++;
        if (i < (j / 2)) i = j / 2;

        relax_ptp_rxmesh<float, blockThreads>
            <<<launch_box.blocks, blockThreads, launch_box.smem_bytes_dyn>>>(
                rx.get_context(),
                *coords,
                *double_buffer[!d],
                *double_buffer[d],
                *d_toplesets,
                i, j,
                d_error,
                std::numeric_limits<float>::infinity(),
                float(1e-3));

        CUDA_ERROR(cudaMemcpy(&h_error, d_error, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemset(d_error, 0, sizeof(uint32_t)));

        const uint32_t n_cond = limits[i + 1] - limits[i];
        if (n_cond == h_error) i++;
        if (j < limits.size() - 1) j++;

        d = !d;
    }

    CUDA_ERROR(cudaDeviceSynchronize());

    // Copy result
    geo_dist->copy_from(*double_buffer[d], DEVICE, HOST);

    AttributeResult result;
    result.num_elements = static_cast<int>(rx.get_num_vertices());
    result.num_cols = 1;
    result.data.resize(result.num_elements);

    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rx.map_to_global(vh);
        result.data[v_id] = static_cast<double>((*geo_dist)(vh, 0));
    });

    CUDA_ERROR(cudaFree(d_error));
    if (verbose) {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "[pyrxmesh] geodesic: %d verts, %d seeds, %.1f ms\n",
                num_vertices, num_seeds, ms);
    }
    return result;
}
