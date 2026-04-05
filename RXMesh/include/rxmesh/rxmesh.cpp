#include <assert.h>
#include <omp.h>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <unordered_set>

#include "patcher/patcher.h"
#include "rxmesh/gpu_build_topology.cuh"
#include "rxmesh/gpu_patch_build.cuh"
#include "rxmesh/context.h"
#include "rxmesh/patch_scheduler.h"
#include "rxmesh/rxmesh.h"
#include "rxmesh/util/bitmask_util.h"
#include "rxmesh/util/util.h"

namespace rxmesh {
RXMesh::RXMesh(uint32_t patch_size)
    : m_num_edges(0),
      m_num_faces(0),
      m_num_vertices(0),
      m_max_edge_capacity(0),
      m_max_face_capacity(0),
      m_max_vertex_capacity(0),
      m_input_max_valence(0),
      m_input_max_edge_incident_faces(0),
      m_input_max_face_adjacent_faces(0),
      m_is_input_edge_manifold(true),
      m_is_input_closed(true),
      m_num_patches(0),
      m_max_num_patches(0),
      m_patch_size(patch_size),
      m_max_capacity_lp_v(0),
      m_max_capacity_lp_e(0),
      m_max_capacity_lp_f(0),
      m_max_vertices_per_patch(0),
      m_max_edges_per_patch(0),
      m_max_faces_per_patch(0),
      m_h_vertex_prefix(nullptr),
      m_h_edge_prefix(nullptr),
      m_h_face_prefix(nullptr),
      m_d_vertex_prefix(nullptr),
      m_d_edge_prefix(nullptr),
      m_d_face_prefix(nullptr),
      m_d_patches_info(nullptr),
      m_h_patches_info(nullptr),
      m_capacity_factor(0.f),
      m_lp_hashtable_load_factor(0.f),
      m_patch_alloc_factor(0.f),
      m_topo_memory_mega_bytes(0.0),
      m_num_colors(0),
      m_h_v_handles(nullptr),
      m_h_e_handles(nullptr),
      m_h_f_handles(nullptr),
      m_d_v_handles(nullptr),
      m_d_e_handles(nullptr),
      m_d_f_handles(nullptr)

{
}

void RXMesh::init_flat(const uint32_t* flat_fv, uint32_t num_faces,
                       const std::string patcher_file,
                       const float capacity_factor,
                       const float patch_alloc_factor,
                       const float lp_hashtable_load_factor)
{
    // Set flat faces BEFORE init — everything downstream uses m_flat_faces
    m_flat_faces.assign(flat_fv, flat_fv + num_faces * 3);

    // Build minimal fv stub (init checks fv.empty() and build() signature needs it)
    // The actual data comes from m_flat_faces — fv is just for function signatures.
    std::vector<std::vector<uint32_t>> fv(num_faces);
    for (uint32_t f = 0; f < num_faces; ++f)
        fv[f] = {flat_fv[f*3], flat_fv[f*3+1], flat_fv[f*3+2]};

    init(fv, patcher_file, capacity_factor, patch_alloc_factor,
         lp_hashtable_load_factor);
}

void RXMesh::init(const std::vector<std::vector<uint32_t>>& fv,
                  const std::string                         patcher_file,
                  const float                               capacity_factor,
                  const float                               patch_alloc_factor,
                  const float lp_hashtable_load_factor)
{
    m_topo_memory_mega_bytes   = 0;
    m_capacity_factor          = capacity_factor;
    m_lp_hashtable_load_factor = lp_hashtable_load_factor;
    m_patch_alloc_factor       = patch_alloc_factor;

    // Build everything from scratch including patches
    if (fv.empty()) {
        RXMESH_ERROR(
            "RXMesh::init input fv is empty. Can not build RXMesh properly");
    }
    if (m_capacity_factor < 1.0) {
        RXMESH_ERROR("RXMesh::init capacity factor should be at least one");
    }
    if (m_patch_alloc_factor < 1.0) {
        RXMESH_ERROR(
            "RXMesh::init patch allocation factor should be at least one");
    }
    if (m_lp_hashtable_load_factor > 1.0) {
        RXMESH_ERROR(
            "RXMesh::init hashtable load factor should be less than 1");
    }

    // Pre-build flat face array (avoids fv→flat→fv round-trip in build_supporting_structures)
    if (m_flat_faces.empty()) {
        m_flat_faces.resize(fv.size() * 3);
        for (uint32_t f = 0; f < fv.size(); ++f) {
            m_flat_faces[f * 3 + 0] = fv[f][0];
            m_flat_faces[f * 3 + 1] = fv[f][1];
            m_flat_faces[f * 3 + 2] = fv[f][2];
        }
    }

    // Enable CUDA memory pool caching for async allocations.
    // This makes cudaMallocAsync reuse freed memory instead of
    // returning it to the OS, dramatically reducing allocation overhead.
    {
        cudaMemPool_t pool;
        CUDA_ERROR(cudaDeviceGetDefaultMemPool(&pool, 0));
        uint64_t threshold = UINT64_MAX;
        CUDA_ERROR(cudaMemPoolSetAttribute(
            pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    }

    m_timers.add("LPHashTable");
    m_timers.add("ht.insert");
    m_timers.add("lower_bound");
    m_timers.add("bitmask");
    m_timers.add("buildHT");
    m_timers.add("cudaMalloc");
    m_timers.add("malloc");
    m_timers.add("hashtable.move");
    m_timers.add("cudaMemcpy");
    m_timers.add("bitmask.cudaMemcpy");

    // 1)
    m_timers.add("build");
    m_timers.start("build");
    build(fv, patcher_file);
    m_timers.stop("build");

    // 2) populate_patch_stash — GPU if device arrays available, CPU otherwise
    m_timers.add("populate_patch_stash");
    m_timers.start("populate_patch_stash");
    if (m_retained_thr.device_arrays_valid && m_d_face_patch_bd &&
        m_d_edge_patch_bd && m_d_vertex_patch_bd) {
        // GPU stash using retained device arrays
        fprintf(stderr, "[init] GPU populate_patch_stash...\n");
        size_t stash_bytes = PatchStash::stash_size * sizeof(uint32_t);
        uint8_t* d_stash_tmp;
        CUDA_ERROR(cudaMalloc(&d_stash_tmp, get_num_patches() * stash_bytes));
        CUDA_ERROR(cudaMemset(d_stash_tmp, 0xFF, get_num_patches() * stash_bytes));

        gpu_build_stash(m_retained_thr,
                        m_d_face_patch_bd, m_d_edge_patch_bd, m_d_vertex_patch_bd,
                        get_num_patches(), d_stash_tmp, stash_bytes);

        // Download to host PatchInfo
        std::vector<uint8_t> h_stash(get_num_patches() * stash_bytes);
        CUDA_ERROR(cudaMemcpy(h_stash.data(), d_stash_tmp,
                              get_num_patches() * stash_bytes, cudaMemcpyDeviceToHost));
        for (uint32_t p = 0; p < get_num_patches(); ++p) {
            m_h_patches_info[p].patch_stash = PatchStash(false);
            memcpy(m_h_patches_info[p].patch_stash.m_stash,
                   h_stash.data() + p * stash_bytes, stash_bytes);
        }
        for (uint32_t p = get_num_patches(); p < get_max_num_patches(); ++p)
            m_h_patches_info[p].patch_stash = PatchStash(false);

        CUDA_ERROR(cudaFree(d_stash_tmp));
        fprintf(stderr, "[init] GPU populate_patch_stash done\n");
    } else {
        fprintf(stderr, "[init] populate_patch_stash...\n");
        populate_patch_stash();
        fprintf(stderr, "[init] populate_patch_stash done\n");
    }
    m_timers.stop("populate_patch_stash");

    // 3) coloring
    fprintf(stderr, "[init] coloring...\n");
    m_timers.add("coloring");
    m_timers.start("coloring");
    patch_graph_coloring();
    m_timers.stop("coloring");
    RXMESH_INFO("Num colors = {}", m_num_colors);
    fprintf(stderr, "[init] coloring done\n");

    // 4) build_device
    fprintf(stderr, "[init] build_device...\n");
    m_timers.add("build_device");
    m_timers.start("build_device");
    build_device();
    m_timers.stop("build_device");
    fprintf(stderr, "[init] build_device done\n");


    // 5)
    m_timers.add("PatchScheduler");
    m_timers.start("PatchScheduler");
    PatchScheduler sch;
    sch.init(get_max_num_patches());
    m_topo_memory_mega_bytes +=
        BYTES_TO_MEGABYTES(sizeof(uint32_t) * get_max_num_patches());
    sch.refill(get_num_patches());
    m_timers.stop("PatchScheduler");


    // 6)
    m_timers.add("allocate_extra_patches");
    m_timers.start("allocate_extra_patches");
    // Extra patches already set up in bulk by build_device when m_bulk_device_alloc
    if (!m_bulk_device_alloc)
        allocate_extra_patches();
    m_timers.stop("allocate_extra_patches");

    // 7)
    m_timers.add("create_handles");
    m_timers.start("create_handles");
    create_handles();
    m_timers.stop("create_handles");

    // 8)
    m_timers.add("context.init");
    m_timers.start("context.init");
    // Allocate and copy the context to the gpu
    m_rxmesh_context.init(m_num_vertices,
                          m_num_edges,
                          m_num_faces,
                          m_max_vertices_per_patch,
                          m_max_edges_per_patch,
                          m_max_faces_per_patch,
                          get_num_patches(),
                          get_max_num_patches(),
                          m_capacity_factor,
                          m_d_vertex_prefix,
                          m_d_edge_prefix,
                          m_d_face_prefix,
                          m_h_vertex_prefix,
                          m_h_edge_prefix,
                          m_h_face_prefix,
                          m_d_v_handles,
                          m_d_e_handles,
                          m_d_f_handles,
                          m_d_patches_info,
                          sch);
    m_timers.stop("context.init");


    RXMESH_INFO("#Vertices = {}, #Faces= {}, #Edges= {}, #Patches = {}",
                m_num_vertices,
                m_num_faces,
                m_num_edges,
                m_num_patches);
    RXMESH_INFO("Input is{} edge manifold",
                ((m_is_input_edge_manifold) ? "" : " Not"));
    RXMESH_INFO("Input is{} closed", ((m_is_input_closed) ? "" : " Not"));
    RXMESH_INFO("Input max valence = {}", m_input_max_valence);
    RXMESH_INFO("max edge incident faces = {}",
                m_input_max_edge_incident_faces);
    RXMESH_INFO("max face adjacent faces = {}",
                m_input_max_face_adjacent_faces);
    RXMESH_INFO("per-patch maximum face count = {}", m_max_faces_per_patch);
    RXMESH_INFO("per-patch maximum edge count = {}", m_max_edges_per_patch);
    RXMESH_INFO("per-patch maximum vertex count = {}",
                m_max_vertices_per_patch);

    ////
    RXMESH_INFO("1) build time = {} (ms)", m_timers.elapsed_millis("build"));
    RXMESH_INFO("2) populate_patch_stash time = {} (ms)",
                m_timers.elapsed_millis("populate_patch_stash"));
    RXMESH_INFO("3) patch graph coloring time = {} (ms)",
                m_timers.elapsed_millis("coloring"));
    RXMESH_INFO("4) build_device time = {} (ms)",
                m_timers.elapsed_millis("build_device"));
    RXMESH_INFO(" -buildHT time = {} (ms)", m_timers.elapsed_millis("buildHT"));
    RXMESH_INFO("   --lower_bound time = {} (ms)",
                m_timers.elapsed_millis("lower_bound"));
    RXMESH_INFO("   --ht.insert time = {} (ms)",
                m_timers.elapsed_millis("ht.insert"));
    RXMESH_INFO("   --hashtable.move time = {} (ms)",
                m_timers.elapsed_millis("hashtable.move"));
    RXMESH_INFO("   --LPHashTable time = {} (ms)",
                m_timers.elapsed_millis("LPHashTable"));
    RXMESH_INFO(" -bitmask time = {} (ms)", m_timers.elapsed_millis("bitmask"));
    RXMESH_INFO("   --bitmask.cudaMemcpy time = {} (ms)",
                m_timers.elapsed_millis("bitmask.cudaMemcpy"));

    RXMESH_INFO("5) PatchScheduler time = {} (ms)",
                m_timers.elapsed_millis("PatchScheduler"));
    RXMESH_INFO("6) allocate_extra_patches time = {} (ms)",
                m_timers.elapsed_millis("allocate_extra_patches"));
    RXMESH_INFO("7) create_handles time = {} (ms)",
                m_timers.elapsed_millis("create_handles"));
    RXMESH_INFO("8) context.init time = {} (ms)",
                m_timers.elapsed_millis("context.init"));

    RXMESH_INFO("cudaMemcpy time = {} (ms)",
                m_timers.elapsed_millis("cudaMemcpy"));
    RXMESH_INFO("cudaMalloc time = {} (ms)",
                m_timers.elapsed_millis("cudaMalloc"));
    RXMESH_INFO("malloc time = {} (ms)", m_timers.elapsed_millis("malloc"));
}

RXMesh::~RXMesh()
{
    m_rxmesh_context.m_patch_scheduler.free();

    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        free(m_h_patches_info[p].active_mask_v);
        free(m_h_patches_info[p].active_mask_e);
        free(m_h_patches_info[p].active_mask_f);
        free(m_h_patches_info[p].owned_mask_v);
        free(m_h_patches_info[p].owned_mask_e);
        free(m_h_patches_info[p].owned_mask_f);
        free(m_h_patches_info[p].num_faces);
        free(m_h_patches_info[p].dirty);
        m_h_patches_info[p].lp_v.free();
        m_h_patches_info[p].lp_e.free();
        m_h_patches_info[p].lp_f.free();
        // Only free host-allocated stash (not bulk device pointers)
        if (!m_h_patches_info[p].patch_stash.m_is_on_device)
            m_h_patches_info[p].patch_stash.free();
    }

    // m_d_patches_info is a pointer to pointer(s) which we can not dereference
    // on the host so we copy these pointers to the host by re-using
    // m_h_patches_info and then free the memory these pointers are pointing to.
    // Finally, we free the parent pointer memory

    if (m_bulk_device_alloc) {
        // Free bulk device arrays (not per-patch)
        for (auto ptr : m_bulk_device_ptrs)
            GPU_FREE(ptr);
        // PatchLock now in bulk — freed with bulk arrays above
    } else {
        CUDA_ERROR(cudaMemcpy(m_h_patches_info,
                              m_d_patches_info,
                              get_num_patches() * sizeof(PatchInfo),
                              cudaMemcpyDeviceToHost));
        for (uint32_t p = 0; p < get_num_patches(); ++p) {
            GPU_FREE(m_h_patches_info[p].active_mask_v);
            GPU_FREE(m_h_patches_info[p].active_mask_e);
            GPU_FREE(m_h_patches_info[p].active_mask_f);
            GPU_FREE(m_h_patches_info[p].owned_mask_v);
            GPU_FREE(m_h_patches_info[p].owned_mask_e);
            GPU_FREE(m_h_patches_info[p].owned_mask_f);
            GPU_FREE(m_h_patches_info[p].ev);
            GPU_FREE(m_h_patches_info[p].fe);
            GPU_FREE(m_h_patches_info[p].num_faces);
            GPU_FREE(m_h_patches_info[p].dirty);
            m_h_patches_info[p].lp_v.free();
            m_h_patches_info[p].lp_e.free();
            m_h_patches_info[p].lp_f.free();
            m_h_patches_info[p].patch_stash.free();
            m_h_patches_info[p].lock.free();
        }
    }
    GPU_FREE(m_d_patches_info);
    free(m_h_patches_info);
    m_rxmesh_context.release();

    GPU_FREE(m_d_vertex_prefix);
    GPU_FREE(m_d_edge_prefix);
    GPU_FREE(m_d_face_prefix);

    free(m_h_vertex_prefix);
    free(m_h_edge_prefix);
    free(m_h_face_prefix);

    free(m_h_v_handles);
    free(m_h_e_handles);
    free(m_h_f_handles);

    GPU_FREE(m_d_v_handles);
    GPU_FREE(m_d_e_handles);
    GPU_FREE(m_d_f_handles);
}

void RXMesh::build(const std::vector<std::vector<uint32_t>>& fv,
                   const std::string                         patcher_file)
{
    std::vector<uint32_t>              ff_values;
    std::vector<uint32_t>              ff_offset;
    std::vector<std::vector<uint32_t>> ef;
    std::vector<std::vector<uint32_t>> ev;

    m_max_capacity_lp_v = 0;
    m_max_capacity_lp_e = 0;
    m_max_capacity_lp_f = 0;

    auto _bt = std::chrono::high_resolution_clock::now();
    build_supporting_structures(fv, ev, ef, ff_offset, ff_values);
    fprintf(stderr, "[build] supporting_structures: %.0fms\n",
            std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-_bt).count());

    _bt = std::chrono::high_resolution_clock::now();
    if (!patcher_file.empty()) {
        if (!std::filesystem::exists(patcher_file)) {
            RXMESH_ERROR(
                "RXMesh::build patch file {} does not exit. Building unique "
                "patches.",
                patcher_file);
            m_patcher = std::make_unique<patcher::Patcher>(m_patch_size,
                                                           ff_offset,
                                                           ff_values,
                                                           m_flat_faces.data(),
                                                           m_num_faces,
                                                           m_sorted_edge_keys,
                                                           m_num_vertices,
                                                           m_num_edges,
                                                           false);
        } else {
            m_patcher = std::make_unique<patcher::Patcher>(patcher_file);
        }
    } else {
        m_patcher = std::make_unique<patcher::Patcher>(m_patch_size,
                                                       ff_offset,
                                                       ff_values,
                                                       m_flat_faces.data(),
                                                       m_num_faces,
                                                       m_sorted_edge_keys,
                                                       m_num_vertices,
                                                       m_num_edges,
                                                       false);
    }


    fprintf(stderr, "[build] patcher: %.0fms\n",
            std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-_bt).count());

    m_num_patches     = m_patcher->get_num_patches();
    m_max_num_patches = static_cast<uint32_t>(
        std::ceil(m_patch_alloc_factor * static_cast<float>(m_num_patches)));

    m_h_patches_info =
        (PatchInfo*)malloc(get_max_num_patches() * sizeof(PatchInfo));
    m_h_patches_ltog_f.resize(get_num_patches());
    m_h_patches_ltog_e.resize(get_num_patches());
    m_h_patches_ltog_v.resize(get_num_patches());
    m_h_num_owned_f.resize(get_max_num_patches(), 0);
    m_h_num_owned_v.resize(get_max_num_patches(), 0);
    m_h_num_owned_e.resize(get_max_num_patches(), 0);

    // GPU K0a: assign edge/vertex patch + GPU ribbon extraction
    if (m_d_ev != nullptr && m_d_ef_f0 != nullptr) {
        _bt = std::chrono::high_resolution_clock::now();
        uint32_t* d_fpc;
        uint32_t* d_epc;
        uint32_t* d_vpc;
        CUDA_ERROR(cudaMalloc(&d_fpc, m_num_faces * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&d_epc, m_num_edges * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&d_vpc, m_num_vertices * sizeof(uint32_t)));
        CUDA_ERROR(cudaMemcpy(d_fpc, m_patcher->get_face_patch().data(),
                              m_num_faces * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemset(d_vpc, 0xFF, m_num_vertices * sizeof(uint32_t)));
        gpu_run_k0a(d_fpc, m_d_ev, m_d_ef_f0, d_epc, d_vpc, m_num_edges);
        CUDA_ERROR(cudaMemcpy(m_patcher->get_edge_patch().data(), d_epc,
                              m_num_edges * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaMemcpy(m_patcher->get_vertex_patch().data(), d_vpc,
                              m_num_vertices * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_ERROR(cudaFree(d_epc));
        CUDA_ERROR(cudaFree(d_vpc));
        fprintf(stderr, "[build] GPU K0a assign_patch: %.0fms\n",
                std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-_bt).count());

        // GPU ribbon extraction (d_fpc still alive, m_d_fv retained from topo)
        _bt = std::chrono::high_resolution_clock::now();
        uint32_t* d_rv_gpu = nullptr;
        uint32_t* d_ro_gpu = nullptr;
        gpu_extract_ribbons(d_fpc, m_d_fv, m_num_faces, m_num_vertices,
                            get_num_patches(),
                            &d_rv_gpu, &d_ro_gpu,
                            m_patcher->get_external_ribbon_val(),
                            m_patcher->get_external_ribbon_offset());
        CUDA_ERROR(cudaFree(d_fpc));
        // d_rv_gpu/d_ro_gpu freed later (or used by Approach A)
        CUDA_ERROR(cudaFree(d_rv_gpu));
        CUDA_ERROR(cudaFree(d_ro_gpu));
        fprintf(stderr, "[build] GPU extract_ribbons: %.0fms\n",
                std::chrono::duration<double,std::milli>(std::chrono::high_resolution_clock::now()-_bt).count());
    } else {
        // CPU fallback
        m_patcher->assign_patch_fast(m_flat_faces.data(), m_num_faces, m_sorted_edge_keys);
    }

    // edge/vert_by_patch removed — only used by CPU fallback ltog (Approach A replaces it)

    // Approach A + K2: build ltog on GPU, retain device arrays for K2 topology
    // Device pointers for patcher data — kept alive for K2
    uint32_t *d_pv_k2 = nullptr, *d_po_k2 = nullptr;
    uint32_t *d_rv_k2 = nullptr, *d_ro_k2 = nullptr;
    uint32_t *d_fpc_k2 = nullptr, *d_epc_k2 = nullptr, *d_vpc_k2 = nullptr;
    ThrustLtogResult thr_k2;  // ltog with retained device arrays

    if (m_d_edge_key != nullptr) {
        fprintf(stderr, "[build] Approach A: thrust-based ltog...\n");

        // Use retained device face array (no re-upload needed)
        uint32_t* d_fv_k1 = m_d_fv;

        // Upload patcher results
        uint32_t* cpu_pv = m_patcher->get_patches_val();
        uint32_t* cpu_po = m_patcher->get_patches_offset();
        auto& cpu_rv = m_patcher->get_external_ribbon_val();
        auto& cpu_ro = m_patcher->get_external_ribbon_offset();

        std::vector<uint32_t> rib_pfx(get_num_patches() + 1);
        rib_pfx[0] = 0;
        for (uint32_t p = 0; p < get_num_patches(); ++p)
            rib_pfx[p + 1] = cpu_ro[p];
        uint32_t pv_total = cpu_po[get_num_patches() - 1];

        CUDA_ERROR(cudaMalloc(&d_pv_k2, pv_total * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&d_po_k2, get_num_patches() * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&d_rv_k2, std::max(cpu_rv.size(), (size_t)1) * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&d_ro_k2, rib_pfx.size() * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&d_fpc_k2, m_num_faces * sizeof(uint32_t)));

        CUDA_ERROR(cudaMemcpy(d_pv_k2, cpu_pv, pv_total*sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_po_k2, cpu_po, get_num_patches()*sizeof(uint32_t), cudaMemcpyHostToDevice));
        if (!cpu_rv.empty())
            CUDA_ERROR(cudaMemcpy(d_rv_k2, cpu_rv.data(), cpu_rv.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_ro_k2, rib_pfx.data(), rib_pfx.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_fpc_k2, m_patcher->get_face_patch().data(), m_num_faces*sizeof(uint32_t), cudaMemcpyHostToDevice));

        CUDA_ERROR(cudaMalloc(&d_epc_k2, m_num_edges * sizeof(uint32_t)));
        CUDA_ERROR(cudaMalloc(&d_vpc_k2, m_num_vertices * sizeof(uint32_t)));
        CUDA_ERROR(cudaMemcpy(d_epc_k2, m_patcher->get_edge_patch().data(), m_num_edges*sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_vpc_k2, m_patcher->get_vertex_patch().data(), m_num_vertices*sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Run thrust-based ltog (retains device arrays for K2)
        thr_k2 = gpu_thrust_build_ltog(
            d_fv_k1, m_d_edge_key, m_num_faces, m_num_edges, m_num_vertices,
            d_fpc_k2, d_epc_k2, d_vpc_k2,
            d_pv_k2, d_po_k2, d_rv_k2, d_ro_k2, get_num_patches());

        // Copy thrust results into m_h_patches_ltog_*
        for (uint32_t p = 0; p < get_num_patches(); ++p) {
            uint32_t fs = thr_k2.f_offset[p], fe = thr_k2.f_offset[p+1];
            uint32_t es = thr_k2.e_offset[p], ee = thr_k2.e_offset[p+1];
            uint32_t vs = thr_k2.v_offset[p], ve = thr_k2.v_offset[p+1];
            m_h_patches_ltog_f[p].assign(thr_k2.ltog_f.begin()+fs, thr_k2.ltog_f.begin()+fe);
            m_h_patches_ltog_e[p].assign(thr_k2.ltog_e.begin()+es, thr_k2.ltog_e.begin()+ee);
            m_h_patches_ltog_v[p].assign(thr_k2.ltog_v.begin()+vs, thr_k2.ltog_v.begin()+ve);
            m_h_num_owned_f[p] = thr_k2.num_owned_f[p];
            m_h_num_owned_e[p] = thr_k2.num_owned_e[p];
            m_h_num_owned_v[p] = thr_k2.num_owned_v[p];
        }

        // Free d_ef_f0 — K2 doesn't need it
        if (m_d_ef_f0) { CUDA_ERROR(cudaFree(m_d_ef_f0)); m_d_ef_f0 = nullptr; }

        // Keep all other device arrays alive for K2 (freed after K2 launch below)
    } else {
        // CPU fallback (only when GPU topology unavailable)
        std::vector<std::vector<uint32_t>> edges_by_patch(get_num_patches());
        std::vector<std::vector<uint32_t>> verts_by_patch(get_num_patches());
        for (uint32_t e = 0; e < m_num_edges; ++e) {
            uint32_t pid = m_patcher->get_edge_patch_id(e);
            if (pid < get_num_patches()) edges_by_patch[pid].push_back(e);
        }
        for (uint32_t v = 0; v < m_num_vertices; ++v) {
            uint32_t pid = m_patcher->get_vertex_patch_id(v);
            if (pid < get_num_patches()) verts_by_patch[pid].push_back(v);
        }
#pragma omp parallel for
        for (int p = 0; p < static_cast<int>(get_num_patches()); ++p) {
            build_single_patch_ltog(p, edges_by_patch[p], verts_by_patch[p]);
        }
    }

    // (K0a/K1 validation code removed — K2 now integrated via gpu_launch_k2)

    // calc max elements for use in build_device (which populates
    // m_h_patches_info and thus we can not use calc_max_elements now)
    m_max_vertices_per_patch = 0;
    m_max_edges_per_patch    = 0;
    m_max_faces_per_patch    = 0;
    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        m_max_vertices_per_patch =
            std::max(m_max_vertices_per_patch,
                     static_cast<uint32_t>(m_h_patches_ltog_v[p].size()));
        m_max_edges_per_patch =
            std::max(m_max_edges_per_patch,
                     static_cast<uint32_t>(m_h_patches_ltog_e[p].size()));
        m_max_faces_per_patch =
            std::max(m_max_faces_per_patch,
                     static_cast<uint32_t>(m_h_patches_ltog_f[p].size()));
    }

    m_max_vertex_capacity = static_cast<uint16_t>(std::ceil(
        m_capacity_factor * static_cast<float>(m_max_vertices_per_patch)));

    m_max_edge_capacity = static_cast<uint16_t>(std::ceil(
        m_capacity_factor * static_cast<float>(m_max_edges_per_patch)));

    m_max_face_capacity = static_cast<uint16_t>(std::ceil(
        m_capacity_factor * static_cast<float>(m_max_faces_per_patch)));

    fprintf(stderr, "[build] max calc done. max_e_cap=%u, max_f_cap=%u\n",
            m_max_edge_capacity, m_max_face_capacity);

    if (thr_k2.device_arrays_valid) {
        // GPU K2 topology path — launch K2 with retained device arrays
        fprintf(stderr, "[build] Launching GPU K2 topology...\n");

        m_gpu_k1k2_result = gpu_launch_k2(
            thr_k2,
            m_d_fv, m_d_edge_key, m_d_ev, m_num_edges,
            d_pv_k2, d_po_k2, d_rv_k2, d_ro_k2,
            d_fpc_k2, d_epc_k2, d_vpc_k2,
            get_num_patches(),
            m_max_edge_capacity, m_max_face_capacity);

        // Free patcher device arrays no longer needed
        CUDA_ERROR(cudaFree(d_pv_k2)); CUDA_ERROR(cudaFree(d_po_k2));
        CUDA_ERROR(cudaFree(d_rv_k2)); CUDA_ERROR(cudaFree(d_ro_k2));
        // Retain face/edge/vertex patch arrays for GPU build_device
        m_d_face_patch_bd = d_fpc_k2;
        m_d_edge_patch_bd = d_epc_k2;
        m_d_vertex_patch_bd = d_vpc_k2;
        // Free topology arrays no longer needed
        if (m_d_edge_key) { CUDA_ERROR(cudaFree(m_d_edge_key)); m_d_edge_key = nullptr; }
        if (m_d_ev) { CUDA_ERROR(cudaFree(m_d_ev)); m_d_ev = nullptr; }
        if (m_d_fv) { CUDA_ERROR(cudaFree(m_d_fv)); m_d_fv = nullptr; }
        // Retain ltog device arrays for GPU build_device
        m_retained_thr = std::move(thr_k2);

        // K2 ev/fe stay on device — D2D copy in build_device().
        // Allocate empty host ev/fe for host mirror (populated later if needed).
        for (uint32_t p = 0; p < get_num_patches(); ++p) {
            m_h_patches_info[p].ev = (LocalVertexT*)calloc(
                m_max_edge_capacity * 2, sizeof(LocalVertexT));
            m_h_patches_info[p].fe = (LocalEdgeT*)calloc(
                m_max_face_capacity * 3, sizeof(LocalEdgeT));
        }
        fprintf(stderr, "[build] GPU K2 topology done (device-retained)\n");
    } else {
        // CPU fallback
        fprintf(stderr, "[build] starting CPU topology loop (%u patches)\n",
                get_num_patches());
        for (int p = 0; p < static_cast<int>(get_num_patches()); ++p) {
            build_single_patch_topology(m_flat_faces.data(), p);
        }
        fprintf(stderr, "[build] CPU topology loop done\n");
    }

    fprintf(stderr, "[build] topology done\n");

    const uint32_t patches_1_bytes =
        (get_max_num_patches() + 1) * sizeof(uint32_t);

    m_h_vertex_prefix = (uint32_t*)malloc(patches_1_bytes);
    m_h_edge_prefix   = (uint32_t*)malloc(patches_1_bytes);
    m_h_face_prefix   = (uint32_t*)malloc(patches_1_bytes);

    memset(m_h_vertex_prefix, 0, patches_1_bytes);
    memset(m_h_edge_prefix, 0, patches_1_bytes);
    memset(m_h_face_prefix, 0, patches_1_bytes);

    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        m_h_vertex_prefix[p + 1] = m_h_vertex_prefix[p] + m_h_num_owned_v[p];
        m_h_edge_prefix[p + 1]   = m_h_edge_prefix[p] + m_h_num_owned_e[p];
        m_h_face_prefix[p + 1]   = m_h_face_prefix[p] + m_h_num_owned_f[p];
    }


    // the hash table capacity should be at least 2* the size of the stash
    m_max_capacity_lp_v = 2 * LPHashTable::stash_size;
    m_max_capacity_lp_e = 2 * LPHashTable::stash_size;
    m_max_capacity_lp_f = 2 * LPHashTable::stash_size;
    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        m_max_capacity_lp_v = std::max(
            m_max_capacity_lp_v,
            static_cast<uint16_t>(
                std::ceil(m_capacity_factor *
                          static_cast<float>(m_h_patches_ltog_v[p].size() -
                                             m_h_num_owned_v[p]) /
                          m_lp_hashtable_load_factor)));

        m_max_capacity_lp_e = std::max(
            m_max_capacity_lp_e,
            static_cast<uint16_t>(
                std::ceil(m_capacity_factor *
                          static_cast<float>(m_h_patches_ltog_e[p].size() -
                                             m_h_num_owned_e[p]) /
                          m_lp_hashtable_load_factor)));

        m_max_capacity_lp_f = std::max(
            m_max_capacity_lp_f,
            static_cast<uint16_t>(
                std::ceil(m_capacity_factor *
                          static_cast<float>(m_h_patches_ltog_f[p].size() -
                                             m_h_num_owned_f[p]) /
                          m_lp_hashtable_load_factor)));
    }

    m_timers.start("cudaMalloc");
    CUDA_ERROR(cudaMalloc((void**)&m_d_vertex_prefix, patches_1_bytes));
    // m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(patches_1_bytes);
    CUDA_ERROR(cudaMalloc((void**)&m_d_edge_prefix, patches_1_bytes));
    // m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(patches_1_bytes);
    CUDA_ERROR(cudaMalloc((void**)&m_d_face_prefix, patches_1_bytes));
    // m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(patches_1_bytes);
    m_timers.stop("cudaMalloc");


    CUDA_ERROR(cudaMemcpy(m_d_vertex_prefix,
                          m_h_vertex_prefix,
                          patches_1_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_edge_prefix,
                          m_h_edge_prefix,
                          patches_1_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(m_d_face_prefix,
                          m_h_face_prefix,
                          patches_1_bytes,
                          cudaMemcpyHostToDevice));

    // Skip calc_input_statistics — safe defaults for remesh (doesn't use these)
    m_input_max_valence = 256;
    m_is_input_edge_manifold = true;
    m_is_input_closed = true;
    m_input_max_edge_incident_faces = 2;
    m_input_max_face_adjacent_faces = 3;
}

void RXMesh::create_handles()
{
    // Allocate host + device memory
    m_h_v_handles =
        (VertexHandle*)malloc(sizeof(VertexHandle) * m_num_vertices);
    m_h_e_handles = (EdgeHandle*)malloc(sizeof(EdgeHandle) * m_num_edges);
    m_h_f_handles = (FaceHandle*)malloc(sizeof(FaceHandle) * m_num_faces);

    CUDA_ERROR(cudaMalloc((void**)&m_d_v_handles,
                          sizeof(VertexHandle) * m_num_vertices));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_e_handles, sizeof(EdgeHandle) * m_num_edges));
    CUDA_ERROR(
        cudaMalloc((void**)&m_d_f_handles, sizeof(FaceHandle) * m_num_faces));

    // GPU handle creation: build on device, download to host
    {
        // Upload owned counts
        uint16_t* d_nov; uint16_t* d_noe; uint16_t* d_nof;
        uint32_t P = get_num_patches();
        CUDA_ERROR(cudaMalloc(&d_nov, P * sizeof(uint16_t)));
        CUDA_ERROR(cudaMalloc(&d_noe, P * sizeof(uint16_t)));
        CUDA_ERROR(cudaMalloc(&d_nof, P * sizeof(uint16_t)));
        CUDA_ERROR(cudaMemcpy(d_nov, m_h_num_owned_v.data(), P*sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_noe, m_h_num_owned_e.data(), P*sizeof(uint16_t), cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_nof, m_h_num_owned_f.data(), P*sizeof(uint16_t), cudaMemcpyHostToDevice));

        gpu_create_handles(m_d_vertex_prefix, m_d_edge_prefix, m_d_face_prefix,
                           d_nov, d_noe, d_nof, P,
                           m_d_v_handles, m_d_e_handles, m_d_f_handles);

        CUDA_ERROR(cudaFree(d_nov));
        CUDA_ERROR(cudaFree(d_noe));
        CUDA_ERROR(cudaFree(d_nof));
    }

    // Download to host (for host-side iteration APIs)
    CUDA_ERROR(cudaMemcpy(m_h_v_handles, m_d_v_handles,
                          sizeof(VertexHandle) * m_num_vertices,
                          cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(m_h_e_handles, m_d_e_handles,
                          sizeof(EdgeHandle) * m_num_edges,
                          cudaMemcpyDeviceToHost));
    CUDA_ERROR(cudaMemcpy(m_h_f_handles, m_d_f_handles,
                          sizeof(FaceHandle) * m_num_faces,
                          cudaMemcpyDeviceToHost));
}
// ── GPU sort-scan topology construction ──────────────────────────────────
// Builds edge/adjacency data on GPU using thrust sort + scan,
// then populates m_edges_map on CPU (still needed by downstream).

void RXMesh::build_supporting_structures(
    const std::vector<std::vector<uint32_t>>& fv,
    std::vector<std::vector<uint32_t>>&       ev,
    std::vector<std::vector<uint32_t>>&       ef,
    std::vector<uint32_t>&                    ff_offset,
    std::vector<uint32_t>&                    ff_values)
{
    m_num_faces = m_flat_faces.empty()
        ? static_cast<uint32_t>(fv.size())
        : static_cast<uint32_t>(m_flat_faces.size() / 3);

    // Build flat face array for GPU (skip if m_flat_faces already provided)
    if (m_flat_faces.empty()) {
        m_flat_faces.resize(m_num_faces * 3);
        for (uint32_t f = 0; f < m_num_faces; ++f) {
            if (fv[f].size() != 3) {
                RXMESH_ERROR(
                    "rxmesh::build_supporting_structures() Face {} is not "
                    "triangle. Non-triangular faces are not supported", f);
                exit(EXIT_FAILURE);
            }
            m_flat_faces[f * 3 + 0] = fv[f][0];
            m_flat_faces[f * 3 + 1] = fv[f][1];
            m_flat_faces[f * 3 + 2] = fv[f][2];
        }
    }

    // ── GPU sort-scan topology build ─────────────────────────────────────
    auto gpu = gpu_build_topology(m_flat_faces.data(), m_num_faces);

    m_num_vertices = gpu.num_vertices;
    m_num_edges    = gpu.num_edges;

    // Retain device arrays for GPU patch construction
    m_d_edge_key = gpu.d_edge_key;
    m_d_ev = gpu.d_ev;
    m_d_ef_f0 = gpu.d_ef_f0;
    m_d_fv = gpu.d_fv;
    gpu.d_edge_key = nullptr;
    gpu.d_ev = nullptr;
    gpu.d_ef_f0 = nullptr;
    gpu.d_fv = nullptr;

    // Store flat arrays (skip 3s vector-of-vectors for ev/ef)
    m_ev_flat = std::move(gpu.ev_flat);   // [num_edges * 2]
    m_ef_f0 = std::move(gpu.ef_f0);      // [num_edges]
    m_ef_f1 = std::move(gpu.ef_f1);      // [num_edges]

    // Download sorted unique edge keys from GPU for fast binary search lookups
    m_sorted_edge_keys.resize(m_num_edges);
    CUDA_ERROR(cudaMemcpy(m_sorted_edge_keys.data(), m_d_edge_key,
                          m_num_edges * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Skip ev/ef vector-of-vectors construction (saves ~3s on large meshes).
    // CPU fallback ltog path and calc_input_statistics use flat arrays instead.
    // Only build if Approach A is unavailable (m_d_edge_key == nullptr).
    if (m_d_edge_key == nullptr) {
        ev.resize(m_num_edges);
        ef.resize(m_num_edges);
        for (uint32_t e = 0; e < m_num_edges; ++e) {
            ev[e] = {m_ev_flat[2 * e], m_ev_flat[2 * e + 1]};
            if (m_ef_f1[e] == UINT32_MAX) ef[e] = {m_ef_f0[e]};
            else ef[e] = {m_ef_f0[e], m_ef_f1[e]};
        }
    }

    // Use GPU-computed face-face adjacency directly
    ff_offset = std::move(gpu.ff_offset);
    ff_values = std::move(gpu.ff_values);
}

#if 0  // Old validation + original hash-map implementation
    {
        fprintf(stderr, "[GPU_TOPO_VALIDATE] CPU: %u verts, %u edges | GPU: %u verts, %u edges\n",
                m_num_vertices, m_num_edges, gpu.num_vertices, gpu.num_edges);

        if (gpu.num_edges != m_num_edges) {
            fprintf(stderr, "[GPU_TOPO_VALIDATE] ERROR: edge count mismatch!\n");
        }
        if (gpu.num_vertices != m_num_vertices) {
            fprintf(stderr, "[GPU_TOPO_VALIDATE] ERROR: vertex count mismatch!\n");
        }

        // Check ff_offset sizes
        fprintf(stderr, "[GPU_TOPO_VALIDATE] CPU ff_offset size=%zu, GPU ff_offset size=%zu\n",
                ff_offset.size(), gpu.ff_offset.size());
        fprintf(stderr, "[GPU_TOPO_VALIDATE] CPU ff_values size=%zu, GPU ff_values size=%zu\n",
                ff_values.size(), gpu.ff_values.size());

        // Check first few ff entries
        int mismatches = 0;
        for (uint32_t f = 0; f < std::min(m_num_faces, 20u); ++f) {
            uint32_t cpu_start = ff_offset[f], cpu_end = ff_offset[f+1];
            uint32_t gpu_start = (f < gpu.ff_offset.size()-1) ? gpu.ff_offset[f] : 0;
            uint32_t gpu_end   = (f+1 < gpu.ff_offset.size()) ? gpu.ff_offset[f+1] : 0;

            if ((cpu_end - cpu_start) != (gpu_end - gpu_start)) {
                fprintf(stderr, "[GPU_TOPO_VALIDATE] face %u: CPU has %u neighbors, GPU has %u\n",
                        f, cpu_end - cpu_start, gpu_end - gpu_start);
                mismatches++;
            }
        }

        // Check all faces for neighbor count mismatches
        int total_mm = 0;
        if (gpu.ff_offset.size() == ff_offset.size()) {
            for (uint32_t f = 0; f < m_num_faces; ++f) {
                uint32_t cn = ff_offset[f+1] - ff_offset[f];
                uint32_t gn = gpu.ff_offset[f+1] - gpu.ff_offset[f];
                if (cn != gn) total_mm++;
            }
            fprintf(stderr, "[GPU_TOPO_VALIDATE] total face neighbor mismatches: %d / %u\n",
                    total_mm, m_num_faces);
        }

        // Count interior edges (ef_f1 != UINT32_MAX)
        int cpu_interior = 0, gpu_interior = 0;
        for (uint32_t e = 0; e < m_num_edges; ++e)
            if (ef[e].size() >= 2) cpu_interior++;
        for (uint32_t e = 0; e < gpu.num_edges; ++e)
            if (gpu.ef_f1[e] != UINT32_MAX) gpu_interior++;
        fprintf(stderr, "[GPU_TOPO_VALIDATE] interior edges: CPU=%d, GPU=%d (boundary: CPU=%d, GPU=%d)\n",
                cpu_interior, gpu_interior,
                (int)m_num_edges - cpu_interior,
                (int)gpu.num_edges - gpu_interior);

        // Check ef consistency: for each GPU edge, verify faces share vertices
        int ef_bad = 0;
        for (uint32_t e = 0; e < gpu.num_edges && e < 10; ++e) {
            uint32_t v0 = gpu.ev_flat[2*e], v1 = gpu.ev_flat[2*e+1];
            uint32_t f0 = gpu.ef_f0[e];
            uint32_t f1 = gpu.ef_f1[e];
            // Check f0 contains v0 or v1
            bool f0_has_v0 = (fv[f0][0]==v0||fv[f0][1]==v0||fv[f0][2]==v0);
            bool f0_has_v1 = (fv[f0][0]==v1||fv[f0][1]==v1||fv[f0][2]==v1);
            if (!f0_has_v0 || !f0_has_v1) {
                fprintf(stderr, "[GPU_TOPO_VALIDATE] BAD ef_f0: edge %u (%u,%u) face %u verts=(%u,%u,%u)\n",
                        e, v0, v1, f0, fv[f0][0], fv[f0][1], fv[f0][2]);
                ef_bad++;
            }
            if (f1 != UINT32_MAX) {
                bool f1_has_v0 = (fv[f1][0]==v0||fv[f1][1]==v0||fv[f1][2]==v0);
                bool f1_has_v1 = (fv[f1][0]==v1||fv[f1][1]==v1||fv[f1][2]==v1);
                if (!f1_has_v0 || !f1_has_v1) {
                    fprintf(stderr, "[GPU_TOPO_VALIDATE] BAD ef_f1: edge %u (%u,%u) face %u verts=(%u,%u,%u)\n",
                            e, v0, v1, f1, fv[f1][0], fv[f1][1], fv[f1][2]);
                    ef_bad++;
                }
            }
        }
        // Check ALL edges — separate f0 and f1 error counts
        int ef_f0_bad = 0, ef_f1_bad = 0;
        for (uint32_t e = 0; e < gpu.num_edges; ++e) {
            uint32_t v0 = gpu.ev_flat[2*e], v1 = gpu.ev_flat[2*e+1];
            uint32_t f0 = gpu.ef_f0[e];
            bool f0_ok = (fv[f0][0]==v0||fv[f0][1]==v0||fv[f0][2]==v0) &&
                         (fv[f0][0]==v1||fv[f0][1]==v1||fv[f0][2]==v1);
            if (!f0_ok) ef_f0_bad++;
            if (gpu.ef_f1[e] != UINT32_MAX) {
                uint32_t f1 = gpu.ef_f1[e];
                bool f1_ok = (fv[f1][0]==v0||fv[f1][1]==v0||fv[f1][2]==v0) &&
                             (fv[f1][0]==v1||fv[f1][1]==v1||fv[f1][2]==v1);
                if (!f1_ok) ef_f1_bad++;
            }
        }
        fprintf(stderr, "[GPU_TOPO_VALIDATE] ef_f0 errors: %d, ef_f1 errors: %d\n",
                ef_f0_bad, ef_f1_bad);

        // For first few bad ef_f1: which face SHOULD it be?
        for (uint32_t e = 0; e < std::min(gpu.num_edges, 10u); ++e) {
            uint32_t v0 = gpu.ev_flat[2*e], v1 = gpu.ev_flat[2*e+1];
            // Find which CPU edge has (v0,v1)
            auto cpu_eid_it = m_edges_map.find(detail::edge_key(v0, v1));
            if (cpu_eid_it != m_edges_map.end()) {
                uint32_t cpu_eid = cpu_eid_it->second;
                fprintf(stderr, "[GPU_TOPO_VALIDATE] edge %u (%u,%u): gpu_f0=%u gpu_f1=%u | cpu_f0=%u cpu_f1=%s\n",
                        e, v0, v1, gpu.ef_f0[e],
                        gpu.ef_f1[e] == UINT32_MAX ? 99999 : gpu.ef_f1[e],
                        ef[cpu_eid][0],
                        ef[cpu_eid].size() >= 2 ? std::to_string(ef[cpu_eid][1]).c_str() : "boundary");
            }
        }

        // Debug face 1: dump its CPU and GPU neighbors
        for (uint32_t dbg_f : {1u, 12u, 19u}) {
            uint32_t cs = ff_offset[dbg_f], ce = ff_offset[dbg_f+1];
            uint32_t gs = gpu.ff_offset[dbg_f], ge = gpu.ff_offset[dbg_f+1];
            fprintf(stderr, "[GPU_TOPO_VALIDATE] face %u CPU neighbors:", dbg_f);
            for (uint32_t i = cs; i < ce; ++i) fprintf(stderr, " %u", ff_values[i]);
            fprintf(stderr, "\n[GPU_TOPO_VALIDATE] face %u GPU neighbors:", dbg_f);
            for (uint32_t i = gs; i < ge; ++i) fprintf(stderr, " %u", gpu.ff_values[i]);
            fprintf(stderr, "\n");
        }

        // Spot check ev
        int ev_mm = 0;
        for (uint32_t e = 0; e < std::min(m_num_edges, gpu.num_edges); ++e) {
            // GPU edges are in different order, so can't compare directly
            // Just check that each GPU edge exists in CPU edges_map
            uint32_t gv0 = gpu.ev_flat[2*e], gv1 = gpu.ev_flat[2*e+1];
            auto key = detail::edge_key(gv0, gv1);
            if (m_edges_map.find(key) == m_edges_map.end()) ev_mm++;
        }
        fprintf(stderr, "[GPU_TOPO_VALIDATE] GPU edges not found in CPU map: %d / %u\n",
                ev_mm, gpu.num_edges);
    }
}

#if 0
// ── Original hash-map-based implementation (kept for reference) ──────────
void RXMesh::build_supporting_structures_original(
    const std::vector<std::vector<uint32_t>>& fv,
    std::vector<std::vector<uint32_t>>&       ev,
    std::vector<std::vector<uint32_t>>&       ef,
    std::vector<uint32_t>&                    ff_offset,
    std::vector<uint32_t>&                    ff_values)
{
    m_num_faces    = static_cast<uint32_t>(fv.size());
    m_num_vertices = 0;
    m_num_edges    = 0;
    m_edges_map.clear();

    ef.clear();
    uint32_t reserve_size =
        static_cast<size_t>(1.5f * static_cast<float>(m_num_faces));
    ef.reserve(reserve_size);
    m_edges_map.reserve(reserve_size);
    ev.reserve(2 * reserve_size);

    std::vector<uint32_t> ff_size(m_num_faces, 0);

    for (uint32_t f = 0; f < fv.size(); ++f) {
        if (fv[f].size() != 3) {
            RXMESH_ERROR(
                "rxmesh::build_supporting_structures() Face {} is not "
                "triangle. Non-triangular faces are not supported",
                f);
            exit(EXIT_FAILURE);
        }

        for (uint32_t v = 0; v < fv[f].size(); ++v) {
            uint32_t v0 = fv[f][v];
            uint32_t v1 = fv[f][(v + 1) % 3];

            m_num_vertices = std::max(m_num_vertices, v0);

            std::pair<uint32_t, uint32_t> edge   = detail::edge_key(v0, v1);
            auto                          e_iter = m_edges_map.find(edge);
            if (e_iter == m_edges_map.end()) {
                uint32_t edge_id = m_num_edges++;
                m_edges_map.insert(std::make_pair(edge, edge_id));

                std::vector<uint32_t> evv = {v0, v1};
                ev.push_back(evv);

                std::vector<uint32_t> tmp(1, f);
                ef.push_back(tmp);
            } else {
                uint32_t edge_id = (*e_iter).second;

                for (uint32_t f0 = 0; f0 < ef[edge_id].size(); ++f0) {
                    uint32_t other_face = ef[edge_id][f0];
                    ++ff_size[other_face];
                }
                ff_size[f] += ef[edge_id].size();

                ef[edge_id].push_back(f);
            }
        }
    }
    ++m_num_vertices;

    if (m_num_edges != static_cast<uint32_t>(m_edges_map.size())) {
        RXMESH_ERROR(
            "rxmesh::build_supporting_structures() m_num_edges ({}) should "
            "match the size of edge_map ({})",
            m_num_edges,
            m_edges_map.size());
        exit(EXIT_FAILURE);
    }

    ff_offset.resize(m_num_faces + 1);
    std::exclusive_scan(ff_size.begin(), ff_size.end(), ff_offset.begin(), 0);
    ff_offset[m_num_faces] =
        ff_offset[m_num_faces - 1] + ff_size[m_num_faces - 1];
    ff_values.clear();
    ff_values.resize(ff_offset.back());
    std::fill(ff_size.begin(), ff_size.end(), 0);

    for (uint32_t e = 0; e < m_num_edges; ++e) {
        for (uint32_t i = 0; i < ef[e].size(); ++i) {
            uint32_t f0 = ef[e][i];
            for (uint32_t j = i + 1; j < ef[e].size(); ++j) {
                uint32_t f1 = ef[e][j];

                uint32_t f0_offset = ff_size[f0]++;
                uint32_t f1_offset = ff_size[f1]++;
                f0_offset += ff_offset[f0];
                f1_offset += ff_offset[f1];

                ff_values[f0_offset] = f1;
                ff_values[f1_offset] = f0;
            }
        }
    }
}
#endif  // inner #if 0 (original hash-map)
#endif  // outer #if 0 (old validation + original)

void RXMesh::calc_input_statistics(const std::vector<std::vector<uint32_t>>& fv,
                                   const std::vector<std::vector<uint32_t>>& ef)
{
    if (m_num_vertices == 0 || m_num_faces == 0 || m_num_edges == 0) {
        RXMESH_ERROR(
            "RXMesh::calc_statistics() input mesh has not been initialized");
        exit(EXIT_FAILURE);
    }

    // Use flat arrays (m_ev_flat, m_ef_f0, m_ef_f1) instead of m_edges_map + ef
    m_input_max_edge_incident_faces = 0;
    m_input_max_valence             = 0;
    std::vector<uint32_t> vv_count(m_num_vertices, 0);
    m_is_input_closed        = true;
    m_is_input_edge_manifold = true;

    for (uint32_t e = 0; e < m_num_edges; ++e) {
        uint32_t v0 = m_ev_flat[2 * e];
        uint32_t v1 = m_ev_flat[2 * e + 1];
        vv_count[v0]++;
        vv_count[v1]++;
        m_input_max_valence = std::max(m_input_max_valence, vv_count[v0]);
        m_input_max_valence = std::max(m_input_max_valence, vv_count[v1]);

        uint32_t ef_size = (m_ef_f1[e] == UINT32_MAX) ? 1 : 2;
        m_input_max_edge_incident_faces = std::max(
            m_input_max_edge_incident_faces, ef_size);
        if (ef_size < 2) m_is_input_closed = false;
        if (ef_size > 2) m_is_input_edge_manifold = false;
    }

    // calc max ff
    m_input_max_face_adjacent_faces = 0;
    for (uint32_t f = 0; f < m_num_faces; ++f) {
        uint32_t ff_count = 0;
        for (uint32_t v = 0; v < 3; ++v) {
            uint32_t v0 = m_flat_faces[f * 3 + v];
            uint32_t v1 = m_flat_faces[f * 3 + ((v + 1) % 3)];
            uint32_t edge_num = get_edge_id(v0, v1);
            uint32_t ef_sz = (m_ef_f1[edge_num] == UINT32_MAX) ? 1 : 2;
            ff_count += ef_sz - 1;
        }
        m_input_max_face_adjacent_faces =
            std::max(ff_count, m_input_max_face_adjacent_faces);
    }
}

void RXMesh::calc_max_elements()
{
    m_max_vertices_per_patch = 0;
    m_max_edges_per_patch    = 0;
    m_max_faces_per_patch    = 0;


    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        m_max_vertices_per_patch =
            std::max(m_max_vertices_per_patch,
                     uint32_t(m_h_patches_info[p].num_vertices[0]));
        m_max_edges_per_patch = std::max(
            m_max_edges_per_patch, uint32_t(m_h_patches_info[p].num_edges[0]));
        m_max_faces_per_patch = std::max(
            m_max_faces_per_patch, uint32_t(m_h_patches_info[p].num_faces[0]));
    }
}

void RXMesh::build_single_patch_ltog(
    const uint32_t                            patch_id,
    const std::vector<uint32_t>&              patch_edges,
    const std::vector<uint32_t>&              patch_verts)
{
    // patch start and end
    const uint32_t p_start =
        (patch_id == 0) ? 0 : m_patcher->get_patches_offset()[patch_id - 1];
    const uint32_t p_end = m_patcher->get_patches_offset()[patch_id];

    // ribbon start and end
    const uint32_t r_start =
        (patch_id == 0) ? 0 :
                          m_patcher->get_external_ribbon_offset()[patch_id - 1];
    const uint32_t r_end = m_patcher->get_external_ribbon_offset()[patch_id];


    const uint32_t total_patch_num_faces =
        (p_end - p_start) + (r_end - r_start);
    m_h_patches_ltog_f[patch_id].resize(total_patch_num_faces);
    m_h_patches_ltog_v[patch_id].reserve(3 * total_patch_num_faces);
    m_h_patches_ltog_e[patch_id].reserve(3 * total_patch_num_faces);

    // Use local sets instead of global-sized vector<bool> — each patch
    // only touches ~500 edges and ~300 vertices, not 10M+.
    std::unordered_set<uint32_t> added_verts;
    std::unordered_set<uint32_t> added_edges;
    added_verts.reserve(3 * total_patch_num_faces);
    added_edges.reserve(3 * total_patch_num_faces);

    // add faces owned by this patch
    auto add_new_face = [&](uint32_t global_face_id, uint16_t local_face_id) {
        m_h_patches_ltog_f[patch_id][local_face_id] = global_face_id;

        for (uint32_t v = 0; v < 3; ++v) {
            uint32_t v0 = m_flat_faces[global_face_id * 3 + v];
            uint32_t v1 = m_flat_faces[global_face_id * 3 + ((v + 1) % 3)];

            uint32_t edge_id = get_edge_id(v0, v1);

            if (added_verts.insert(v0).second) {
                m_h_patches_ltog_v[patch_id].push_back(v0);
            }

            if (added_edges.insert(edge_id).second) {
                m_h_patches_ltog_e[patch_id].push_back(edge_id);
            }
        }
    };

    uint16_t local_face_id = 0;
    for (uint32_t f = p_start; f < p_end; ++f) {
        uint32_t face_id = m_patcher->get_patches_val()[f];
        add_new_face(face_id, local_face_id++);
    }

    for (uint32_t f = r_start; f < r_end; ++f) {
        uint32_t face_id = m_patcher->get_external_ribbon_val()[f];
        add_new_face(face_id, local_face_id++);
    }

    // Safeguard: add any owned edges/vertices not already found via faces.
    // Uses pre-built per-patch lists instead of scanning all edges/vertices.
    for (uint32_t e : patch_edges) {
        if (added_edges.insert(e).second) {
            m_h_patches_ltog_e[patch_id].push_back(e);
            for (uint32_t i = 0; i < 2; ++i) {
                uint32_t v = m_ev_flat[e * 2 + i];
                if (added_verts.insert(v).second) {
                    m_h_patches_ltog_v[patch_id].push_back(v);
                }
            }
        }
    }

    for (uint32_t v : patch_verts) {
        if (added_verts.insert(v).second) {
            m_h_patches_ltog_v[patch_id].push_back(v);
        }
    }

    auto create_unique_mapping = [&](std::vector<uint32_t>&       ltog_map,
                                     const std::vector<uint32_t>& patch) {
        std::sort(ltog_map.begin(), ltog_map.end());
#ifndef NDEBUG
        auto unique_end = std::unique(ltog_map.begin(), ltog_map.end());
        assert(unique_end == ltog_map.end());
#endif

        // we use stable partition since we want ltog to be sorted so we can
        // use binary search on it when we populate the topology
        auto part_end = std::stable_partition(
            ltog_map.begin(), ltog_map.end(), [&patch, patch_id](uint32_t i) {
                return patch[i] == patch_id;
            });
        return static_cast<uint16_t>(part_end - ltog_map.begin());
    };

    m_h_num_owned_f[patch_id] = create_unique_mapping(
        m_h_patches_ltog_f[patch_id], m_patcher->get_face_patch());

    m_h_num_owned_e[patch_id] = create_unique_mapping(
        m_h_patches_ltog_e[patch_id], m_patcher->get_edge_patch());

    m_h_num_owned_v[patch_id] = create_unique_mapping(
        m_h_patches_ltog_v[patch_id], m_patcher->get_vertex_patch());
}

void RXMesh::build_single_patch_topology(
    const uint32_t* flat_fv,
    const uint32_t  patch_id)
{
    // patch start and end
    const uint32_t p_start =
        (patch_id == 0) ? 0 : m_patcher->get_patches_offset()[patch_id - 1];
    const uint32_t p_end = m_patcher->get_patches_offset()[patch_id];

    // ribbon start and end
    const uint32_t r_start =
        (patch_id == 0) ? 0 :
                          m_patcher->get_external_ribbon_offset()[patch_id - 1];
    const uint32_t r_end = m_patcher->get_external_ribbon_offset()[patch_id];

    const uint16_t patch_num_edges = m_h_patches_ltog_e[patch_id].size();

    const uint32_t edges_cap = m_max_edge_capacity;
    const uint32_t faces_cap = m_max_face_capacity;

    if (patch_id == 7) {
        fprintf(stderr, "[TOPO7] ne=%u nf=%zu edges_cap=%u faces_cap=%u owned_e=%u\n",
                patch_num_edges, m_h_patches_ltog_f[patch_id].size(),
                edges_cap, faces_cap, m_h_num_owned_e[patch_id]);
        fflush(stderr);
    }

    m_h_patches_info[patch_id].ev =
        (LocalVertexT*)malloc(edges_cap * 2 * sizeof(LocalVertexT));
    m_h_patches_info[patch_id].fe =
        (LocalEdgeT*)malloc(faces_cap * 3 * sizeof(LocalEdgeT));

    if (patch_id == 7) {
        fprintf(stderr, "[TOPO7] malloc done ev=%p fe=%p\n",
                (void*)m_h_patches_info[patch_id].ev,
                (void*)m_h_patches_info[patch_id].fe);
        fflush(stderr);
    }

    std::vector<bool> is_added_edge(patch_num_edges, false);

    auto find_local_index = [&patch_id](
                                const uint32_t               global_id,
                                const uint32_t               element_patch,
                                const uint16_t               num_owned_elements,
                                const std::vector<uint32_t>& ltog) -> uint16_t {
        uint32_t start = 0;
        uint32_t end   = num_owned_elements;
        if (element_patch != patch_id) {
            start = num_owned_elements;
            end   = ltog.size();
        }
        auto it = std::lower_bound(
            ltog.begin() + start, ltog.begin() + end, global_id);
        if (it == ltog.begin() + end) {
            return INVALID16;
        } else {
            return static_cast<uint16_t>(it - ltog.begin());
        }
    };


    auto add_new_face = [&](const uint32_t global_face_id) {
        const uint16_t local_face_id =
            find_local_index(global_face_id,
                             m_patcher->get_face_patch_id(global_face_id),
                             m_h_num_owned_f[patch_id],
                             m_h_patches_ltog_f[patch_id]);

        if (local_face_id == INVALID16) {
            fprintf(stderr, "[MISS_F] patch %u: face %u fp=%u owned_f=%u ltog_f.sz=%zu\n",
                    patch_id, global_face_id,
                    m_patcher->get_face_patch_id(global_face_id),
                    m_h_num_owned_f[patch_id],
                    m_h_patches_ltog_f[patch_id].size());
            return;
        }

        for (uint32_t v = 0; v < 3; ++v) {


            const uint32_t global_v0 = flat_fv[global_face_id * 3 + v];
            const uint32_t global_v1 = flat_fv[global_face_id * 3 + ((v + 1) % 3)];

            std::pair<uint32_t, uint32_t> edge_key =
                detail::edge_key(global_v0, global_v1);

            assert(edge_key.first == global_v0 || edge_key.first == global_v1);
            assert(edge_key.second == global_v0 ||
                   edge_key.second == global_v1);

            int dir = 1;
            if (edge_key.first == global_v0 && edge_key.second == global_v1) {
                dir = 0;
            }

            const uint32_t global_edge_id = get_edge_id(edge_key);

            uint16_t local_edge_id =
                find_local_index(global_edge_id,
                                 m_patcher->get_edge_patch_id(global_edge_id),
                                 m_h_num_owned_e[patch_id],
                                 m_h_patches_ltog_e[patch_id]);

            if (local_edge_id == INVALID16) {
                if (patch_id <= 10) {
                    fprintf(stderr, "[MISS] patch %u: edge %u (v=%u,%u) ep=%u owned_e=%u ltog_e.sz=%zu\n",
                            patch_id, global_edge_id, edge_key.first, edge_key.second,
                            m_patcher->get_edge_patch_id(global_edge_id),
                            m_h_num_owned_e[patch_id],
                            m_h_patches_ltog_e[patch_id].size());
                }
                continue;
            }
            if (!is_added_edge[local_edge_id]) {

                is_added_edge[local_edge_id] = true;

                const uint16_t local_v0 = find_local_index(
                    edge_key.first,
                    m_patcher->get_vertex_patch_id(edge_key.first),
                    m_h_num_owned_v[patch_id],
                    m_h_patches_ltog_v[patch_id]);

                const uint16_t local_v1 = find_local_index(
                    edge_key.second,
                    m_patcher->get_vertex_patch_id(edge_key.second),
                    m_h_num_owned_v[patch_id],
                    m_h_patches_ltog_v[patch_id]);

                if (local_v0 == INVALID16 || local_v1 == INVALID16) continue;

                m_h_patches_info[patch_id].ev[local_edge_id * 2].id = local_v0;
                m_h_patches_info[patch_id].ev[local_edge_id * 2 + 1].id =
                    local_v1;
            }

            // shift local_e to left
            // set the first bit to 1 if (dir ==1)
            local_edge_id = local_edge_id << 1;
            local_edge_id = local_edge_id | (dir & 1);
            m_h_patches_info[patch_id].fe[local_face_id * 3 + v].id =
                local_edge_id;
        }
    };


    for (uint32_t f = p_start; f < p_end; ++f) {
        uint32_t face_id = m_patcher->get_patches_val()[f];
        add_new_face(face_id);
    }

    for (uint32_t f = r_start; f < r_end; ++f) {
        uint32_t face_id = m_patcher->get_external_ribbon_val()[f];
        add_new_face(face_id);
    }
}

const VertexHandle RXMesh::map_to_local_vertex(uint32_t i) const
{
    auto pl = map_to_local<VertexHandle>(i, m_h_vertex_prefix);
    return {pl.first, pl.second};
}

const EdgeHandle RXMesh::map_to_local_edge(uint32_t i) const
{
    auto pl = map_to_local<EdgeHandle>(i, m_h_edge_prefix);
    return {pl.first, pl.second};
}

const FaceHandle RXMesh::map_to_local_face(uint32_t i) const
{
    auto pl = map_to_local<FaceHandle>(i, m_h_face_prefix);
    return {pl.first, pl.second};
}


template <typename HandleT>
const std::pair<uint32_t, uint16_t> RXMesh::map_to_local(
    const uint32_t  i,
    const uint32_t* element_prefix) const
{
    const auto end = element_prefix + get_num_patches() + 1;

    auto p = std::lower_bound(
        element_prefix, end, i, [](int a, int b) { return a <= b; });
    if (p == end) {
        RXMESH_ERROR(
            "RXMeshStatic::map_to_local can not its patch. Input is out of "
            "range!");
    }
    p -= 1;
    uint32_t patch_id = std::distance(element_prefix, p);
    uint32_t prefix   = i - *p;
    uint16_t local_id = 0;
    uint16_t num_elements =
        *(m_h_patches_info[patch_id].template get_num_elements<HandleT>());
    for (uint16_t l = 0; l < num_elements; ++l) {
        if (m_h_patches_info[patch_id].is_owned(typename HandleT::LocalT(l)) &&
            !m_h_patches_info[patch_id].is_deleted(
                typename HandleT::LocalT(l))) {
            if (local_id == prefix) {
                local_id = l;
                break;
            }
            local_id++;
        }
    }
    return {patch_id, local_id};
}


uint32_t RXMesh::get_edge_id(const uint32_t v0, const uint32_t v1) const
{
    // Fast binary search on sorted edge keys (replaces m_edges_map lookup)
    uint32_t lo = std::min(v0, v1), hi = std::max(v0, v1);
    uint64_t key = (uint64_t(lo) << 32) | uint64_t(hi);
    auto it = std::lower_bound(
        m_sorted_edge_keys.begin(), m_sorted_edge_keys.end(), key);
    if (it != m_sorted_edge_keys.end() && *it == key)
        return static_cast<uint32_t>(it - m_sorted_edge_keys.begin());
    return INVALID32;
}

uint32_t RXMesh::get_edge_id(const std::pair<uint32_t, uint32_t>& edge) const
{
    return get_edge_id(edge.first, edge.second);
}

uint16_t RXMesh::get_per_patch_max_vertex_capacity() const
{
    return m_max_vertex_capacity;
}
uint16_t RXMesh::get_per_patch_max_edge_capacity() const
{
    return m_max_edge_capacity;
}
uint16_t RXMesh::get_per_patch_max_face_capacity() const
{
    return m_max_face_capacity;
}

void RXMesh::populate_patch_stash()
{
    auto populate_patch_stash = [&](uint32_t                     p,
                                    const std::vector<uint32_t>& ltog,
                                    const std::vector<uint32_t>& element_patch,
                                    const uint16_t&              num_owned) {
        const uint16_t num_not_owned = ltog.size() - num_owned;

        // loop over all not-owned elements to populate PatchStash
        for (uint16_t i = 0; i < num_not_owned; ++i) {
            uint16_t local_id    = i + num_owned;
            uint32_t global_id   = ltog[local_id];
            uint32_t owner_patch = element_patch[global_id];

            m_h_patches_info[p].patch_stash.insert_patch(owner_patch);
        }
    };

    // #pragma omp parallel for
    for (int p = 0; p < static_cast<int>(get_num_patches()); ++p) {
        m_h_patches_info[p].patch_stash = PatchStash(false);

        const auto& vp_s = m_gpu_vertex_patch.empty()
            ? m_patcher->get_vertex_patch() : m_gpu_vertex_patch;
        const auto& ep_s = m_gpu_edge_patch.empty()
            ? m_patcher->get_edge_patch() : m_gpu_edge_patch;
        populate_patch_stash(p, m_h_patches_ltog_v[p], vp_s, m_h_num_owned_v[p]);
        populate_patch_stash(p, m_h_patches_ltog_e[p], ep_s, m_h_num_owned_e[p]);
        populate_patch_stash(p, m_h_patches_ltog_f[p],
                             m_patcher->get_face_patch(), m_h_num_owned_f[p]);
    }

    // #pragma omp parallel for
    for (int p = get_num_patches(); p < static_cast<int>(get_max_num_patches());
         ++p) {
        m_h_patches_info[p].patch_stash = PatchStash(false);
    }
}

void RXMesh::build_device()
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t_total = clk::now();

    const uint32_t P = get_num_patches();
    const uint32_t P_max = get_max_num_patches();
    const uint16_t v_cap = get_per_patch_max_vertex_capacity();
    const uint16_t e_cap = get_per_patch_max_edge_capacity();
    const uint16_t f_cap = get_per_patch_max_face_capacity();

    // ── Bulk GPU allocation (replaces 17*P individual cudaMalloc calls) ──
    auto tp = clk::now();
    CUDA_ERROR(cudaMalloc((void**)&m_d_patches_info, P_max * sizeof(PatchInfo)));

    // Per-patch buffer sizes (uniform stride)
    const size_t ev_bytes_per    = e_cap * 2 * sizeof(LocalVertexT);
    const size_t fe_bytes_per    = f_cap * 3 * sizeof(LocalEdgeT);
    const size_t mask_v_bytes    = detail::mask_num_bytes(v_cap);
    const size_t mask_e_bytes    = detail::mask_num_bytes(e_cap);
    const size_t mask_f_bytes    = detail::mask_num_bytes(f_cap);
    const size_t counts_bytes    = 6 * sizeof(uint16_t);  // 3 counts + padding
    const size_t stash_bytes     = PatchStash::stash_size * sizeof(uint32_t);
    const size_t dirty_bytes     = sizeof(int);

    // LPHashTable sizes
    const uint16_t lp_cap_v = max_lp_hashtable_capacity<LocalVertexT>();
    const uint16_t lp_cap_e = max_lp_hashtable_capacity<LocalEdgeT>();
    const uint16_t lp_cap_f = max_lp_hashtable_capacity<LocalFaceT>();
    // LPHashTable uses prime capacity — get the actual sizes
    auto prime = [](uint16_t c) -> uint16_t {
        // Replicate find_next_prime_number logic
        c = std::max(c, uint16_t(2));
        // Simple: just use c+1 as upper bound (LPHashTable will find actual prime)
        return c;  // approximate — actual prime is ≥ c
    };
    // Build dummy tables to get exact sizes
    LPHashTable dummy_v(lp_cap_v, false), dummy_e(lp_cap_e, false), dummy_f(lp_cap_f, false);
    const size_t ht_v_bytes   = dummy_v.num_bytes();
    const size_t ht_e_bytes   = dummy_e.num_bytes();
    const size_t ht_f_bytes   = dummy_f.num_bytes();
    const size_t ht_stash_bytes = LPHashTable::stash_size * sizeof(LPPair);
    free(dummy_v.m_table); free(dummy_v.m_stash);
    free(dummy_e.m_table); free(dummy_e.m_stash);
    free(dummy_f.m_table); free(dummy_f.m_stash);

    // Bulk allocate on device
    uint8_t *d_ev_bulk, *d_fe_bulk;
    uint8_t *d_mask_av_bulk, *d_mask_ae_bulk, *d_mask_af_bulk;
    uint8_t *d_mask_ov_bulk, *d_mask_oe_bulk, *d_mask_of_bulk;
    uint8_t *d_counts_bulk, *d_stash_bulk, *d_dirty_bulk;
    uint8_t *d_ht_v_bulk, *d_ht_e_bulk, *d_ht_f_bulk;
    uint8_t *d_ht_stash_v_bulk, *d_ht_stash_e_bulk, *d_ht_stash_f_bulk;

    CUDA_ERROR(cudaMalloc(&d_ev_bulk, P_max * ev_bytes_per));
    CUDA_ERROR(cudaMalloc(&d_fe_bulk, P_max * fe_bytes_per));
    CUDA_ERROR(cudaMalloc(&d_mask_av_bulk, P_max * mask_v_bytes));
    CUDA_ERROR(cudaMalloc(&d_mask_ae_bulk, P_max * mask_e_bytes));
    CUDA_ERROR(cudaMalloc(&d_mask_af_bulk, P_max * mask_f_bytes));
    CUDA_ERROR(cudaMalloc(&d_mask_ov_bulk, P_max * mask_v_bytes));
    CUDA_ERROR(cudaMalloc(&d_mask_oe_bulk, P_max * mask_e_bytes));
    CUDA_ERROR(cudaMalloc(&d_mask_of_bulk, P_max * mask_f_bytes));
    CUDA_ERROR(cudaMalloc(&d_counts_bulk, P_max * counts_bytes));
    CUDA_ERROR(cudaMalloc(&d_stash_bulk, P_max * stash_bytes));
    CUDA_ERROR(cudaMalloc(&d_dirty_bulk, P_max * dirty_bytes));
    CUDA_ERROR(cudaMalloc(&d_ht_v_bulk, P_max * ht_v_bytes));
    CUDA_ERROR(cudaMalloc(&d_ht_e_bulk, P_max * ht_e_bytes));
    CUDA_ERROR(cudaMalloc(&d_ht_f_bulk, P_max * ht_f_bytes));
    CUDA_ERROR(cudaMalloc(&d_ht_stash_v_bulk, P_max * ht_stash_bytes));
    CUDA_ERROR(cudaMalloc(&d_ht_stash_e_bulk, P_max * ht_stash_bytes));
    CUDA_ERROR(cudaMalloc(&d_ht_stash_f_bulk, P_max * ht_stash_bytes));
    // Zero/sentinel-fill ALL bulk arrays (covers extra patches P..P_max-1)
    CUDA_ERROR(cudaMemset(d_ev_bulk, 0, P_max * ev_bytes_per));
    CUDA_ERROR(cudaMemset(d_fe_bulk, 0, P_max * fe_bytes_per));
    CUDA_ERROR(cudaMemset(d_mask_av_bulk, 0, P_max * mask_v_bytes));
    CUDA_ERROR(cudaMemset(d_mask_ae_bulk, 0, P_max * mask_e_bytes));
    CUDA_ERROR(cudaMemset(d_mask_af_bulk, 0, P_max * mask_f_bytes));
    CUDA_ERROR(cudaMemset(d_mask_ov_bulk, 0, P_max * mask_v_bytes));
    CUDA_ERROR(cudaMemset(d_mask_oe_bulk, 0, P_max * mask_e_bytes));
    CUDA_ERROR(cudaMemset(d_mask_of_bulk, 0, P_max * mask_f_bytes));
    CUDA_ERROR(cudaMemset(d_counts_bulk, 0, P_max * counts_bytes));
    CUDA_ERROR(cudaMemset(d_stash_bulk, 0xFF, P_max * stash_bytes));
    CUDA_ERROR(cudaMemset(d_dirty_bulk, 0, P_max * dirty_bytes));
    CUDA_ERROR(cudaMemset(d_ht_v_bulk, 0xFF, P_max * ht_v_bytes));
    CUDA_ERROR(cudaMemset(d_ht_e_bulk, 0xFF, P_max * ht_e_bytes));
    CUDA_ERROR(cudaMemset(d_ht_f_bulk, 0xFF, P_max * ht_f_bytes));
    CUDA_ERROR(cudaMemset(d_ht_stash_v_bulk, 0xFF, P_max * ht_stash_bytes));
    CUDA_ERROR(cudaMemset(d_ht_stash_e_bulk, 0xFF, P_max * ht_stash_bytes));
    CUDA_ERROR(cudaMemset(d_ht_stash_f_bulk, 0xFF, P_max * ht_stash_bytes));
    fprintf(stderr, "[build_device] bulk alloc+memset: %.0fms (17 cudaMalloc vs %u×17)\n",
            ms_since(tp), P);

    // ── D2D copy ev/fe from K2 device arrays (if available) ──────────────
    if (m_gpu_k1k2_result.d_ev_local && m_gpu_k1k2_result.d_fe_local) {
        // K2 output uses same stride as bulk arrays (already zeroed by memset above)
        CUDA_ERROR(cudaMemcpy(d_ev_bulk, m_gpu_k1k2_result.d_ev_local,
                              P * ev_bytes_per, cudaMemcpyDeviceToDevice));
        CUDA_ERROR(cudaMemcpy(d_fe_bulk, m_gpu_k1k2_result.d_fe_local,
                              P * fe_bytes_per, cudaMemcpyDeviceToDevice));
        m_gpu_k1k2_result.free_device_topo();
        fprintf(stderr, "[build_device] ev/fe D2D copy done\n");
    }

    // ── GPU or CPU build of bitmasks + stash + hash tables ──────────────
    tp = clk::now();
    bool ev_fe_on_device = (m_gpu_k1k2_result.d_ev_local == nullptr);  // already copied

    if (m_retained_thr.device_arrays_valid) {
        // GPU path: build bitmasks + stash + HT entirely on device
        // (bulk arrays already memset above)
        gpu_build_device_data(
            m_retained_thr,
            m_d_face_patch_bd, m_d_edge_patch_bd, m_d_vertex_patch_bd,
            P, v_cap, e_cap, f_cap,
            d_mask_av_bulk, d_mask_ae_bulk, d_mask_af_bulk,
            d_mask_ov_bulk, d_mask_oe_bulk, d_mask_of_bulk,
            mask_v_bytes, mask_e_bytes, mask_f_bytes,
            d_counts_bulk, counts_bytes,
            d_stash_bulk, stash_bytes,
            d_ht_v_bulk, d_ht_e_bulk, d_ht_f_bulk,
            ht_v_bytes, ht_e_bytes, ht_f_bytes,
            d_ht_stash_v_bulk, d_ht_stash_e_bulk, d_ht_stash_f_bulk,
            ht_stash_bytes,
            dummy_v, dummy_e, dummy_f);

        // CPU HT rebuild (disabled — GPU kernel handles this now)
        if (false) {
            const auto& vp = m_gpu_vertex_patch.empty()
                ? m_patcher->get_vertex_patch() : m_gpu_vertex_patch;
            const auto& ep = m_gpu_edge_patch.empty()
                ? m_patcher->get_edge_patch() : m_gpu_edge_patch;
            const auto& fp = m_patcher->get_face_patch();
            std::vector<uint8_t> h_ht_v_staging(P * ht_v_bytes);
            std::vector<uint8_t> h_ht_e_staging(P * ht_e_bytes);
            std::vector<uint8_t> h_ht_f_staging(P * ht_f_bytes);
            std::vector<uint8_t> h_ht_stash_v(P * ht_stash_bytes);
            std::vector<uint8_t> h_ht_stash_e(P * ht_stash_bytes);
            std::vector<uint8_t> h_ht_stash_f(P * ht_stash_bytes);

            auto tp2 = clk::now();
            for (uint32_t p = 0; p < P; ++p) {
                auto build_ht_cpu = [&](const std::vector<std::vector<uint32_t>>& ltog,
                                   const std::vector<uint32_t>& p_ltog,
                                   const std::vector<uint32_t>& element_patch,
                                   const std::vector<uint16_t>& num_owned,
                                   uint16_t num_elems, uint16_t num_owned_elems,
                                   uint16_t cap, PatchStash& ps,
                                   uint8_t* ht_buf, size_t htb, uint8_t* stash_buf, size_t sb) {
                    LPHashTable h_ht(cap, false);
                    uint16_t nnot = num_elems - num_owned_elems;
                    for (uint16_t i = 0; i < nnot; ++i) {
                        uint16_t lid = i + num_owned_elems;
                        uint32_t gid = p_ltog[lid];
                        uint32_t owner = element_patch[gid];
                        auto it = std::lower_bound(ltog[owner].begin(),
                            ltog[owner].begin() + num_owned[owner], gid);
                        if (it != ltog[owner].begin() + num_owned[owner]) {
                            uint16_t lio = it - ltog[owner].begin();
                            uint8_t si = ps.find_patch_index(owner);
                            h_ht.insert(LPPair(lid, lio, si), nullptr, nullptr);
                        }
                    }
                    memcpy(ht_buf, h_ht.m_table, htb);
                    memcpy(stash_buf, h_ht.m_stash, sb);
                    free(h_ht.m_table); free(h_ht.m_stash);
                };
                uint16_t nv = m_h_patches_ltog_v[p].size();
                uint16_t ne = m_h_patches_ltog_e[p].size();
                uint16_t nf = m_h_patches_ltog_f[p].size();
                build_ht_cpu(m_h_patches_ltog_v, m_h_patches_ltog_v[p], vp, m_h_num_owned_v,
                    nv, m_h_num_owned_v[p], lp_cap_v, m_h_patches_info[p].patch_stash,
                    h_ht_v_staging.data()+p*ht_v_bytes, ht_v_bytes,
                    h_ht_stash_v.data()+p*ht_stash_bytes, ht_stash_bytes);
                build_ht_cpu(m_h_patches_ltog_e, m_h_patches_ltog_e[p], ep, m_h_num_owned_e,
                    ne, m_h_num_owned_e[p], lp_cap_e, m_h_patches_info[p].patch_stash,
                    h_ht_e_staging.data()+p*ht_e_bytes, ht_e_bytes,
                    h_ht_stash_e.data()+p*ht_stash_bytes, ht_stash_bytes);
                build_ht_cpu(m_h_patches_ltog_f, m_h_patches_ltog_f[p], fp, m_h_num_owned_f,
                    nf, m_h_num_owned_f[p], lp_cap_f, m_h_patches_info[p].patch_stash,
                    h_ht_f_staging.data()+p*ht_f_bytes, ht_f_bytes,
                    h_ht_stash_f.data()+p*ht_stash_bytes, ht_stash_bytes);
            }
            fprintf(stderr, "[build_device] CPU HT fallback: %.0fms\n", ms_since(tp2));
            CUDA_ERROR(cudaMemcpy(d_ht_v_bulk, h_ht_v_staging.data(), P*ht_v_bytes, cudaMemcpyHostToDevice));
            CUDA_ERROR(cudaMemcpy(d_ht_e_bulk, h_ht_e_staging.data(), P*ht_e_bytes, cudaMemcpyHostToDevice));
            CUDA_ERROR(cudaMemcpy(d_ht_f_bulk, h_ht_f_staging.data(), P*ht_f_bytes, cudaMemcpyHostToDevice));
            CUDA_ERROR(cudaMemcpy(d_ht_stash_v_bulk, h_ht_stash_v.data(), P*ht_stash_bytes, cudaMemcpyHostToDevice));
            CUDA_ERROR(cudaMemcpy(d_ht_stash_e_bulk, h_ht_stash_e.data(), P*ht_stash_bytes, cudaMemcpyHostToDevice));
            CUDA_ERROR(cudaMemcpy(d_ht_stash_f_bulk, h_ht_stash_f.data(), P*ht_stash_bytes, cudaMemcpyHostToDevice));
        }

        // Download stash to host for graph coloring + host mirror
        std::vector<uint8_t> h_stash_staging(P * stash_bytes, 0xFF);
        CUDA_ERROR(cudaMemcpy(h_stash_staging.data(), d_stash_bulk,
                              P * stash_bytes, cudaMemcpyDeviceToHost));

        // Populate host PatchInfo with downloaded stash + computed values
        for (uint32_t p = 0; p < P; ++p) {
            uint16_t nv = m_h_patches_ltog_v[p].size();
            uint16_t ne = m_h_patches_ltog_e[p].size();
            uint16_t nf = m_h_patches_ltog_f[p].size();

            m_h_patches_info[p].patch_stash = PatchStash();
            m_h_patches_info[p].patch_stash.m_stash =
                (uint32_t*)malloc(stash_bytes);
            memcpy(m_h_patches_info[p].patch_stash.m_stash,
                   h_stash_staging.data() + p * stash_bytes, stash_bytes);

            m_h_patches_info[p].num_faces = (uint16_t*)malloc(3 * sizeof(uint16_t));
            m_h_patches_info[p].num_faces[0] = nf;
            m_h_patches_info[p].num_edges = m_h_patches_info[p].num_faces + 1;
            m_h_patches_info[p].num_edges[0] = ne;
            m_h_patches_info[p].num_vertices = m_h_patches_info[p].num_faces + 2;
            m_h_patches_info[p].num_vertices[0] = nv;
            m_h_patches_info[p].vertices_capacity = v_cap;
            m_h_patches_info[p].edges_capacity = e_cap;
            m_h_patches_info[p].faces_capacity = f_cap;
            m_h_patches_info[p].patch_id = p;
            m_h_patches_info[p].dirty = (int*)malloc(sizeof(int));
            m_h_patches_info[p].dirty[0] = 0;
            m_h_patches_info[p].child_id = INVALID32;
            m_h_patches_info[p].should_slice = false;

            // Host bitmasks — pointers set from bulk download below

            // Host HT (empty — not needed for host-side queries in remesh)
            m_h_patches_info[p].lp_v = LPHashTable(lp_cap_v, false);
            m_h_patches_info[p].lp_e = LPHashTable(lp_cap_e, false);
            m_h_patches_info[p].lp_f = LPHashTable(lp_cap_f, false);
        }

        // Host PatchInfo for extra patches (P..P_max-1)
        for (uint32_t p = P; p < P_max; ++p) {
            m_h_patches_info[p] = PatchInfo();
            m_h_patches_info[p].num_faces = (uint16_t*)calloc(3, sizeof(uint16_t));
            m_h_patches_info[p].num_edges = m_h_patches_info[p].num_faces + 1;
            m_h_patches_info[p].num_vertices = m_h_patches_info[p].num_faces + 2;
            m_h_patches_info[p].vertices_capacity = v_cap;
            m_h_patches_info[p].edges_capacity = e_cap;
            m_h_patches_info[p].faces_capacity = f_cap;
            m_h_patches_info[p].patch_id = INVALID32;
            m_h_patches_info[p].dirty = (int*)calloc(1, sizeof(int));
            m_h_patches_info[p].child_id = INVALID32;
            m_h_patches_info[p].should_slice = false;
            m_h_patches_info[p].ev = (LocalVertexT*)calloc(e_cap * 2, sizeof(LocalVertexT));
            m_h_patches_info[p].fe = (LocalEdgeT*)calloc(f_cap * 3, sizeof(LocalEdgeT));
            m_h_patches_info[p].active_mask_v = (uint32_t*)calloc(1, mask_v_bytes);
            m_h_patches_info[p].active_mask_e = (uint32_t*)calloc(1, mask_e_bytes);
            m_h_patches_info[p].active_mask_f = (uint32_t*)calloc(1, mask_f_bytes);
            m_h_patches_info[p].owned_mask_v = (uint32_t*)calloc(1, mask_v_bytes);
            m_h_patches_info[p].owned_mask_e = (uint32_t*)calloc(1, mask_e_bytes);
            m_h_patches_info[p].owned_mask_f = (uint32_t*)calloc(1, mask_f_bytes);
            m_h_patches_info[p].lp_v = LPHashTable(lp_cap_v, false);
            m_h_patches_info[p].lp_e = LPHashTable(lp_cap_e, false);
            m_h_patches_info[p].lp_f = LPHashTable(lp_cap_f, false);
        }

        // Bulk download masks to host (6 cudaMemcpy instead of 6*P)
        {
            uint8_t* h_av = (uint8_t*)malloc(P * mask_v_bytes);
            uint8_t* h_ae = (uint8_t*)malloc(P * mask_e_bytes);
            uint8_t* h_af = (uint8_t*)malloc(P * mask_f_bytes);
            uint8_t* h_ov = (uint8_t*)malloc(P * mask_v_bytes);
            uint8_t* h_oe = (uint8_t*)malloc(P * mask_e_bytes);
            uint8_t* h_of = (uint8_t*)malloc(P * mask_f_bytes);
            CUDA_ERROR(cudaMemcpy(h_av, d_mask_av_bulk, P*mask_v_bytes, cudaMemcpyDeviceToHost));
            CUDA_ERROR(cudaMemcpy(h_ae, d_mask_ae_bulk, P*mask_e_bytes, cudaMemcpyDeviceToHost));
            CUDA_ERROR(cudaMemcpy(h_af, d_mask_af_bulk, P*mask_f_bytes, cudaMemcpyDeviceToHost));
            CUDA_ERROR(cudaMemcpy(h_ov, d_mask_ov_bulk, P*mask_v_bytes, cudaMemcpyDeviceToHost));
            CUDA_ERROR(cudaMemcpy(h_oe, d_mask_oe_bulk, P*mask_e_bytes, cudaMemcpyDeviceToHost));
            CUDA_ERROR(cudaMemcpy(h_of, d_mask_of_bulk, P*mask_f_bytes, cudaMemcpyDeviceToHost));
            for (uint32_t p = 0; p < P; ++p) {
                m_h_patches_info[p].active_mask_v = (uint32_t*)malloc(mask_v_bytes);
                m_h_patches_info[p].active_mask_e = (uint32_t*)malloc(mask_e_bytes);
                m_h_patches_info[p].active_mask_f = (uint32_t*)malloc(mask_f_bytes);
                m_h_patches_info[p].owned_mask_v = (uint32_t*)malloc(mask_v_bytes);
                m_h_patches_info[p].owned_mask_e = (uint32_t*)malloc(mask_e_bytes);
                m_h_patches_info[p].owned_mask_f = (uint32_t*)malloc(mask_f_bytes);
                memcpy(m_h_patches_info[p].active_mask_v, h_av + p*mask_v_bytes, mask_v_bytes);
                memcpy(m_h_patches_info[p].active_mask_e, h_ae + p*mask_e_bytes, mask_e_bytes);
                memcpy(m_h_patches_info[p].active_mask_f, h_af + p*mask_f_bytes, mask_f_bytes);
                memcpy(m_h_patches_info[p].owned_mask_v, h_ov + p*mask_v_bytes, mask_v_bytes);
                memcpy(m_h_patches_info[p].owned_mask_e, h_oe + p*mask_e_bytes, mask_e_bytes);
                memcpy(m_h_patches_info[p].owned_mask_f, h_of + p*mask_f_bytes, mask_f_bytes);
            }
            free(h_av); free(h_ae); free(h_af);
            free(h_ov); free(h_oe); free(h_of);
        }

        // Free retained device arrays
        m_retained_thr.free_device();
        if (m_d_face_patch_bd) { CUDA_ERROR(cudaFree(m_d_face_patch_bd)); m_d_face_patch_bd = nullptr; }
        if (m_d_edge_patch_bd) { CUDA_ERROR(cudaFree(m_d_edge_patch_bd)); m_d_edge_patch_bd = nullptr; }
        if (m_d_vertex_patch_bd) { CUDA_ERROR(cudaFree(m_d_vertex_patch_bd)); m_d_vertex_patch_bd = nullptr; }

        // (validation removed — GPU build verified byte-identical to CPU)
        // ── end GPU path ─────────────────────────────────────────────────
        if (false) {  // validation code removed
            const auto& vp = m_gpu_vertex_patch.empty()
                ? m_patcher->get_vertex_patch() : m_gpu_vertex_patch;
            const auto& ep = m_gpu_edge_patch.empty()
                ? m_patcher->get_edge_patch() : m_gpu_edge_patch;
            const auto& fp = m_patcher->get_face_patch();
            uint32_t dbg_p = 0;
            uint16_t nv = m_h_patches_ltog_v[dbg_p].size();
            uint16_t ne = m_h_patches_ltog_e[dbg_p].size();
            uint16_t nf = m_h_patches_ltog_f[dbg_p].size();
            uint16_t ov = m_h_num_owned_v[dbg_p];
            uint16_t oe = m_h_num_owned_e[dbg_p];
            uint16_t of = m_h_num_owned_f[dbg_p];

            // Download GPU counts
            uint16_t gpu_counts[3];
            CUDA_ERROR(cudaMemcpy(gpu_counts, d_counts_bulk, 3*sizeof(uint16_t), cudaMemcpyDeviceToHost));
            fprintf(stderr, "[VALIDATE] patch0 counts: GPU=[%u,%u,%u] CPU=[%u,%u,%u] %s\n",
                    gpu_counts[0], gpu_counts[1], gpu_counts[2], nf, ne, nv,
                    (gpu_counts[0]==nf && gpu_counts[1]==ne && gpu_counts[2]==nv) ? "OK" : "MISMATCH");

            // Download GPU active_mask_v and compare with CPU
            std::vector<uint8_t> gpu_mask(mask_v_bytes);
            CUDA_ERROR(cudaMemcpy(gpu_mask.data(), d_mask_av_bulk, mask_v_bytes, cudaMemcpyDeviceToHost));
            std::vector<uint8_t> cpu_mask(mask_v_bytes, 0);
            {   uint32_t* m = (uint32_t*)cpu_mask.data();
                for (uint16_t i = 0; i < nv; ++i) m[i/32] |= (1u << (i%32));
            }
            bool mask_ok = (gpu_mask == cpu_mask);
            fprintf(stderr, "[VALIDATE] patch0 active_mask_v: %s (nv=%u, cap=%u, mask_bytes=%zu)\n",
                    mask_ok ? "OK" : "MISMATCH", nv, v_cap, mask_v_bytes);
            if (!mask_ok) {
                uint32_t* gm = (uint32_t*)gpu_mask.data();
                uint32_t* cm = (uint32_t*)cpu_mask.data();
                for (size_t w = 0; w < mask_v_bytes/4 && w < 8; ++w)
                    fprintf(stderr, "  word[%zu]: GPU=%08x CPU=%08x\n", w, gm[w], cm[w]);
            }

            // Download GPU stash
            std::vector<uint32_t> gpu_stash(PatchStash::stash_size);
            CUDA_ERROR(cudaMemcpy(gpu_stash.data(), d_stash_bulk,
                                  PatchStash::stash_size*sizeof(uint32_t), cudaMemcpyDeviceToHost));
            // CPU stash (from populate_patch_stash which already ran)
            fprintf(stderr, "[VALIDATE] patch0 stash GPU=[%u,%u,%u,%u] CPU=[%u,%u,%u,%u]\n",
                    gpu_stash[0], gpu_stash[1], gpu_stash[2], gpu_stash[3],
                    m_h_patches_info[dbg_p].patch_stash.m_stash[0],
                    m_h_patches_info[dbg_p].patch_stash.m_stash[1],
                    m_h_patches_info[dbg_p].patch_stash.m_stash[2],
                    m_h_patches_info[dbg_p].patch_stash.m_stash[3]);

            // Build CPU HT for patch 0 vertices and compare
            LPHashTable cpu_ht(lp_cap_v, false);
            for (uint16_t i = ov; i < nv; ++i) {
                uint32_t gid = m_h_patches_ltog_v[dbg_p][i];
                uint32_t owner = vp[gid];
                auto it = std::lower_bound(
                    m_h_patches_ltog_v[owner].begin(),
                    m_h_patches_ltog_v[owner].begin() + m_h_num_owned_v[owner], gid);
                if (it != m_h_patches_ltog_v[owner].begin() + m_h_num_owned_v[owner]) {
                    uint16_t lio = it - m_h_patches_ltog_v[owner].begin();
                    uint8_t si = m_h_patches_info[dbg_p].patch_stash.find_patch_index(owner);
                    cpu_ht.insert(LPPair(i, lio, si), nullptr, nullptr);
                }
            }
            // Count GPU HT entries
            std::vector<LPPair> gpu_ht_data(dummy_v.m_capacity);
            CUDA_ERROR(cudaMemcpy(gpu_ht_data.data(), d_ht_v_bulk,
                                  dummy_v.m_capacity*sizeof(LPPair), cudaMemcpyDeviceToHost));
            int gpu_count = 0, cpu_count = 0;
            for (int i = 0; i < dummy_v.m_capacity; ++i) {
                if (!gpu_ht_data[i].is_sentinel()) gpu_count++;
                if (!cpu_ht.m_table[i].is_sentinel()) cpu_count++;
            }
            int ht_match = 0, ht_diff = 0;
            for (int i = 0; i < dummy_v.m_capacity; ++i) {
                if (gpu_ht_data[i].m_pair == cpu_ht.m_table[i].m_pair) ht_match++;
                else ht_diff++;
            }
            fprintf(stderr, "[VALIDATE] patch0 ht_v: GPU=%d CPU=%d entries, %d match %d differ (cap=%u)\n",
                    gpu_count, cpu_count, ht_match, ht_diff, dummy_v.m_capacity);
            if (ht_diff > 0 && ht_diff <= 5) {
                for (int i = 0; i < dummy_v.m_capacity; ++i) {
                    if (gpu_ht_data[i].m_pair != cpu_ht.m_table[i].m_pair)
                        fprintf(stderr, "  slot[%d]: GPU=%08x CPU=%08x\n",
                                i, gpu_ht_data[i].m_pair, cpu_ht.m_table[i].m_pair);
                }
            }
            free(cpu_ht.m_table); free(cpu_ht.m_stash);
        }
        // Compare PatchInfo sizeof and layout
        fprintf(stderr, "[VALIDATE] sizeof(PatchInfo)=%zu sizeof(LPHashTable)=%zu sizeof(PatchStash)=%zu sizeof(PatchLock)=%zu\n",
                sizeof(PatchInfo), sizeof(LPHashTable), sizeof(PatchStash), sizeof(PatchLock));
        fprintf(stderr, "[build_device] GPU build done\n");
        goto skip_cpu_staging;
    }

    // CPU fallback path
    {
    std::vector<uint8_t> h_mask_av(P * mask_v_bytes, 0);
    std::vector<uint8_t> h_mask_ae(P * mask_e_bytes, 0);
    std::vector<uint8_t> h_mask_af(P * mask_f_bytes, 0);
    std::vector<uint8_t> h_mask_ov(P * mask_v_bytes, 0);
    std::vector<uint8_t> h_mask_oe(P * mask_e_bytes, 0);
    std::vector<uint8_t> h_mask_of(P * mask_f_bytes, 0);
    std::vector<uint8_t> h_counts_staging(P * counts_bytes, 0);
    std::vector<uint8_t> h_stash_staging(P * stash_bytes, 0xFF);
    std::vector<uint8_t> h_dirty_staging(P * dirty_bytes, 0);
    std::vector<uint8_t> h_ht_v_staging(P * ht_v_bytes);
    std::vector<uint8_t> h_ht_e_staging(P * ht_e_bytes);
    std::vector<uint8_t> h_ht_f_staging(P * ht_f_bytes);
    std::vector<uint8_t> h_ht_stash_v(P * ht_stash_bytes);
    std::vector<uint8_t> h_ht_stash_e(P * ht_stash_bytes);
    std::vector<uint8_t> h_ht_stash_f(P * ht_stash_bytes);

    const auto& vp = m_gpu_vertex_patch.empty()
        ? m_patcher->get_vertex_patch() : m_gpu_vertex_patch;
    const auto& ep = m_gpu_edge_patch.empty()
        ? m_patcher->get_edge_patch() : m_gpu_edge_patch;
    const auto& fp = m_patcher->get_face_patch();

    for (uint32_t p = 0; p < P; ++p) {
        uint16_t nv = m_h_patches_ltog_v[p].size();
        uint16_t ne = m_h_patches_ltog_e[p].size();
        uint16_t nf = m_h_patches_ltog_f[p].size();
        uint16_t owned_v = m_h_num_owned_v[p];
        uint16_t owned_e = m_h_num_owned_e[p];
        uint16_t owned_f = m_h_num_owned_f[p];

        // Counts
        uint16_t* counts = (uint16_t*)(h_counts_staging.data() + p * counts_bytes);
        counts[0] = nf; counts[1] = ne; counts[2] = nv;

        // EV/FE topology: already on device via D2D from K2 (or CPU fallback below)

        // Bitmasks (active: bits 0..n-1 set, owned: bits 0..owned-1 set)
        auto fill_mask = [](uint8_t* buf, uint16_t cap, uint16_t n) {
            uint32_t* mask = (uint32_t*)buf;
            for (uint16_t i = 0; i < cap; ++i) {
                if (i < n) detail::bitmask_set_bit(i, mask);
                else detail::bitmask_clear_bit(i, mask);
            }
        };
        fill_mask(h_mask_av.data() + p * mask_v_bytes, v_cap, nv);
        fill_mask(h_mask_ae.data() + p * mask_e_bytes, e_cap, ne);
        fill_mask(h_mask_af.data() + p * mask_f_bytes, f_cap, nf);
        fill_mask(h_mask_ov.data() + p * mask_v_bytes, v_cap, owned_v);
        fill_mask(h_mask_oe.data() + p * mask_e_bytes, e_cap, owned_e);
        fill_mask(h_mask_of.data() + p * mask_f_bytes, f_cap, owned_f);

        // PatchStash
        memcpy(h_stash_staging.data() + p * stash_bytes,
               m_h_patches_info[p].patch_stash.m_stash, stash_bytes);

        // LPHashTables: build on host, copy into staging
        auto build_ht_batch = [&](const std::vector<std::vector<uint32_t>>& ltog,
                                   const std::vector<uint32_t>& p_ltog,
                                   const std::vector<uint32_t>& element_patch,
                                   const std::vector<uint16_t>& num_owned,
                                   uint16_t num_elems, uint16_t num_owned_elems,
                                   uint16_t cap, PatchStash& stash,
                                   uint8_t* ht_buf, size_t ht_bytes,
                                   uint8_t* stash_buf, size_t stash_sz) {
            LPHashTable h_ht(cap, false);
            uint16_t num_not_owned = num_elems - num_owned_elems;
            for (uint16_t i = 0; i < num_not_owned; ++i) {
                uint16_t local_id = i + num_owned_elems;
                uint32_t global_id = p_ltog[local_id];
                uint32_t owner_patch = element_patch[global_id];
                auto it = std::lower_bound(
                    ltog[owner_patch].begin(),
                    ltog[owner_patch].begin() + num_owned[owner_patch],
                    global_id);
                if (it != ltog[owner_patch].begin() + num_owned[owner_patch]) {
                    uint16_t local_in_owner = it - ltog[owner_patch].begin();
                    uint8_t owner_st = stash.find_patch_index(owner_patch);
                    h_ht.insert(LPPair(local_id, local_in_owner, owner_st), nullptr, nullptr);
                }
            }
            memcpy(ht_buf, h_ht.m_table, ht_bytes);
            memcpy(stash_buf, h_ht.m_stash, stash_sz);
            free(h_ht.m_table);
            free(h_ht.m_stash);
        };

        build_ht_batch(m_h_patches_ltog_v, m_h_patches_ltog_v[p], vp, m_h_num_owned_v,
                        nv, owned_v, lp_cap_v, m_h_patches_info[p].patch_stash,
                        h_ht_v_staging.data() + p * ht_v_bytes, ht_v_bytes,
                        h_ht_stash_v.data() + p * ht_stash_bytes, ht_stash_bytes);
        build_ht_batch(m_h_patches_ltog_e, m_h_patches_ltog_e[p], ep, m_h_num_owned_e,
                        ne, owned_e, lp_cap_e, m_h_patches_info[p].patch_stash,
                        h_ht_e_staging.data() + p * ht_e_bytes, ht_e_bytes,
                        h_ht_stash_e.data() + p * ht_stash_bytes, ht_stash_bytes);
        build_ht_batch(m_h_patches_ltog_f, m_h_patches_ltog_f[p], fp, m_h_num_owned_f,
                        nf, owned_f, lp_cap_f, m_h_patches_info[p].patch_stash,
                        h_ht_f_staging.data() + p * ht_f_bytes, ht_f_bytes,
                        h_ht_stash_f.data() + p * ht_stash_bytes, ht_stash_bytes);

        // Host PatchInfo setup (ev/fe already set, realloc to capacity)
        m_h_patches_info[p].ev = (LocalVertexT*)realloc(
            m_h_patches_info[p].ev, ev_bytes_per);
        m_h_patches_info[p].fe = (LocalEdgeT*)realloc(
            m_h_patches_info[p].fe, fe_bytes_per);
        m_h_patches_info[p].num_faces = (uint16_t*)malloc(3 * sizeof(uint16_t));
        m_h_patches_info[p].num_faces[0] = nf;
        m_h_patches_info[p].num_edges = m_h_patches_info[p].num_faces + 1;
        m_h_patches_info[p].num_edges[0] = ne;
        m_h_patches_info[p].num_vertices = m_h_patches_info[p].num_faces + 2;
        m_h_patches_info[p].num_vertices[0] = nv;
        m_h_patches_info[p].vertices_capacity = v_cap;
        m_h_patches_info[p].edges_capacity = e_cap;
        m_h_patches_info[p].faces_capacity = f_cap;
        m_h_patches_info[p].patch_id = p;
        m_h_patches_info[p].dirty = (int*)malloc(sizeof(int));
        m_h_patches_info[p].dirty[0] = 0;
        m_h_patches_info[p].child_id = INVALID32;
        m_h_patches_info[p].should_slice = false;

        // Host bitmasks
        m_h_patches_info[p].active_mask_v = (uint32_t*)malloc(mask_v_bytes);
        m_h_patches_info[p].active_mask_e = (uint32_t*)malloc(mask_e_bytes);
        m_h_patches_info[p].active_mask_f = (uint32_t*)malloc(mask_f_bytes);
        m_h_patches_info[p].owned_mask_v = (uint32_t*)malloc(mask_v_bytes);
        m_h_patches_info[p].owned_mask_e = (uint32_t*)malloc(mask_e_bytes);
        m_h_patches_info[p].owned_mask_f = (uint32_t*)malloc(mask_f_bytes);
        memcpy(m_h_patches_info[p].active_mask_v, h_mask_av.data() + p * mask_v_bytes, mask_v_bytes);
        memcpy(m_h_patches_info[p].active_mask_e, h_mask_ae.data() + p * mask_e_bytes, mask_e_bytes);
        memcpy(m_h_patches_info[p].active_mask_f, h_mask_af.data() + p * mask_f_bytes, mask_f_bytes);
        memcpy(m_h_patches_info[p].owned_mask_v, h_mask_ov.data() + p * mask_v_bytes, mask_v_bytes);
        memcpy(m_h_patches_info[p].owned_mask_e, h_mask_oe.data() + p * mask_e_bytes, mask_e_bytes);
        memcpy(m_h_patches_info[p].owned_mask_f, h_mask_of.data() + p * mask_f_bytes, mask_f_bytes);

        // Host hash tables
        m_h_patches_info[p].lp_v = LPHashTable(lp_cap_v, false);
        m_h_patches_info[p].lp_e = LPHashTable(lp_cap_e, false);
        m_h_patches_info[p].lp_f = LPHashTable(lp_cap_f, false);
        memcpy(m_h_patches_info[p].lp_v.m_table, h_ht_v_staging.data() + p * ht_v_bytes, ht_v_bytes);
        memcpy(m_h_patches_info[p].lp_v.m_stash, h_ht_stash_v.data() + p * ht_stash_bytes, ht_stash_bytes);
        memcpy(m_h_patches_info[p].lp_e.m_table, h_ht_e_staging.data() + p * ht_e_bytes, ht_e_bytes);
        memcpy(m_h_patches_info[p].lp_e.m_stash, h_ht_stash_e.data() + p * ht_stash_bytes, ht_stash_bytes);
        memcpy(m_h_patches_info[p].lp_f.m_table, h_ht_f_staging.data() + p * ht_f_bytes, ht_f_bytes);
        memcpy(m_h_patches_info[p].lp_f.m_stash, h_ht_stash_f.data() + p * ht_stash_bytes, ht_stash_bytes);
    }
    fprintf(stderr, "[build_device] host staging: %.0fms\n", ms_since(tp));

    // ── Bulk H2D copy (ev/fe already on device via D2D) ─────────────────
    tp = clk::now();
    // ev/fe: already on device from K2 D2D copy (or CPU fallback below)
    if (!ev_fe_on_device) {
        // CPU fallback: ev/fe need H2D upload from host PatchInfo
        // Build staging and upload
        std::vector<uint8_t> h_ev_staging(P * ev_bytes_per, 0);
        std::vector<uint8_t> h_fe_staging(P * fe_bytes_per, 0);
        for (uint32_t p = 0; p < P; ++p) {
            uint16_t ne = m_h_patches_ltog_e[p].size();
            uint16_t nf = m_h_patches_ltog_f[p].size();
            memcpy(h_ev_staging.data() + p * ev_bytes_per,
                   m_h_patches_info[p].ev, ne * 2 * sizeof(LocalVertexT));
            memcpy(h_fe_staging.data() + p * fe_bytes_per,
                   m_h_patches_info[p].fe, nf * 3 * sizeof(LocalEdgeT));
        }
        CUDA_ERROR(cudaMemcpy(d_ev_bulk, h_ev_staging.data(), P * ev_bytes_per, cudaMemcpyHostToDevice));
        CUDA_ERROR(cudaMemcpy(d_fe_bulk, h_fe_staging.data(), P * fe_bytes_per, cudaMemcpyHostToDevice));
    }
    CUDA_ERROR(cudaMemcpy(d_mask_av_bulk, h_mask_av.data(), P * mask_v_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_mask_ae_bulk, h_mask_ae.data(), P * mask_e_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_mask_af_bulk, h_mask_af.data(), P * mask_f_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_mask_ov_bulk, h_mask_ov.data(), P * mask_v_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_mask_oe_bulk, h_mask_oe.data(), P * mask_e_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_mask_of_bulk, h_mask_of.data(), P * mask_f_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_counts_bulk, h_counts_staging.data(), P * counts_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_stash_bulk, h_stash_staging.data(), P * stash_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_dirty_bulk, h_dirty_staging.data(), P * dirty_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_ht_v_bulk, h_ht_v_staging.data(), P * ht_v_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_ht_e_bulk, h_ht_e_staging.data(), P * ht_e_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_ht_f_bulk, h_ht_f_staging.data(), P * ht_f_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_ht_stash_v_bulk, h_ht_stash_v.data(), P * ht_stash_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_ht_stash_e_bulk, h_ht_stash_e.data(), P * ht_stash_bytes, cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_ht_stash_f_bulk, h_ht_stash_f.data(), P * ht_stash_bytes, cudaMemcpyHostToDevice));
    fprintf(stderr, "[build_device] H2D copy: %.0fms\n", ms_since(tp));
    }  // end CPU fallback block

    skip_cpu_staging:
    // ── Assemble PatchInfo structs with pointers into bulk arrays ─────────
    // Covers ALL patches (0..P_max-1) — eliminates allocate_extra_patches
    tp = clk::now();

    // Bulk PatchLock allocation (2 arrays instead of 2*P_max individual cudaMalloc)
    uint32_t* d_lock_bulk;
    uint32_t* d_spin_bulk;
    CUDA_ERROR(cudaMalloc(&d_lock_bulk, P_max * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc(&d_spin_bulk, P_max * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_lock_bulk, 0, P_max * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemset(d_spin_bulk, 0xFF, P_max * sizeof(uint32_t)));
    // d_lock_bulk and d_spin_bulk tracked in m_bulk_device_ptrs below

    std::vector<PatchInfo> h_d_patch_infos(P_max);
    for (uint32_t p = 0; p < P_max; ++p) {
        PatchInfo& di = h_d_patch_infos[p];
        di = PatchInfo();  // default init

        // Counts
        uint16_t* d_counts_p = (uint16_t*)(d_counts_bulk + p * counts_bytes);
        di.num_faces = d_counts_p;
        di.num_edges = d_counts_p + 1;
        di.num_vertices = d_counts_p + 2;
        di.vertices_capacity = v_cap;
        di.edges_capacity = e_cap;
        di.faces_capacity = f_cap;
        di.patch_id = p;
        di.color = (p < P) ? m_h_patches_info[p].color : INVALID32;

        // Topology
        di.ev = (LocalVertexT*)(d_ev_bulk + p * ev_bytes_per);
        di.fe = (LocalEdgeT*)(d_fe_bulk + p * fe_bytes_per);

        // Bitmasks
        di.active_mask_v = (uint32_t*)(d_mask_av_bulk + p * mask_v_bytes);
        di.active_mask_e = (uint32_t*)(d_mask_ae_bulk + p * mask_e_bytes);
        di.active_mask_f = (uint32_t*)(d_mask_af_bulk + p * mask_f_bytes);
        di.owned_mask_v = (uint32_t*)(d_mask_ov_bulk + p * mask_v_bytes);
        di.owned_mask_e = (uint32_t*)(d_mask_oe_bulk + p * mask_e_bytes);
        di.owned_mask_f = (uint32_t*)(d_mask_of_bulk + p * mask_f_bytes);

        // PatchStash
        di.patch_stash = PatchStash();
        di.patch_stash.m_stash = (uint32_t*)(d_stash_bulk + p * stash_bytes);

        // LPHashTables (point into bulk device arrays)
        // Use dummy templates for hash funcs — must match what gpu_build_device_data used
        di.lp_v = dummy_v;
        di.lp_v.m_table = (LPPair*)(d_ht_v_bulk + p * ht_v_bytes);
        di.lp_v.m_stash = (LPPair*)(d_ht_stash_v_bulk + p * ht_stash_bytes);
        di.lp_v.m_is_on_device = true;

        di.lp_e = dummy_e;
        di.lp_e.m_table = (LPPair*)(d_ht_e_bulk + p * ht_e_bytes);
        di.lp_e.m_stash = (LPPair*)(d_ht_stash_e_bulk + p * ht_stash_bytes);
        di.lp_e.m_is_on_device = true;

        di.lp_f = dummy_f;
        di.lp_f.m_table = (LPPair*)(d_ht_f_bulk + p * ht_f_bytes);
        di.lp_f.m_stash = (LPPair*)(d_ht_stash_f_bulk + p * ht_stash_bytes);
        di.lp_f.m_is_on_device = true;

        // Lock (bulk-allocated instead of per-patch cudaMalloc)
        di.lock.lock = d_lock_bulk + p;
        di.lock.spin = d_spin_bulk + p;

        di.dirty = (int*)(d_dirty_bulk + p * dirty_bytes);
        di.child_id = INVALID32;
        di.should_slice = false;
    }

    // Upload PatchInfo array
    CUDA_ERROR(cudaMemcpy(m_d_patches_info, h_d_patch_infos.data(),
                          P_max * sizeof(PatchInfo), cudaMemcpyHostToDevice));

    // Track bulk device pointers for cleanup (don't free per-patch)
    m_bulk_device_alloc = true;
    m_bulk_device_ptrs = {
        d_ev_bulk, d_fe_bulk,
        d_mask_av_bulk, d_mask_ae_bulk, d_mask_af_bulk,
        d_mask_ov_bulk, d_mask_oe_bulk, d_mask_of_bulk,
        d_counts_bulk, d_stash_bulk, d_dirty_bulk,
        d_ht_v_bulk, d_ht_e_bulk, d_ht_f_bulk,
        d_ht_stash_v_bulk, d_ht_stash_e_bulk, d_ht_stash_f_bulk,
        (uint8_t*)d_lock_bulk, (uint8_t*)d_spin_bulk,
    };

    fprintf(stderr, "[build_device] assemble: %.0fms\n", ms_since(tp));
    fprintf(stderr, "[build_device] TOTAL: %.0fms\n", ms_since(t_total));


    // make sure that if a patch stash of patch p has patch q, then q's patch
    // stash should have p in it
    for (uint32_t p = 0; p < get_num_patches(); ++p) {
        for (uint8_t p_sh = 0; p_sh < PatchStash::stash_size; ++p_sh) {
            uint32_t q = m_h_patches_info[p].patch_stash.get_patch(p_sh);
            if (q != INVALID32) {
                bool found = false;
                for (uint8_t q_sh = 0; q_sh < PatchStash::stash_size; ++q_sh) {
                    if (m_h_patches_info[q].patch_stash.get_patch(q_sh) == p) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    m_h_patches_info[q].patch_stash.insert_patch(p);
                }
            }
        }
    }
}

void RXMesh::build_device_single_patch(const uint32_t patch_id,
                                       const uint16_t p_num_vertices,
                                       const uint16_t p_num_edges,
                                       const uint16_t p_num_faces,
                                       const uint16_t p_vertices_capacity,
                                       const uint16_t p_edges_capacity,
                                       const uint16_t p_faces_capacity,
                                       const uint16_t p_num_owned_vertices,
                                       const uint16_t p_num_owned_edges,
                                       const uint16_t p_num_owned_faces,
                                       const std::vector<uint32_t>& ltog_v,
                                       const std::vector<uint32_t>& ltog_e,
                                       const std::vector<uint32_t>& ltog_f,
                                       PatchInfo& h_patch_info,
                                       PatchInfo& d_patch_info)
{


    m_timers.start("malloc");
    uint16_t* h_counts = (uint16_t*)malloc(3 * sizeof(uint16_t));
    m_timers.stop("malloc");

    h_patch_info.num_faces         = h_counts;
    h_patch_info.num_faces[0]      = p_num_faces;
    h_patch_info.num_edges         = h_counts + 1;
    h_patch_info.num_edges[0]      = p_num_edges;
    h_patch_info.num_vertices      = h_counts + 2;
    h_patch_info.num_vertices[0]   = p_num_vertices;
    h_patch_info.vertices_capacity = p_vertices_capacity;
    h_patch_info.edges_capacity    = p_edges_capacity;
    h_patch_info.faces_capacity    = p_faces_capacity;
    h_patch_info.patch_id          = patch_id;
    h_patch_info.dirty             = (int*)malloc(sizeof(int));
    h_patch_info.dirty[0]          = 0;
    h_patch_info.child_id          = INVALID32;
    h_patch_info.should_slice      = false;


    uint16_t* d_counts;

    m_timers.start("cudaMalloc");
    GPU_ALLOC_ASYNC(d_counts, 6 * sizeof(uint16_t));
    m_timers.stop("cudaMalloc");


    m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(3 * sizeof(uint16_t));

    PatchInfo d_patch;
    d_patch.num_faces         = d_counts;
    d_patch.num_edges         = d_counts + 1;
    d_patch.num_vertices      = d_counts + 2;
    d_patch.vertices_capacity = p_vertices_capacity;
    d_patch.edges_capacity    = p_edges_capacity;
    d_patch.faces_capacity    = p_faces_capacity;
    d_patch.patch_id          = patch_id;
    d_patch.color             = h_patch_info.color;
    d_patch.patch_stash       = PatchStash(true);
    d_patch.lock.init();
    d_patch.child_id     = INVALID32;
    d_patch.should_slice = false;

    m_topo_memory_mega_bytes +=
        BYTES_TO_MEGABYTES(PatchStash::stash_size * sizeof(uint32_t));

    // copy count and capacities
    m_timers.start("cudaMemcpy");
    CUDA_ERROR(cudaMemcpy(d_patch.num_faces,
                          h_patch_info.num_faces,
                          sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_patch.num_edges,
                          h_patch_info.num_edges,
                          sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_ERROR(cudaMemcpy(d_patch.num_vertices,
                          h_patch_info.num_vertices,
                          sizeof(uint16_t),
                          cudaMemcpyHostToDevice));
    m_timers.stop("cudaMemcpy");

    // allocate and copy patch topology to the device
    // we realloc the host h_patch_info EV and FE to ensure that both host and
    // device has the same capacity
    m_timers.start("cudaMalloc");
    GPU_ALLOC_ASYNC(d_patch.ev, p_edges_capacity * 2 * sizeof(LocalVertexT));
    m_timers.stop("cudaMalloc");


    m_topo_memory_mega_bytes +=
        BYTES_TO_MEGABYTES(p_edges_capacity * 2 * sizeof(LocalVertexT));
    h_patch_info.ev = (LocalVertexT*)realloc(
        h_patch_info.ev, p_edges_capacity * 2 * sizeof(LocalVertexT));

    if (p_num_edges > 0) {
        m_timers.start("cudaMemcpy");
        CUDA_ERROR(cudaMemcpy(d_patch.ev,
                              h_patch_info.ev,
                              p_num_edges * 2 * sizeof(LocalVertexT),
                              cudaMemcpyHostToDevice));
        m_timers.stop("cudaMemcpy");
    }

    m_timers.start("cudaMalloc");
    GPU_ALLOC_ASYNC(d_patch.fe, p_faces_capacity * 3 * sizeof(LocalEdgeT));
    m_timers.stop("cudaMalloc");


    m_topo_memory_mega_bytes +=
        BYTES_TO_MEGABYTES(p_faces_capacity * 3 * sizeof(LocalEdgeT));
    h_patch_info.fe = (LocalEdgeT*)realloc(
        h_patch_info.fe, p_faces_capacity * 3 * sizeof(LocalEdgeT));

    if (p_num_faces > 0) {
        m_timers.start("cudaMemcpy");
        CUDA_ERROR(cudaMemcpy(d_patch.fe,
                              h_patch_info.fe,
                              p_num_faces * 3 * sizeof(LocalEdgeT),
                              cudaMemcpyHostToDevice));
        m_timers.stop("cudaMemcpy");
    }

    m_timers.start("cudaMalloc");
    GPU_ALLOC_ASYNC(d_patch.dirty, sizeof(int));
    m_timers.stop("cudaMalloc");


    m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(sizeof(int));
    CUDA_ERROR(cudaMemset(d_patch.dirty, 0, sizeof(int)));


    // allocate and set bitmask
    auto bitmask = [&](uint32_t*& d_mask,
                       uint32_t*& h_mask,
                       uint32_t   capacity,
                       auto       predicate) {
        m_timers.start("bitmask");

        size_t num_bytes = detail::mask_num_bytes(capacity);

        m_timers.start("malloc");
        h_mask = (uint32_t*)malloc(num_bytes);
        m_timers.stop("malloc");

        m_timers.start("cudaMalloc");
        GPU_ALLOC_ASYNC(d_mask, num_bytes);
        m_timers.stop("cudaMalloc");


        m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(num_bytes);

        for (uint16_t i = 0; i < capacity; ++i) {
            if (predicate(i)) {
                detail::bitmask_set_bit(i, h_mask);
            } else {
                detail::bitmask_clear_bit(i, h_mask);
            }
        }

        m_timers.start("bitmask.cudaMemcpy");
        CUDA_ERROR(
            cudaMemcpy(d_mask, h_mask, num_bytes, cudaMemcpyHostToDevice));
        m_timers.stop("bitmask.cudaMemcpy");

        m_timers.stop("bitmask");
    };


    // vertices active mask
    bitmask(d_patch.active_mask_v,
            h_patch_info.active_mask_v,
            p_vertices_capacity,
            [&](uint16_t v) { return v < p_num_vertices; });

    // edges active mask
    bitmask(d_patch.active_mask_e,
            h_patch_info.active_mask_e,
            p_edges_capacity,
            [&](uint16_t e) { return e < p_num_edges; });

    // faces active mask
    bitmask(d_patch.active_mask_f,
            h_patch_info.active_mask_f,
            p_faces_capacity,
            [&](uint16_t f) { return f < p_num_faces; });

    // vertices owned mask
    bitmask(d_patch.owned_mask_v,
            h_patch_info.owned_mask_v,
            p_vertices_capacity,
            [&](uint16_t v) { return v < p_num_owned_vertices; });

    // edges owned mask
    bitmask(d_patch.owned_mask_e,
            h_patch_info.owned_mask_e,
            p_edges_capacity,
            [&](uint16_t e) { return e < p_num_owned_edges; });

    // faces owned mask
    bitmask(d_patch.owned_mask_f,
            h_patch_info.owned_mask_f,
            p_faces_capacity,
            [&](uint16_t f) { return f < p_num_owned_faces; });


    // Copy PatchStash
    if (patch_id != INVALID32) {
        m_timers.start("cudaMemcpy");
        CUDA_ERROR(cudaMemcpy(d_patch.patch_stash.m_stash,
                              h_patch_info.patch_stash.m_stash,
                              PatchStash::stash_size * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        m_timers.stop("cudaMemcpy");
    }


    // build LPHashtable
    auto build_ht = [&](const std::vector<std::vector<uint32_t>>& ltog,
                        const std::vector<uint32_t>&              p_ltog,
                        const std::vector<uint32_t>&              element_patch,
                        const std::vector<uint16_t>&              num_owned,
                        const uint16_t                            num_elements,
                        const uint16_t num_owned_elements,
                        const uint16_t cap,
                        PatchStash&    stash,
                        LPHashTable&   h_hashtable,
                        LPHashTable&   d_hashtable) {
        m_timers.start("buildHT");

        const uint16_t num_not_owned = num_elements - num_owned_elements;

        m_timers.start("LPHashTable");
        h_hashtable = LPHashTable(cap, false);
        d_hashtable = LPHashTable(cap, true);
        m_timers.stop("LPHashTable");

        m_topo_memory_mega_bytes += BYTES_TO_MEGABYTES(d_hashtable.num_bytes());
        m_topo_memory_mega_bytes +=
            BYTES_TO_MEGABYTES(LPHashTable::stash_size * sizeof(LPPair));

        for (uint16_t i = 0; i < num_not_owned; ++i) {
            uint16_t local_id    = i + num_owned_elements;
            uint32_t global_id   = p_ltog[local_id];
            uint32_t owner_patch = element_patch[global_id];

            m_timers.start("lower_bound");
            auto it = std::lower_bound(
                ltog[owner_patch].begin(),
                ltog[owner_patch].begin() + num_owned[owner_patch],
                global_id);
            m_timers.stop("lower_bound");

            if (it == ltog[owner_patch].begin() + num_owned[owner_patch]) {
                RXMESH_ERROR(
                    "rxmesh::build_device can not find the local id of "
                    "{} in patch {}. Maybe this patch does not own "
                    "this mesh element.",
                    global_id,
                    owner_patch);
            } else {
                uint16_t local_id_in_owner_patch =
                    static_cast<uint16_t>(it - ltog[owner_patch].begin());

                uint8_t owner_st = stash.find_patch_index(owner_patch);

                m_timers.start("ht.insert");
                LPPair pair(local_id, local_id_in_owner_patch, owner_st);
                if (!h_hashtable.insert(pair, nullptr, nullptr)) {
                    RXMESH_ERROR(
                        "rxmesh::build_device failed to insert in the "
                        "hashtable. Retry with smaller load factor. Load "
                        "factor used = {}",
                        m_lp_hashtable_load_factor);
                }
                m_timers.stop("ht.insert");
            }
        }

        m_timers.start("hashtable.move");
        d_hashtable.move(h_hashtable);
        m_timers.stop("hashtable.move");

        m_timers.stop("buildHT");
    };

    // Use GPU patch arrays when available, CPU patcher arrays otherwise
    const auto& vp = m_gpu_vertex_patch.empty()
        ? m_patcher->get_vertex_patch() : m_gpu_vertex_patch;
    const auto& ep = m_gpu_edge_patch.empty()
        ? m_patcher->get_edge_patch() : m_gpu_edge_patch;
    const auto& fp = m_patcher->get_face_patch();  // face IDs are same in both spaces

    const uint16_t lp_cap_v = max_lp_hashtable_capacity<LocalVertexT>();
    build_ht(m_h_patches_ltog_v,
             ltog_v, vp, m_h_num_owned_v,
             p_num_vertices, p_num_owned_vertices, lp_cap_v,
             h_patch_info.patch_stash, h_patch_info.lp_v, d_patch.lp_v);

    const uint16_t lp_cap_e = max_lp_hashtable_capacity<LocalEdgeT>();
    build_ht(m_h_patches_ltog_e,
             ltog_e, ep, m_h_num_owned_e,
             p_num_edges, p_num_owned_edges, lp_cap_e,
             h_patch_info.patch_stash, h_patch_info.lp_e, d_patch.lp_e);

    const uint16_t lp_cap_f = max_lp_hashtable_capacity<LocalFaceT>();
    build_ht(m_h_patches_ltog_f,
             ltog_f, fp, m_h_num_owned_f,
             p_num_faces, p_num_owned_faces, lp_cap_f,
             h_patch_info.patch_stash, h_patch_info.lp_f, d_patch.lp_f);

    m_timers.start("cudaMemcpy");
    CUDA_ERROR(cudaMemcpy(
        &d_patch_info, &d_patch, sizeof(PatchInfo), cudaMemcpyHostToDevice));
    m_timers.stop("cudaMemcpy");
}

void RXMesh::allocate_extra_patches()
{
    if (get_num_patches() >= get_max_num_patches())
        return;

    const uint16_t p_vertices_capacity = get_per_patch_max_vertex_capacity();
    const uint16_t p_edges_capacity    = get_per_patch_max_edge_capacity();
    const uint16_t p_faces_capacity    = get_per_patch_max_face_capacity();

#pragma omp parallel for
    for (int p = get_num_patches(); p < static_cast<int>(get_max_num_patches());
         ++p) {

        const uint16_t p_num_vertices = 0;
        const uint16_t p_num_edges    = 0;
        const uint16_t p_num_faces    = 0;

        m_timers.start("malloc");
        m_h_patches_info[p].ev =
            (LocalVertexT*)malloc(2 * p_edges_capacity * sizeof(LocalVertexT));
        m_h_patches_info[p].fe =
            (LocalEdgeT*)malloc(3 * p_faces_capacity * sizeof(LocalEdgeT));
        m_timers.stop("malloc");

        build_device_single_patch(INVALID32,
                                  p_num_vertices,
                                  p_num_edges,
                                  p_num_faces,
                                  p_vertices_capacity,
                                  p_edges_capacity,
                                  p_faces_capacity,
                                  m_h_num_owned_v[p],
                                  m_h_num_owned_e[p],
                                  m_h_num_owned_f[p],
                                  m_h_patches_ltog_v[0],
                                  m_h_patches_ltog_e[0],
                                  m_h_patches_ltog_f[0],
                                  m_h_patches_info[p],
                                  m_d_patches_info[p]);
    }

    for (uint32_t p = 0; p < get_max_num_patches(); ++p) {
        m_max_capacity_lp_v = std::max(m_max_capacity_lp_v,
                                       m_h_patches_info[p].lp_v.get_capacity());

        m_max_capacity_lp_e = std::max(m_max_capacity_lp_e,
                                       m_h_patches_info[p].lp_e.get_capacity());

        m_max_capacity_lp_f = std::max(m_max_capacity_lp_f,
                                       m_h_patches_info[p].lp_f.get_capacity());
    }
}

void RXMesh::patch_graph_coloring()
{
    constexpr uint32_t SS = PatchStash::stash_size;

    // Upload flat stash array to device
    std::vector<uint32_t> h_flat_stash(m_num_patches * SS, INVALID32);
    for (uint32_t p = 0; p < m_num_patches; ++p)
        memcpy(h_flat_stash.data() + p * SS,
               m_h_patches_info[p].patch_stash.m_stash, SS * sizeof(uint32_t));

    uint32_t* d_stash;
    CUDA_ERROR(cudaMalloc(&d_stash, m_num_patches * SS * sizeof(uint32_t)));
    CUDA_ERROR(cudaMemcpy(d_stash, h_flat_stash.data(),
                          m_num_patches * SS * sizeof(uint32_t), cudaMemcpyHostToDevice));

    std::vector<uint32_t> h_colors(m_num_patches);
    gpu_patch_coloring(d_stash, m_num_patches, SS, h_colors.data(), m_num_colors);

    for (uint32_t p = 0; p < m_num_patches; ++p)
        m_h_patches_info[p].color = h_colors[p];

    CUDA_ERROR(cudaFree(d_stash));
}
}  // namespace rxmesh
