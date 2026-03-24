// C API implementation — thin wrappers around pipeline.h functions.

#include "pyrxmesh_c.h"
#include "pipeline.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

extern "C" {

void pyrxmesh_init(int device_id)
{
    pipeline_init(device_id);
}

PyrxmeshQuadwildParams pyrxmesh_quadwild_default_params(void)
{
    PyrxmeshQuadwildParams p;
    p.target_edge_length = 0;
    p.target_faces       = 10000;
    p.num_iterations     = 3;
    p.num_smooth_iters   = 5;
    p.verbose            = 0;
    return p;
}

int pyrxmesh_quadwild_preprocess(
    const float*    vertices, int num_vertices,
    const int32_t*  faces,    int num_faces,
    const PyrxmeshQuadwildParams* params,
    PyrxmeshMeshResult* result)
{
    PyrxmeshQuadwildParams p = params ? *params : pyrxmesh_quadwild_default_params();

    QuadwildParams qp;
    qp.target_edge_length = p.target_edge_length;
    qp.target_faces       = p.target_faces;
    qp.num_iterations     = p.num_iterations;
    qp.num_smooth_iters   = p.num_smooth_iters;

    try {
        MeshResult mr = pipeline_quadwild_preprocess(
            vertices, num_vertices, faces, num_faces, qp, p.verbose != 0);

        result->num_vertices = mr.num_vertices;
        result->num_faces    = mr.num_faces;
        result->vertices = (float*)malloc(mr.num_vertices * 3 * sizeof(float));
        result->faces    = (int32_t*)malloc(mr.num_faces * 3 * sizeof(int32_t));

        for (int i = 0; i < mr.num_vertices * 3; ++i)
            result->vertices[i] = static_cast<float>(mr.vertices[i]);
        memcpy(result->faces, mr.faces.data(), mr.num_faces * 3 * sizeof(int32_t));

        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[pyrxmesh] quadwild_preprocess error: %s\n", e.what());
        result->vertices = nullptr;
        result->faces    = nullptr;
        result->num_vertices = 0;
        result->num_faces    = 0;
        return -1;
    }
}

void pyrxmesh_free_result(PyrxmeshMeshResult* result)
{
    free(result->vertices);
    free(result->faces);
    result->vertices = nullptr;
    result->faces    = nullptr;
    result->num_vertices = 0;
    result->num_faces    = 0;
}

int pyrxmesh_remesh(
    const double*   vertices, int num_vertices,
    const int32_t*  faces,    int num_faces,
    double relative_len, int iterations, int smooth_iterations,
    int verbose,
    double** out_vertices, int* out_num_vertices,
    int32_t** out_faces, int* out_num_faces)
{
    try {
        MeshResult mr = pipeline_remesh(
            vertices, num_vertices, faces, num_faces,
            relative_len, iterations, smooth_iterations, verbose != 0);

        *out_num_vertices = mr.num_vertices;
        *out_num_faces    = mr.num_faces;
        *out_vertices = (double*)malloc(mr.num_vertices * 3 * sizeof(double));
        *out_faces    = (int32_t*)malloc(mr.num_faces * 3 * sizeof(int32_t));

        memcpy(*out_vertices, mr.vertices.data(), mr.num_vertices * 3 * sizeof(double));
        memcpy(*out_faces, mr.faces.data(), mr.num_faces * 3 * sizeof(int32_t));

        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[pyrxmesh] remesh error: %s\n", e.what());
        *out_vertices = nullptr;
        *out_faces    = nullptr;
        *out_num_vertices = 0;
        *out_num_faces    = 0;
        return -1;
    }
}

} // extern "C"
