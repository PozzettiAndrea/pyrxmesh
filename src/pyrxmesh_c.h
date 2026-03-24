/* pyrxmesh C API — for linking from non-CUDA projects (e.g. QuadWild).
 *
 * Link against libpyrxmesh_c.so (or .a) which contains all CUDA/RXMesh
 * code with device symbols already resolved. The caller never needs
 * CUDA headers, nvcc, or separable compilation.
 */

#ifndef PYRXMESH_C_H
#define PYRXMESH_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Init ────────────────────────────────────────────────────────────── */

/* Initialize CUDA device. Call once before any other function. */
void pyrxmesh_init(int device_id);

/* ── QuadWild preprocessing ──────────────────────────────────────────── */

typedef struct {
    float target_edge_length;   /* 0 = auto from mesh area + target_faces */
    int   target_faces;         /* target face count for auto edge length (default: 10000) */
    int   num_iterations;       /* outer remesh iterations (default: 3) */
    int   num_smooth_iters;     /* inner smoothing sub-iterations (default: 5) */
    int   verbose;              /* print timing breakdown to stderr */
} PyrxmeshQuadwildParams;

typedef struct {
    float*    vertices;         /* flat [x0,y0,z0, x1,...], caller must free() */
    int32_t*  faces;            /* flat [v0,v1,v2, ...], caller must free() */
    int       num_vertices;
    int       num_faces;
} PyrxmeshMeshResult;

/* Default parameters. */
PyrxmeshQuadwildParams pyrxmesh_quadwild_default_params(void);

/* GPU isotropic remeshing for QuadWild preprocessing.
 *
 * Input:  vertices[num_vertices*3] (float), faces[num_faces*3] (int32)
 * Output: result with malloc'd arrays — caller must free vertices and faces.
 * Returns 0 on success, -1 on error (message printed to stderr).
 */
int pyrxmesh_quadwild_preprocess(
    const float*    vertices, int num_vertices,
    const int32_t*  faces,    int num_faces,
    const PyrxmeshQuadwildParams* params,
    PyrxmeshMeshResult* result);

/* Free a mesh result's arrays. */
void pyrxmesh_free_result(PyrxmeshMeshResult* result);

/* ── Generic remesh (double precision) ───────────────────────────────── */

/* GPU isotropic remeshing with explicit relative_len.
 * Same as pyrxmesh.remesh() in Python.
 */
int pyrxmesh_remesh(
    const double*   vertices, int num_vertices,
    const int32_t*  faces,    int num_faces,
    double relative_len, int iterations, int smooth_iterations,
    int verbose,
    double** out_vertices, int* out_num_vertices,
    int32_t** out_faces, int* out_num_faces);

#ifdef __cplusplus
}
#endif

#endif /* PYRXMESH_C_H */
