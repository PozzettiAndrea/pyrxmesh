// Penner coordinate optimization — CUDA kernels
// Pure CUDA, no RXMesh. Operates on flat halfedge arrays.

#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace penner {

// ═══════════════════════════════════════════════════════════════════════
// Kernel 1: Compute corner angles + cotangents from edge lengths
// One thread per face. Reads 3 halfedge log-lengths, writes 3 angles + 3 cots.
// ═══════════════════════════════════════════════════════════════════════

__global__ void k_corner_angles(
    int num_faces,
    const int* __restrict__ fhe,          // [num_faces] one halfedge per face
    const int* __restrict__ next,         // [num_halfedges]
    const double* __restrict__ log_length,// [num_halfedges] log(l_e)
    double* __restrict__ he_angle,        // [num_halfedges] output: corner angle
    double* __restrict__ he_cot)          // [num_halfedges] output: cotangent
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    // Get the 3 halfedges of this face
    int hi = fhe[f];
    int hj = next[hi];
    int hk = next[hj];

    // Edge lengths from log-lengths
    double li = exp(log_length[hi]);
    double lj = exp(log_length[hj]);
    double lk = exp(log_length[hk]);

    // Heron's formula for 4*area
    double li2 = li * li, lj2 = lj * lj, lk2 = lk * lk;
    double area16 = 2.0 * (li2 * lj2 + lj2 * lk2 + lk2 * li2) -
                    (li2 * li2 + lj2 * lj2 + lk2 * lk2);
    double Aijk4 = sqrt(fmax(area16, 0.0));

    // Cosine rule numerators
    double Ijk = -li2 + lj2 + lk2;
    double iJk =  li2 - lj2 + lk2;
    double ijK =  li2 + lj2 - lk2;

    // Cotangents
    const double cot_inf = 1e10;
    he_cot[hi] = (Aijk4 == 0.0) ? copysign(cot_inf, Ijk) : (Ijk / Aijk4);
    he_cot[hj] = (Aijk4 == 0.0) ? copysign(cot_inf, iJk) : (iJk / Aijk4);
    he_cot[hk] = (Aijk4 == 0.0) ? copysign(cot_inf, ijK) : (ijK / Aijk4);

    // Angles via acos
    he_angle[hi] = acos(fmin(fmax(Ijk / (2.0 * lj * lk), -1.0), 1.0));
    he_angle[hj] = acos(fmin(fmax(iJk / (2.0 * lk * li), -1.0), 1.0));
    he_angle[hk] = acos(fmin(fmax(ijK / (2.0 * li * lj), -1.0), 1.0));
}


// ═══════════════════════════════════════════════════════════════════════
// Kernel 2: Sum corner angles at vertices → vertex cone angles
// Scatter-add: each halfedge contributes its corner angle to its tip vertex.
// ═══════════════════════════════════════════════════════════════════════

__global__ void k_vertex_angles(
    int num_halfedges,
    const int* __restrict__ next,
    const int* __restrict__ to,
    const double* __restrict__ he_angle,
    double* __restrict__ vertex_angle)    // [num_vertices] output, must be zeroed first
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_halfedges) return;

    // The angle at the tip vertex of halfedge h is stored at he_angle[prev(prev(h))]
    // which is he_angle[next[next[h]]] since prev = next∘next in a triangle mesh.
    // But actually: angle at vertex to[h] is the angle of the corner at to[h],
    // which is the angle opposite the edge across from to[h].
    // In the reference code: vertex_angles[v_rep[to[h]]] += he2angle[next[next[h]]]
    int v = to[h];
    int hk = next[next[h]];  // the halfedge whose angle is at vertex v
    atomicAdd(&vertex_angle[v], he_angle[hk]);
}


// ═══════════════════════════════════════════════════════════════════════
// Kernel 3: Compute constraint residual F = Θ - Θ_target
// One thread per vertex.
// ═══════════════════════════════════════════════════════════════════════

__global__ void k_constraint_residual(
    int num_vertices,
    const double* __restrict__ vertex_angle,
    const double* __restrict__ Th_hat,
    double* __restrict__ residual)         // [num_vertices] output
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    residual[v] = vertex_angle[v] - Th_hat[v];
}


// ═══════════════════════════════════════════════════════════════════════
// Kernel 4: Assemble Jacobian triplets (COO format)
// Each halfedge h contributes 3 entries to J = dΘ/dμ.
// From the reference (constraint.cpp:vertex_angles_with_jacobian_helper):
//   J[v_rep[to[next[h]]], next[next[h]]] += -0.5 * cot[next[h]]
//   J[v_rep[to[next[h]]], h]              +=  0.5 * cot[next[h]] + 0.5 * cot[next[next[h]]]
//   J[v_rep[to[next[h]]], next[h]]        += -0.5 * cot[next[next[h]]]
//
// For the conformal case with Euclidean coordinates, the reduced coordinate
// Jacobian is per-edge (not per-halfedge), so we need he→edge mapping.
// For now, output in halfedge COO format; reduction happens on host or in cuSPARSE.
// ═══════════════════════════════════════════════════════════════════════

__global__ void k_jacobian_triplets(
    int num_halfedges,
    const int* __restrict__ next,
    const int* __restrict__ to,
    const int* __restrict__ edge,
    const double* __restrict__ he_cot,
    int* __restrict__ row,                // [3 * num_halfedges] output COO row
    int* __restrict__ col,                // [3 * num_halfedges] output COO col
    double* __restrict__ val)             // [3 * num_halfedges] output COO val
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_halfedges) return;

    int hn = next[h];
    int hnn = next[hn];
    int v = to[hn];  // vertex at which this contribution acts

    // Edge indices for reduced coordinates
    int e_hnn = edge[hnn];
    int e_h   = edge[h];
    int e_hn  = edge[hn];

    double cot_n  = he_cot[hn];
    double cot_nn = he_cot[hnn];

    int base = 3 * h;

    // Entry 1: J[v, edge(next(next(h)))] -= 0.5 * cot(next(h))
    row[base + 0] = v;
    col[base + 0] = e_hnn;
    val[base + 0] = -0.5 * cot_n;

    // Entry 2: J[v, edge(h)] += 0.5 * cot(next(h)) + 0.5 * cot(next(next(h)))
    row[base + 1] = v;
    col[base + 1] = e_h;
    val[base + 1] = 0.5 * cot_n + 0.5 * cot_nn;

    // Entry 3: J[v, edge(next(h))] -= 0.5 * cot(next(next(h)))
    row[base + 2] = v;
    col[base + 2] = e_hn;
    val[base + 2] = -0.5 * cot_nn;
}


// ═══════════════════════════════════════════════════════════════════════
// Kernel 5: Metric interpolation toward equilateral
// Scales log-lengths toward the average: μ[h] = (1-α)·μ[h] + α·μ_avg
// This conditions the metric for Newton convergence.
// ═══════════════════════════════════════════════════════════════════════

__global__ void k_metric_interpolation(
    int num_halfedges,
    double alpha,                          // interpolation parameter (0.1 = gentle)
    double mu_avg,                         // average log-length (target)
    double* __restrict__ log_length)       // [num_halfedges] in/out
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_halfedges) return;
    log_length[h] = (1.0 - alpha) * log_length[h] + alpha * mu_avg;
}


// ═══════════════════════════════════════════════════════════════════════
// Kernel 6: Compute minimum angle across all faces (reduction)
// Returns per-face min angle; host does final reduction.
// ═══════════════════════════════════════════════════════════════════════

__global__ void k_min_angle_per_face(
    int num_faces,
    const int* __restrict__ fhe,
    const int* __restrict__ next,
    const double* __restrict__ he_angle,
    double* __restrict__ face_min_angle)   // [num_faces] output
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;

    int hi = fhe[f];
    int hj = next[hi];
    int hk = next[hj];

    face_min_angle[f] = fmin(he_angle[hi], fmin(he_angle[hj], he_angle[hk]));
}


// ═══════════════════════════════════════════════════════════════════════
// Kernel 7: Apply Newton step: μ = μ + t * δμ
// δμ is per-edge; need to scatter to both halfedges of each edge.
// ═══════════════════════════════════════════════════════════════════════

__global__ void k_apply_step(
    int num_halfedges,
    const int* __restrict__ edge,
    double step_size,
    const double* __restrict__ delta_mu,   // [num_edges] Newton direction
    double* __restrict__ log_length)       // [num_halfedges] in/out
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_halfedges) return;
    log_length[h] += step_size * delta_mu[edge[h]];
}


// ═══════════════════════════════════════════════════════════════════════
// Kernel 8: Compute L2 norm of residual (partial reduction)
// ═══════════════════════════════════════════════════════════════════════

__global__ void k_residual_norm_sq(
    int num_vertices,
    const double* __restrict__ residual,
    double* __restrict__ partial_sum)      // [gridDim.x] output
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < num_vertices) ? residual[i] * residual[i] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sum[blockIdx.x] = sdata[0];
}

} // namespace penner
