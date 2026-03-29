// Penner coordinate optimization — corner angle computation on GPU
// Computes per-corner angles and cotangents from edge lengths.
// All computation is per-face, embarrassingly parallel.

#pragma once

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/attribute.h"
#include <cmath>

namespace penner {

// Heron's formula: 16·A² = 2(a²b² + b²c² + c²a²) - (a⁴ + b⁴ + c⁴)
// Returns A² (squared area), not 16·A²
template <typename T>
__device__ inline T squared_area_from_lengths(T a, T b, T c)
{
    T a2 = a * a, b2 = b * b, c2 = c * c;
    T val = T(2) * (a2 * b2 + b2 * c2 + c2 * a2) - (a2 * a2 + b2 * b2 + c2 * c2);
    return val / T(16);
}

// Compute corner angles and cotangents from edge lengths.
// For face (i,j,k) with edges of lengths li, lj, lk (opposite vertices i, j, k):
//   angle_i = angle at vertex i = angle opposite edge i
//   Uses law of cosines: cos(α_i) = (-l²_i + l²_j + l²_k) / (2·l_j·l_k)
//
// This matches the reference implementation in constraint.cpp:corner_angles()

template <typename T>
__device__ inline void compute_face_angles(
    T li, T lj, T lk,       // edge lengths of the face
    T& angle_i, T& angle_j, T& angle_k,
    T& cot_i, T& cot_j, T& cot_k)
{
    const T cot_infty = T(1e10);

    // 4 * area via Heron's formula
    T area_sq = squared_area_from_lengths(li, lj, lk);
    T Aijk4 = T(4) * sqrt(max(area_sq, T(0)));

    // Cosine rule numerators: (-l²_i + l²_j + l²_k), etc.
    T Ijk = -li * li + lj * lj + lk * lk;
    T iJk =  li * li - lj * lj + lk * lk;
    T ijK =  li * li + lj * lj - lk * lk;

    // Cotangents
    cot_i = (Aijk4 == T(0)) ? copysign(cot_infty, Ijk) : (Ijk / Aijk4);
    cot_j = (Aijk4 == T(0)) ? copysign(cot_infty, iJk) : (iJk / Aijk4);
    cot_k = (Aijk4 == T(0)) ? copysign(cot_infty, ijK) : (ijK / Aijk4);

    // Angles via acos (clamped for numerical safety)
    angle_i = acos(min(max(Ijk / (T(2) * lj * lk), T(-1)), T(1)));
    angle_j = acos(min(max(iJk / (T(2) * lk * li), T(-1)), T(1)));
    angle_k = acos(min(max(ijK / (T(2) * li * lj), T(-1)), T(1)));
}

// Compute the minimum angle in the mesh (for metric interpolation convergence check)
template <typename T>
__device__ inline T min_angle_from_lengths(T li, T lj, T lk)
{
    T Ijk = -li * li + lj * lj + lk * lk;
    T iJk =  li * li - lj * lj + lk * lk;
    T ijK =  li * li + lj * lj - lk * lk;

    T ai = acos(min(max(Ijk / (T(2) * lj * lk), T(-1)), T(1)));
    T aj = acos(min(max(iJk / (T(2) * lk * li), T(-1)), T(1)));
    T ak = acos(min(max(ijK / (T(2) * li * lj), T(-1)), T(1)));

    return min(ai, min(aj, ak));
}

// Check if an edge is non-Delaunay (for Ptolemy flip decision).
// Edge is non-Delaunay if the sum of opposite angles > π.
// For edge e between faces (i,j,k) and (i,l,j):
//   non-Delaunay iff angle_at_k + angle_at_l > π
template <typename T>
__device__ inline bool is_non_delaunay(T angle_opposite_in_face0, T angle_opposite_in_face1)
{
    constexpr T pi = T(3.14159265358979323846);
    return (angle_opposite_in_face0 + angle_opposite_in_face1) > pi;
}

// Ptolemy relation: when flipping edge e_ij in quad (i,j,k,l),
// the new edge length is: l_kl = (l_ik·l_jl + l_il·l_jk) / l_ij
template <typename T>
__device__ inline T ptolemy_length(T l_ij, T l_ik, T l_jl, T l_il, T l_jk)
{
    return (l_ik * l_jl + l_il * l_jk) / l_ij;
}

// In log coordinates: log(l_kl) = log(exp(μ_ik)·exp(μ_jl) + exp(μ_il)·exp(μ_jk)) - μ_ij
// Use log-sum-exp for numerical stability
template <typename T>
__device__ inline T ptolemy_log_length(T mu_ij, T mu_ik, T mu_jl, T mu_il, T mu_jk)
{
    T a = mu_ik + mu_jl;
    T b = mu_il + mu_jk;
    T max_ab = max(a, b);
    return max_ab + log(exp(a - max_ab) + exp(b - max_ab)) - mu_ij;
}

} // namespace penner
