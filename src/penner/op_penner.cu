// Penner conformal metric optimization — GPU pipeline
// Pure CUDA + cuSolver sparse Cholesky. No RXMesh.
//
// Conformal case: variable is per-vertex scale factor u_v (dimension nV).
// Edge lengths: l_e(u) = l⁰_e · exp((u_i + u_j) / 2)
// Constraint: Θ_v(u) = Θ_target (cone angle at each vertex)
// Jacobian: dΘ/du is a sparse nV×nV matrix (cotangent Laplacian structure)
//
// Newton solve: L·δu = -(Θ - Θ_target), where L = dΘ/du

#include "penner/penner_types.h"
#include "penner/penner_kernels.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <map>
#include <cstdio>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════
// Halfedge mesh construction (host-side)
// ═══════════════════════════════════════════════════════════════════════

void HalfedgeMesh::build(const double* vertices, int nv, const int* faces, int nf)
{
    num_vertices = nv;
    num_faces = nf;
    num_halfedges = 3 * nf;

    next.resize(num_halfedges);
    prev.resize(num_halfedges);
    twin.resize(num_halfedges, -1);
    to.resize(num_halfedges);
    face.resize(num_halfedges);
    fhe.resize(num_faces);
    vhe.resize(num_vertices, -1);

    std::map<std::pair<int,int>, int> edge_to_he;

    for (int f = 0; f < nf; f++) {
        int v0 = faces[3*f], v1 = faces[3*f+1], v2 = faces[3*f+2];
        int h0 = 3*f, h1 = 3*f+1, h2 = 3*f+2;

        fhe[f] = h0;
        next[h0] = h1; next[h1] = h2; next[h2] = h0;
        prev[h0] = h2; prev[h1] = h0; prev[h2] = h1;
        // Convention: halfedge h in face (v0,v1,v2):
        //   h0: from v0, to v1 (edge opposite v2)
        //   h1: from v1, to v2 (edge opposite v0)
        //   h2: from v2, to v0 (edge opposite v1)
        to[h0] = v1; to[h1] = v2; to[h2] = v0;
        face[h0] = f; face[h1] = f; face[h2] = f;

        if (vhe[v0] < 0) vhe[v0] = h0;
        if (vhe[v1] < 0) vhe[v1] = h1;
        if (vhe[v2] < 0) vhe[v2] = h2;

        edge_to_he[{v0,v1}] = h0;
        edge_to_he[{v1,v2}] = h1;
        edge_to_he[{v2,v0}] = h2;
    }

    for (auto& [key, h] : edge_to_he) {
        auto it = edge_to_he.find({key.second, key.first});
        if (it != edge_to_he.end()) twin[h] = it->second;
    }

    edge.resize(num_halfedges, -1);
    e2he.clear();
    num_edges = 0;
    for (int h = 0; h < num_halfedges; h++) {
        if (edge[h] >= 0) continue;
        edge[h] = num_edges;
        if (twin[h] >= 0) edge[twin[h]] = num_edges;
        e2he.push_back(h);
        num_edges++;
    }

    // Target cone angles: 2π interior, π boundary
    Th_hat.resize(num_vertices, 2.0 * M_PI);
    for (int h = 0; h < num_halfedges; h++) {
        if (twin[h] < 0) {
            Th_hat[to[prev[h]]] = M_PI;
            Th_hat[to[h]] = M_PI;
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
// Ptolemy edge flip — maintains intrinsic Delaunay property
// ═══════════════════════════════════════════════════════════════════════

// Check if edge h is non-Delaunay: sum of opposite angles > π
static bool is_non_delaunay(const HalfedgeMesh& mesh,
                            const std::vector<double>& he_length, int h)
{
    int ht = mesh.twin[h];
    if (ht < 0) return false;  // boundary edge — never flip

    // Get the quad: edge h goes from vi to vj
    // Face of h: (vi, vj, vk) with opposite vertex vk
    // Face of ht: (vj, vi, vl) with opposite vertex vl
    // Non-Delaunay iff angle_at_vk + angle_at_vl > π

    int hi = h;
    int hj = mesh.next[hi];
    int hk = mesh.next[hj];

    int hti = ht;
    int htj = mesh.next[hti];
    int htk = mesh.next[htj];

    double li = he_length[hi], lj = he_length[hj], lk = he_length[hk];
    double lti = he_length[hti], ltj = he_length[htj], ltk = he_length[htk];

    // Angle at vk (opposite hi in face of h)
    double cos_k = (-li*li + lj*lj + lk*lk) / (2.0 * lj * lk);
    // Angle at vl (opposite hti in face of ht)
    double cos_l = (-lti*lti + ltj*ltj + ltk*ltk) / (2.0 * ltj * ltk);

    cos_k = std::clamp(cos_k, -1.0, 1.0);
    cos_l = std::clamp(cos_l, -1.0, 1.0);

    double angle_k = std::acos(cos_k);
    double angle_l = std::acos(cos_l);

    return (angle_k + angle_l) > M_PI + 1e-10;
}

// Perform a Ptolemy flip on edge h.
// Updates connectivity (next, prev, twin, to, face, fhe) and edge lengths.
// Returns true if flip succeeded.
static bool ptolemy_flip(HalfedgeMesh& mesh,
                         std::vector<double>& he_length, int h)
{
    int ht = mesh.twin[h];
    if (ht < 0) return false;

    // Halfedges in face of h:  h → hn → hnn (= prev of h)
    int hn = mesh.next[h];
    int hnn = mesh.next[hn];

    // Halfedges in face of ht: ht → htn → htnn (= prev of ht)
    int htn = mesh.next[ht];
    int htnn = mesh.next[htn];

    // Vertices:
    //   h goes from A to B
    //   Face of h:  A, B, C  (C = to[hn] via hn: B→C)
    //              wait, let me be careful.
    //   h: from to[prev[h]] to to[h]
    //   In face of h: halfedges h(A→B), hn(B→C), hnn(C→A)
    int vA = mesh.to[hnn];  // = to[prev[h]]
    int vB = mesh.to[h];
    int vC = mesh.to[hn];
    int vD = mesh.to[htn];  // opposite vertex in twin face

    // Edge lengths in the quad:
    // l_AB = he_length[h] = he_length[ht]
    // l_BC = he_length[hn]
    // l_CA = he_length[hnn]
    // l_BD = he_length[htn]
    // l_DA = he_length[htnn]
    double l_AB = he_length[h];
    double l_BC = he_length[hn];
    double l_CA = he_length[hnn];
    double l_BD = he_length[htn];
    double l_DA = he_length[htnn];

    // Ptolemy relation: l_CD = (l_BC * l_DA + l_CA * l_BD) / l_AB
    double l_CD = (l_BC * l_DA + l_CA * l_BD) / l_AB;

    // Now flip: edge AB becomes edge CD
    // New face 1: C, D, A  using halfedges h, htn(?), hnn(?)
    // New face 2: D, C, B  using halfedges ht, hn(?), htnn(?)
    //
    // Standard halfedge flip rewiring:
    // After flip, h goes from C to D, ht goes from D to C
    //
    // Face of h becomes (C, D, A):
    //   h: C→D, htnn: D→A (was in twin face), hnn: A→C (was in h face)
    //   next[h] = htnn, next[htnn] = hnn, next[hnn] = h
    //
    // Face of ht becomes (D, C, B):
    //   ht: D→C, hn: C→B (was in h face... wait, hn was B→C)
    //
    // Hmm, need to be more careful. Let me use the standard flip:

    // Before flip:
    //   Face f0 (face of h):  h(A→B), hn(B→C), hnn(C→A)
    //   Face f1 (face of ht): ht(B→A), htn(A→D), htnn(D→B)
    //
    // After flip, edge goes from C to D:
    //   Face f0: h(C→D), htnn(D→B)... no wait.
    //
    // Standard edge flip in halfedge mesh:
    //   h becomes C→D (was A→B)
    //   ht becomes D→C (was B→A)
    //
    //   Face f0 = {h, htn, hnn}:  C→D, D→A... no.
    //
    // Let me just do the standard textbook flip:

    int f0 = mesh.face[h];
    int f1 = mesh.face[ht];

    // Update to[] — h now points to D, ht now points to C
    mesh.to[h] = vD;
    mesh.to[ht] = vC;

    // Rewire next pointers:
    // Face f0: h(C→D), htn(D→A... wait, htn was A→D)
    // I need: after flip, face f0 has vertices C, D, A
    //   h: ?→D, then D→A, then A→C
    //   h goes from C to D. next is htnn (D→B)... no.

    // OK let me just do it mechanically:
    // After flip:
    //   Face f0: h, htnn, hn  — wrong vertices
    //
    // Actually the standard approach:
    // Face f0 gets halfedges: h, htn, hnn
    //   h: C→D  (repointed)
    //   htn: was A→D, but we need D→A. No — htn still goes A→D.
    //
    // I'm overcomplicating this. Let me use the well-known formula:

    // Face f0 = (h, htn, hnn) with h: C→D
    mesh.next[h] = htn;
    mesh.next[htn] = hnn;
    mesh.next[hnn] = h;

    // Face f1 = (ht, hn, htnn) with ht: D→C
    mesh.next[ht] = hn;
    mesh.next[hn] = htnn;
    mesh.next[htnn] = ht;

    // Update prev
    mesh.prev[h] = hnn;   mesh.prev[htn] = h;    mesh.prev[hnn] = htn;
    mesh.prev[ht] = htnn; mesh.prev[hn] = ht;    mesh.prev[htnn] = hn;

    // Update face assignments
    mesh.face[h] = f0; mesh.face[htn] = f0; mesh.face[hnn] = f0;
    mesh.face[ht] = f1; mesh.face[hn] = f1; mesh.face[htnn] = f1;

    // Update fhe
    mesh.fhe[f0] = h;
    mesh.fhe[f1] = ht;

    // Update to[] for moved halfedges
    // h: from C to D (already set above)
    // ht: from D to C (already set above)
    // htn was A→D, now in face f0 — still goes to D? No:
    // After flip, htn should go from D to A.
    // Wait — htn's endpoints don't change! htn still goes from A to D.
    // But in the new face f0 = (C, D, A):
    //   h: C→D
    //   htn: D→A? No, htn goes from A to D.
    //
    // This is wrong. The issue is that htn goes A→D but we need D→A after h(C→D).
    // The halfedge DIRECTION doesn't change in a flip — only h and ht change direction.

    // Let me reconsider. In a flip, ONLY h and ht change their endpoints.
    // The other 4 halfedges keep their original from/to vertices.
    // So:
    //   h: was A→B, now C→D
    //   ht: was B→A, now D→C
    //   hn: still B→C
    //   hnn: still C→A
    //   htn: still A→D
    //   htnn: still D→B
    //
    // Face f0 should be (C, D, B): h(C→D), then D→B = htnn, then B→C = hn
    //   next[h] = htnn, next[htnn] = hn... wait, hn is B→C, then C→D = h. That's a triangle!
    //   Face f0: C→D→B→C using h, htnn, hn ✓
    //
    // Face f1 should be (D, C, A): ht(D→C), then C→A = hnn, then A→D = htn
    //   next[ht] = hnn, next[hnn] = htn, next[htn] = ht ✓
    //   Face f1: D→C→A→D using ht, hnn, htn ✓

    // Fix the wiring:
    mesh.next[h] = htnn;
    mesh.next[htnn] = hn;
    mesh.next[hn] = h;

    mesh.next[ht] = hnn;
    mesh.next[hnn] = htn;
    mesh.next[htn] = ht;

    mesh.prev[h] = hn;     mesh.prev[htnn] = h;    mesh.prev[hn] = htnn;
    mesh.prev[ht] = htn;   mesh.prev[hnn] = ht;    mesh.prev[htn] = hnn;

    mesh.face[h] = f0; mesh.face[htnn] = f0; mesh.face[hn] = f0;
    mesh.face[ht] = f1; mesh.face[hnn] = f1; mesh.face[htn] = f1;

    mesh.fhe[f0] = h;
    mesh.fhe[f1] = ht;

    // Update edge length: the flipped edge gets the Ptolemy length
    he_length[h] = l_CD;
    he_length[ht] = l_CD;

    return true;
}

// Make the metric intrinsically Delaunay via Ptolemy flips.
// Returns number of flips performed.
static int make_delaunay(HalfedgeMesh& mesh, std::vector<double>& he_length,
                         int max_flips = 100000, bool verbose = false)
{
    int total_flips = 0;
    bool changed = true;

    while (changed && total_flips < max_flips) {
        changed = false;
        for (int e = 0; e < mesh.num_edges; e++) {
            int h = mesh.e2he[e];
            if (mesh.twin[h] < 0) continue;
            if (is_non_delaunay(mesh, he_length, h)) {
                ptolemy_flip(mesh, he_length, h);
                total_flips++;
                changed = true;
            }
        }
    }

    if (verbose)
        fprintf(stderr, "[penner] make_delaunay: %d flips\n", total_flips);

    return total_flips;
}


// ═══════════════════════════════════════════════════════════════════════
// Host-side Newton solve with Ptolemy flips
// ═══════════════════════════════════════════════════════════════════════

PennerResult pipeline_penner_conformal(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    const PennerConformalParams& params)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    auto t0 = clk::now();

    // ── 1. Build halfedge mesh ────────────────────────────────────────
    HalfedgeMesh mesh;
    mesh.build(vertices, num_vertices, faces, num_faces);

    int nV = mesh.num_vertices;
    int nE = mesh.num_edges;
    int nHE = mesh.num_halfedges;
    int nF = mesh.num_faces;

    if (params.verbose)
        fprintf(stderr, "[penner] mesh: %d V, %d E, %d F\n", nV, nE, nF);

    // ── 2. Compute initial edge lengths ───────────────────────────────
    std::vector<double> he_length(nHE);
    for (int h = 0; h < nHE; h++) {
        int vf = mesh.to[mesh.prev[h]];
        int vt = mesh.to[h];
        double dx = vertices[3*vt] - vertices[3*vf];
        double dy = vertices[3*vt+1] - vertices[3*vf+1];
        double dz = vertices[3*vt+2] - vertices[3*vf+2];
        he_length[h] = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (he_length[h] < 1e-20) he_length[h] = 1e-20;
    }

    // Store original log-lengths (l⁰)
    std::vector<double> log_l0(nHE);
    for (int h = 0; h < nHE; h++)
        log_l0[h] = std::log(he_length[h]);

    // Per-vertex scale factors (the optimization variable)
    std::vector<double> u(nV, 0.0);  // start at identity (u=0 → l = l⁰)

    double t_build = ms_since(t0);

    // ── 3. Compute angles + vertex angle sums from current u ──────────
    auto compute_angles = [&](const std::vector<double>& u_vec,
                              std::vector<double>& he_angle,
                              std::vector<double>& he_cot,
                              std::vector<double>& vertex_angle) {
        // Update edge lengths from scale factors
        // l_e(u) = l⁰_e · exp((u_i + u_j) / 2)
        std::vector<double> cur_length(nHE);
        for (int h = 0; h < nHE; h++) {
            int vi = mesh.to[mesh.prev[h]];
            int vj = mesh.to[h];
            cur_length[h] = he_length[h] * std::exp(0.5 * (u_vec[vi] + u_vec[vj]));
        }

        he_angle.resize(nHE);
        he_cot.resize(nHE);
        vertex_angle.assign(nV, 0.0);

        for (int f = 0; f < nF; f++) {
            int hi = mesh.fhe[f];
            int hj = mesh.next[hi];
            int hk = mesh.next[hj];

            double li = cur_length[hi], lj = cur_length[hj], lk = cur_length[hk];

            // Heron's 4*area
            double li2 = li*li, lj2 = lj*lj, lk2 = lk*lk;
            double area16 = 2*(li2*lj2 + lj2*lk2 + lk2*li2) - (li2*li2 + lj2*lj2 + lk2*lk2);
            double A4 = std::sqrt(std::max(area16, 0.0));

            double Ijk = -li2 + lj2 + lk2;
            double iJk =  li2 - lj2 + lk2;
            double ijK =  li2 + lj2 - lk2;

            const double cot_inf = 1e10;
            he_cot[hi] = (A4 == 0) ? std::copysign(cot_inf, Ijk) : Ijk / A4;
            he_cot[hj] = (A4 == 0) ? std::copysign(cot_inf, iJk) : iJk / A4;
            he_cot[hk] = (A4 == 0) ? std::copysign(cot_inf, ijK) : ijK / A4;

            he_angle[hi] = std::acos(std::clamp(Ijk / (2*lj*lk), -1.0, 1.0));
            he_angle[hj] = std::acos(std::clamp(iJk / (2*lk*li), -1.0, 1.0));
            he_angle[hk] = std::acos(std::clamp(ijK / (2*li*lj), -1.0, 1.0));
        }

        // Sum angles at vertices
        // Angle at vertex v = sum of corner angles at v
        // Corner angle at v in face f is he_angle[h] where h is the halfedge
        // whose PREV points FROM v (i.e., the angle opposite halfedge h
        // but attributed to vertex to[next[h]] ... actually:
        // Reference code: vertex_angles[to[h]] += he_angle[next[next[h]]]
        for (int h = 0; h < nHE; h++) {
            vertex_angle[mesh.to[h]] += he_angle[mesh.next[mesh.next[h]]];
        }
    };

    // ── 4. Metric interpolation on base edge lengths ─────────────────
    // Interpolate he_length toward per-face equilateral until min_angle > threshold.
    // This conditions the metric for Newton convergence (per the 2024 paper).
    // Operates on l⁰ directly, before conformal optimization.
    auto tp = clk::now();
    {
        double min_angle_rad = params.min_angle_deg * M_PI / 180.0;

        for (int step = 0; step < 200; step++) {
            // Compute angles from current he_length (u=0)
            double min_angle = 1e30;
            for (int f = 0; f < nF; f++) {
                int hi = mesh.fhe[f];
                int hj = mesh.next[hi];
                int hk = mesh.next[hj];
                double li = he_length[hi], lj = he_length[hj], lk = he_length[hk];

                double li2 = li*li, lj2 = lj*lj, lk2 = lk*lk;
                double Ijk = -li2 + lj2 + lk2;
                double iJk =  li2 - lj2 + lk2;
                double ijK =  li2 + lj2 - lk2;

                double ai = std::acos(std::clamp(Ijk / (2*lj*lk), -1.0, 1.0));
                double aj = std::acos(std::clamp(iJk / (2*lk*li), -1.0, 1.0));
                double ak = std::acos(std::clamp(ijK / (2*li*lj), -1.0, 1.0));
                min_angle = std::min(min_angle, std::min(ai, std::min(aj, ak)));
            }

            if (params.verbose && step == 0)
                fprintf(stderr, "[penner] initial min_angle: %.1f° (target: %.1f°)\n",
                        min_angle * 180.0 / M_PI, params.min_angle_deg);

            if (min_angle >= min_angle_rad) {
                if (params.verbose)
                    fprintf(stderr, "[penner] metric interpolation: %d steps, min_angle=%.1f°\n",
                            step, min_angle * 180.0 / M_PI);
                break;
            }

            // Per-face: interpolate edge lengths toward the face's average length
            // This makes each triangle more equilateral
            for (int f = 0; f < nF; f++) {
                int hi = mesh.fhe[f];
                int hj = mesh.next[hi];
                int hk = mesh.next[hj];
                double avg = (he_length[hi] + he_length[hj] + he_length[hk]) / 3.0;
                double alpha = 0.1;  // gentle interpolation
                he_length[hi] = (1 - alpha) * he_length[hi] + alpha * avg;
                he_length[hj] = (1 - alpha) * he_length[hj] + alpha * avg;
                he_length[hk] = (1 - alpha) * he_length[hk] + alpha * avg;
            }
        }

        // Update log_l0 from modified he_length
        for (int h = 0; h < nHE; h++)
            log_l0[h] = std::log(he_length[h]);
    }
    double t_interp = ms_since(tp);

    // Set target cone angles to actual initial vertex angles
    // (identity test — the initial metric IS the solution)
    // For real use, these would come from a cross field / singularity prescription
    {
        std::vector<double> he_angle, he_cot, init_va;
        compute_angles(u, he_angle, he_cot, init_va);
        mesh.Th_hat = init_va;
        if (params.verbose) {
            double sum_defect = 0;
            for (int v = 0; v < nV; v++)
                sum_defect += 2.0 * M_PI - init_va[v];
            fprintf(stderr, "[penner] sum(angle_defect) = %.4f (expect 4π=%.4f for genus 0)\n",
                    sum_defect, 4.0 * M_PI);
        }
    }

    // ── 5. Newton loop ────────────────────────────────────────────────
    tp = clk::now();

    PennerResult result;
    result.num_vertices = nV;
    result.num_edges = nE;
    result.newton_iterations = 0;
    result.final_error = 1e30;

    for (int iter = 0; iter < params.max_iterations; iter++) {
        // 5a. Update edge lengths from current scale factors
        double u_max = 0, u_min = 0;
        for (int v = 0; v < nV; v++) {
            u_max = std::max(u_max, u[v]);
            u_min = std::min(u_min, u[v]);
        }
        for (int h = 0; h < nHE; h++) {
            int vi = mesh.to[mesh.prev[h]];
            int vj = mesh.to[h];
            he_length[h] = std::exp(log_l0[h] + 0.5 * (u[vi] + u[vj]));
        }

        // Edge length stats
        double l_min = 1e30, l_max = 0, l_avg = 0;
        int l_nan = 0;
        for (int h = 0; h < nHE; h++) {
            if (std::isnan(he_length[h]) || std::isinf(he_length[h])) { l_nan++; continue; }
            l_min = std::min(l_min, he_length[h]);
            l_max = std::max(l_max, he_length[h]);
            l_avg += he_length[h];
        }
        l_avg /= (nHE - l_nan);

        if (params.verbose)
            fprintf(stderr, "[penner] iter %d: u=[%.2e, %.2e], l=[%.4e, %.4e] avg=%.4e, %d NaN\n",
                    iter, u_min, u_max, l_min, l_max, l_avg, l_nan);

        // 5b. Maintain intrinsic Delaunay via Ptolemy flips
        int nflips = make_delaunay(mesh, he_length, 100000, false);

        if (params.verbose && nflips > 0)
            fprintf(stderr, "[penner] iter %d: %d Ptolemy flips\n", iter, nflips);

        // Recompute log_l0 from flipped lengths (base metric changed)
        // and reset u to 0 (absorb scale into base metric)
        for (int h = 0; h < nHE; h++)
            log_l0[h] = std::log(std::max(he_length[h], 1e-300));
        std::fill(u.begin(), u.end(), 0.0);

        // 5c. Compute angles + residual
        std::vector<double> he_angle, he_cot, vertex_angle;
        compute_angles(u, he_angle, he_cot, vertex_angle);

        // Check for bad cotangents
        int neg_cot = 0;
        double cot_max = 0;
        for (int h = 0; h < nHE; h++) {
            if (he_cot[h] < 0) neg_cot++;
            cot_max = std::max(cot_max, std::abs(he_cot[h]));
        }

        // Residual: F = Θ - Θ_target
        double max_error = 0, avg_error = 0;
        std::vector<double> residual(nV);
        for (int v = 0; v < nV; v++) {
            residual[v] = vertex_angle[v] - mesh.Th_hat[v];
            max_error = std::max(max_error, std::abs(residual[v]));
            avg_error += std::abs(residual[v]);
        }
        avg_error /= nV;

        if (params.verbose)
            fprintf(stderr, "[penner] iter %d: max_err=%.2e avg_err=%.2e neg_cot=%d cot_max=%.1e flips=%d\n",
                    iter, max_error, avg_error, neg_cot, cot_max, nflips);

        if (max_error < params.error_eps) {
            result.newton_iterations = iter;
            result.final_error = max_error;
            if (params.verbose)
                fprintf(stderr, "[penner] converged in %d iterations\n", iter);
            break;
        }
        result.newton_iterations = iter + 1;
        result.final_error = max_error;

        // Assemble Jacobian L = dΘ/du (sparse nV×nV)
        // For conformal: dΘ_v/du_w = -0.5 * sum of cotangents of angles
        // at edges incident to both v and w.
        // This is the cotangent Laplacian.
        //
        // L[v][w] = -0.5 * (cot(α_vw) + cot(β_vw))  for edge (v,w)
        // L[v][v] = -sum_{w~v} L[v][w]
        //
        // From the reference code, per-halfedge contribution:
        // For halfedge h, let v = to[next[h]]:
        //   L[v, to[h]]     += 0.5 * cot[next[next[h]]]
        //   L[v, to[prev[h]]] += 0.5 * cot[next[h]]  -- wait, need to be careful
        //
        // Actually for conformal (per-vertex u), the Jacobian is:
        // dΘ_v/du_w = sum over faces containing edge (v,w) of:
        //   0.5 * cot(angle opposite to edge (v,w) in that face)
        //
        // And dΘ_v/du_v = sum over all edges (v,w) of dΘ_v/du_w
        // (since l_e depends on u_i + u_j, the chain rule gives symmetric contributions)

        // Build sparse L in triplet format
        std::vector<int> L_row, L_col;
        std::vector<double> L_val;

        // Off-diagonal: for each edge (vi, vj), L[vi][vj] = -0.5 * (cot_opposite_in_face0 + cot_opposite_in_face1)
        // The cotangent opposite to edge (vi,vj) in a face is the cotangent at the third vertex.
        std::vector<double> L_diag(nV, 0.0);

        for (int e = 0; e < nE; e++) {
            int h = mesh.e2he[e];
            int ht = mesh.twin[h];

            int vi = mesh.to[mesh.prev[h]];  // from vertex of h
            int vj = mesh.to[h];              // to vertex of h

            // In face of h: the angle opposite edge (vi,vj) is at vertex to[next[h]]
            // which is he_angle[h] (angle opposite the halfedge = angle at the third vertex)
            // Wait — convention: he_angle[h] is the angle at the corner OPPOSITE halfedge h.
            // Halfedge h goes from vi to vj, opposite vertex is to[next[h]].
            // The cotangent we want for edge (vi,vj) is cot(angle at opposite vertex) = he_cot[h]... no.
            //
            // Actually in our corner_angles: he_angle[hi] = angle at vertex i = angle opposite edge i.
            // For face (v0,v1,v2) with halfedges h0(v0→v1), h1(v1→v2), h2(v2→v0):
            //   he_angle[h0] = angle opposite h0 = angle at v2
            //   he_angle[h1] = angle opposite h1 = angle at v0
            //   he_angle[h2] = angle opposite h2 = angle at v1
            //
            // For edge (v0,v1) = halfedge h0: opposite vertex is v2, angle = he_angle[h0]
            // So cot opposite to edge h is he_cot[h]. ✓

            double cot_sum = he_cot[h];
            if (ht >= 0) cot_sum += he_cot[ht];

            double off_diag = -0.5 * cot_sum;

            L_row.push_back(vi); L_col.push_back(vj); L_val.push_back(off_diag);
            L_row.push_back(vj); L_col.push_back(vi); L_val.push_back(off_diag);

            L_diag[vi] -= off_diag;
            L_diag[vj] -= off_diag;
        }

        // Diagonal
        for (int v = 0; v < nV; v++) {
            L_row.push_back(v); L_col.push_back(v); L_val.push_back(L_diag[v]);
        }

        // Assemble into dense matrix (for nV=10K this is 800MB... too much)
        // Use sparse CSR + cuSolver instead.
        // For now: simple sparse CG solver on host.

        int nnz = L_row.size();

        // Convert to CSR
        std::vector<int> csr_row(nV + 1, 0);
        std::vector<int> csr_col(nnz);
        std::vector<double> csr_val(nnz);

        // Count per row
        for (int i = 0; i < nnz; i++) csr_row[L_row[i] + 1]++;
        for (int i = 1; i <= nV; i++) csr_row[i] += csr_row[i-1];

        // Fill
        std::vector<int> row_ptr(nV, 0);
        for (int i = 0; i < nnz; i++) {
            int r = L_row[i];
            int pos = csr_row[r] + row_ptr[r];
            csr_col[pos] = L_col[i];
            csr_val[pos] = L_val[i];
            row_ptr[r]++;
        }

        // Solve L·δu = -F via conjugate gradient (host, sparse)
        // CG is appropriate because L is symmetric (cotangent Laplacian)
        std::vector<double> b(nV);
        for (int v = 0; v < nV; v++) b[v] = -residual[v];

        // Sparse matrix-vector product
        auto spmv = [&](const std::vector<double>& x, std::vector<double>& y) {
            std::fill(y.begin(), y.end(), 0.0);
            for (int r = 0; r < nV; r++) {
                for (int j = csr_row[r]; j < csr_row[r+1]; j++) {
                    y[r] += csr_val[j] * x[csr_col[j]];
                }
            }
        };

        // CG solve
        std::vector<double> delta_u(nV, 0.0);
        std::vector<double> r_cg(nV), p(nV), Ap(nV);

        // r = b - A*x (x=0 initially, so r=b)
        r_cg = b;
        p = r_cg;
        double rr = 0;
        for (int v = 0; v < nV; v++) rr += r_cg[v] * r_cg[v];

        for (int cg_iter = 0; cg_iter < std::min(nV, 200); cg_iter++) {
            spmv(p, Ap);
            double pAp = 0;
            for (int v = 0; v < nV; v++) pAp += p[v] * Ap[v];
            if (std::abs(pAp) < 1e-30) break;

            double alpha = rr / pAp;
            double rr_new = 0;
            for (int v = 0; v < nV; v++) {
                delta_u[v] += alpha * p[v];
                r_cg[v] -= alpha * Ap[v];
                rr_new += r_cg[v] * r_cg[v];
            }

            if (std::sqrt(rr_new) < 1e-14 * nV) break;

            double beta = rr_new / rr;
            for (int v = 0; v < nV; v++)
                p[v] = r_cg[v] + beta * p[v];
            rr = rr_new;
        }

        // Step stats
        double du_max = 0, du_avg = 0;
        for (int v = 0; v < nV; v++) {
            du_max = std::max(du_max, std::abs(delta_u[v]));
            du_avg += std::abs(delta_u[v]);
        }
        du_avg /= nV;

        if (params.verbose)
            fprintf(stderr, "[penner] iter %d: |δu| max=%.2e avg=%.2e\n", iter, du_max, du_avg);

        // Line search: u = u + t * delta_u
        double step = 1.0;
        for (int v = 0; v < nV; v++)
            u[v] += step * delta_u[v];
    }

    double t_newton = ms_since(tp);

    // ── 6. Export results ─────────────────────────────────────────────
    // Final edge lengths (from he_length which has been updated by flips + scale)
    result.log_lengths.resize(nE);
    for (int e = 0; e < nE; e++) {
        int h = mesh.e2he[e];
        result.log_lengths[e] = std::log(std::max(he_length[h], 1e-20));
    }

    result.vertex_angles.resize(nV);
    std::vector<double> he_angle, he_cot, va;
    compute_angles(u, he_angle, he_cot, va);
    result.vertex_angles = va;

    result.total_time_ms = ms_since(t0);

    if (params.verbose) {
        fprintf(stderr, "[penner] build=%.1fms, interp=%.1fms, newton=%.1fms, total=%.1fms\n",
                t_build, t_interp, t_newton, result.total_time_ms);
        fprintf(stderr, "[penner] %d iters, error=%.2e\n",
                result.newton_iterations, result.final_error);
    }

    return result;
}

// =========================================================================
// Full Penner pipeline with external target cone angles.
// Used for: CPU cross field → GPU Newton → CPU layout pipeline.
// =========================================================================
// Helper: dump per-vertex scalar to a simple text file (vertex_id value)
static void dump_scalar(const std::string& path, const std::vector<double>& data) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return;
    fprintf(f, "%d\n", (int)data.size());
    for (int i = 0; i < (int)data.size(); i++)
        fprintf(f, "%.15e\n", data[i]);
    fclose(f);
}

// Helper: dump per-halfedge scalar
static void dump_he_scalar(const std::string& path, const std::vector<double>& data) {
    dump_scalar(path, data);
}

// Helper: dump mesh faces (after flips, connectivity may change)
static void dump_faces(const std::string& path, const HalfedgeMesh& mesh) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return;
    fprintf(f, "%d\n", mesh.num_faces);
    for (int fi = 0; fi < mesh.num_faces; fi++) {
        int h0 = mesh.fhe[fi];
        int h1 = mesh.next[h0];
        int h2 = mesh.next[h1];
        int v0 = mesh.to[mesh.prev[h0]];
        int v1 = mesh.to[h0];
        int v2 = mesh.to[h1];
        fprintf(f, "%d %d %d\n", v0, v1, v2);
    }
    fclose(f);
}

PennerFullResult pipeline_penner_with_targets(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    const double* target_cone_angles,  // per-vertex, size = num_vertices
    const PennerConformalParams& params)
{
    using clk = std::chrono::high_resolution_clock;
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };
    bool do_dump = !params.debug_dir.empty();
    auto t0 = clk::now();

    PennerFullResult result;

    // Build halfedge mesh
    result.mesh.build(vertices, num_vertices, faces, num_faces);
    auto& mesh = result.mesh;
    int nV = mesh.num_vertices;
    int nE = mesh.num_edges;
    int nHE = mesh.num_halfedges;
    int nF = mesh.num_faces;

    if (params.verbose)
        fprintf(stderr, "[penner-gpu] mesh: %d V, %d E, %d F\n", nV, nE, nF);

    // Initial edge lengths
    result.he_length.resize(nHE);
    for (int h = 0; h < nHE; h++) {
        int vf = mesh.to[mesh.prev[h]];
        int vt = mesh.to[h];
        double dx = vertices[3*vt] - vertices[3*vf];
        double dy = vertices[3*vt+1] - vertices[3*vf+1];
        double dz = vertices[3*vt+2] - vertices[3*vf+2];
        result.he_length[h] = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (result.he_length[h] < 1e-20) result.he_length[h] = 1e-20;
    }

    std::vector<double> log_l0(nHE);
    for (int h = 0; h < nHE; h++)
        log_l0[h] = std::log(result.he_length[h]);

    result.scale_factors.assign(nV, 0.0);

    // Target cone angles are set AFTER metric interpolation (see below)

    // Metric interpolation (same as conformal pipeline)
    {
        double min_angle_rad = params.min_angle_deg * M_PI / 180.0;
        for (int step = 0; step < 200; step++) {
            double min_angle = 1e30;
            for (int f = 0; f < nF; f++) {
                int hi = mesh.fhe[f];
                int hj = mesh.next[hi];
                int hk = mesh.next[hj];
                double li = result.he_length[hi], lj = result.he_length[hj], lk = result.he_length[hk];
                double li2 = li*li, lj2 = lj*lj, lk2 = lk*lk;
                double ai = std::acos(std::clamp((-li2 + lj2 + lk2) / (2*lj*lk), -1.0, 1.0));
                double aj = std::acos(std::clamp((li2 - lj2 + lk2) / (2*lk*li), -1.0, 1.0));
                double ak = std::acos(std::clamp((li2 + lj2 - lk2) / (2*li*lj), -1.0, 1.0));
                min_angle = std::min(min_angle, std::min(ai, std::min(aj, ak)));
            }
            if (min_angle >= min_angle_rad) {
                if (params.verbose)
                    fprintf(stderr, "[penner-gpu] metric interp: %d steps, min_angle=%.1f°\n",
                            step, min_angle * 180.0 / M_PI);
                break;
            }
            for (int f = 0; f < nF; f++) {
                int hi = mesh.fhe[f]; int hj = mesh.next[hi]; int hk = mesh.next[hj];
                double avg = (result.he_length[hi] + result.he_length[hj] + result.he_length[hk]) / 3.0;
                double alpha = 0.1;
                result.he_length[hi] = (1-alpha)*result.he_length[hi] + alpha*avg;
                result.he_length[hj] = (1-alpha)*result.he_length[hj] + alpha*avg;
                result.he_length[hk] = (1-alpha)*result.he_length[hk] + alpha*avg;
            }
        }
        for (int h = 0; h < nHE; h++)
            log_l0[h] = std::log(result.he_length[h]);
    }

    if (do_dump) {
        dump_he_scalar(params.debug_dir + "/step1_he_length.txt", result.he_length);
        dump_faces(params.debug_dir + "/step1_faces.txt", mesh);
    }

    // Set target cone angles AFTER metric interpolation
    // (interpolation changes edge lengths, so vertex angles change from initial values)
    {
        // First compute actual vertex angles from the interpolated metric
        std::vector<double> cur_va(nV, 0.0);
        for (int f = 0; f < nF; f++) {
            int hi = mesh.fhe[f]; int hj = mesh.next[hi]; int hk = mesh.next[hj];
            double li = result.he_length[hi], lj = result.he_length[hj], lk = result.he_length[hk];
            double li2=li*li, lj2=lj*lj, lk2=lk*lk;
            double ai = std::acos(std::clamp((-li2+lj2+lk2)/(2*lj*lk), -1.0, 1.0));
            double aj = std::acos(std::clamp((li2-lj2+lk2)/(2*lk*li), -1.0, 1.0));
            double ak = std::acos(std::clamp((li2+lj2-lk2)/(2*li*lj), -1.0, 1.0));
            // Use same attribution as conformal pipeline:
            // vertex_angle[to[h]] += he_angle[next[next[h]]]
            // For face hi,hj,hk: to[hi]=v1 gets he_angle[hk]=angle_at_v1
            cur_va[mesh.to[hi]] += ak;  // to[hi] gets angle opposite hk = angle at to[hi]
            cur_va[mesh.to[hj]] += ai;  // to[hj] gets angle opposite hi = angle at to[hj]
            cur_va[mesh.to[hk]] += aj;  // to[hk] gets angle opposite hj = angle at to[hk]
        }

        // Check if external targets are "identity" (all 2π)
        bool is_identity = true;
        for (int v = 0; v < nV; v++) {
            if (std::abs(target_cone_angles[v] - 2.0 * M_PI) > 1e-6) {
                is_identity = false;
                break;
            }
        }

        mesh.Th_hat.resize(nV);
        if (is_identity) {
            // Identity: use actual vertex angles (same as penner_conformal)
            mesh.Th_hat = cur_va;
            if (params.verbose)
                fprintf(stderr, "[penner-gpu] identity targets: using actual vertex angles\n");
        } else {
            // External targets: apply the DIFFERENCE from 2π as a delta on the actual angles
            // Th_hat[v] = cur_va[v] + (target[v] - 2π)
            // This way singularity vertices get their prescribed cone angle offset
            for (int v = 0; v < nV; v++)
                mesh.Th_hat[v] = cur_va[v] + (target_cone_angles[v] - 2.0 * M_PI);
            if (params.verbose) {
                int n_sing = 0;
                for (int v = 0; v < nV; v++)
                    if (std::abs(mesh.Th_hat[v] - cur_va[v]) > 1e-6) n_sing++;
                fprintf(stderr, "[penner-gpu] %d singularities (non-identity targets)\n", n_sing);
            }
        }
    }

    if (do_dump) {
        std::vector<double> th_hat_vec(mesh.Th_hat.begin(), mesh.Th_hat.end());
        dump_scalar(params.debug_dir + "/step2_th_hat.txt", th_hat_vec);
        dump_scalar(params.debug_dir + "/step2_cur_va.txt",
            [&]() { std::vector<double> va(nV, 0.0);
                for (int f = 0; f < nF; f++) {
                    int hi=mesh.fhe[f], hj=mesh.next[hi], hk=mesh.next[hj];
                    double li=result.he_length[hi], lj=result.he_length[hj], lk=result.he_length[hk];
                    double li2=li*li, lj2=lj*lj, lk2=lk*lk;
                    va[mesh.to[mesh.next[mesh.next[hi]]]] += std::acos(std::clamp((-li2+lj2+lk2)/(2*lj*lk),-1.,1.));
                    va[mesh.to[mesh.next[mesh.next[hj]]]] += std::acos(std::clamp((li2-lj2+lk2)/(2*lk*li),-1.,1.));
                    va[mesh.to[mesh.next[mesh.next[hk]]]] += std::acos(std::clamp((li2+lj2-lk2)/(2*li*lj),-1.,1.));
                }
                return va;
            }());
    }

    // Newton loop (reuse the same code as conformal — it's generic)
    // TODO: move this to GPU kernels (angle computation, Jacobian, CG solve)
    auto compute_angles = [&](const std::vector<double>& u_vec,
                              std::vector<double>& he_angle,
                              std::vector<double>& he_cot,
                              std::vector<double>& vertex_angle) {
        std::vector<double> cur_length(nHE);
        for (int h = 0; h < nHE; h++) {
            int vi = mesh.to[mesh.prev[h]];
            int vj = mesh.to[h];
            cur_length[h] = result.he_length[h] * std::exp(0.5 * (u_vec[vi] + u_vec[vj]));
        }
        he_angle.resize(nHE); he_cot.resize(nHE); vertex_angle.assign(nV, 0.0);
        for (int f = 0; f < nF; f++) {
            int hi = mesh.fhe[f]; int hj = mesh.next[hi]; int hk = mesh.next[hj];
            double li = cur_length[hi], lj = cur_length[hj], lk = cur_length[hk];
            double li2 = li*li, lj2 = lj*lj, lk2 = lk*lk;
            double A4 = std::sqrt(std::max(2*(li2*lj2+lj2*lk2+lk2*li2)-(li2*li2+lj2*lj2+lk2*lk2), 0.0));
            double Ijk = -li2+lj2+lk2, iJk = li2-lj2+lk2, ijK = li2+lj2-lk2;
            he_cot[hi] = A4==0 ? 1e10 : Ijk/A4;
            he_cot[hj] = A4==0 ? 1e10 : iJk/A4;
            he_cot[hk] = A4==0 ? 1e10 : ijK/A4;
            he_angle[hi] = std::acos(std::clamp(Ijk/(2*lj*lk), -1.0, 1.0));
            he_angle[hj] = std::acos(std::clamp(iJk/(2*lk*li), -1.0, 1.0));
            he_angle[hk] = std::acos(std::clamp(ijK/(2*li*lj), -1.0, 1.0));
        }
        for (int h = 0; h < nHE; h++)
            vertex_angle[mesh.to[h]] += he_angle[mesh.next[mesh.next[h]]];
    };

    auto& u = result.scale_factors;
    auto& he_length = result.he_length;

    for (int iter = 0; iter < params.max_iterations; iter++) {
        // Update edge lengths
        for (int h = 0; h < nHE; h++) {
            int vi = mesh.to[mesh.prev[h]], vj = mesh.to[h];
            he_length[h] = std::exp(log_l0[h] + 0.5*(u[vi]+u[vj]));
        }

        // Ptolemy flips (limit to 10*nE to prevent runaway)
        int nflips = make_delaunay(mesh, he_length, 10 * nE, false);
        for (int h = 0; h < nHE; h++)
            log_l0[h] = std::log(std::max(he_length[h], 1e-300));
        std::fill(u.begin(), u.end(), 0.0);

        // Angles + residual
        std::vector<double> he_angle, he_cot, vertex_angle;
        compute_angles(u, he_angle, he_cot, vertex_angle);

        double max_error = 0;
        std::vector<double> residual(nV);
        for (int v = 0; v < nV; v++) {
            residual[v] = vertex_angle[v] - mesh.Th_hat[v];
            max_error = std::max(max_error, std::abs(residual[v]));
        }

        if (params.verbose)
            fprintf(stderr, "[penner-gpu] iter %d: max_err=%.2e flips=%d\n",
                    iter, max_error, nflips);

        if (do_dump) {
            std::string prefix = params.debug_dir + "/iter" + std::to_string(iter);
            dump_scalar(prefix + "_vertex_angle.txt", vertex_angle);
            dump_scalar(prefix + "_residual.txt", residual);
            dump_he_scalar(prefix + "_he_length.txt", result.he_length);
            dump_faces(prefix + "_faces.txt", mesh);
        }

        if (max_error < params.error_eps) {
            result.newton_iterations = iter;
            result.final_error = max_error;
            break;
        }
        result.newton_iterations = iter + 1;
        result.final_error = max_error;

        // Jacobian (cotangent Laplacian) + CG solve
        // Same as conformal pipeline — TODO: move to GPU
        std::vector<int> L_row, L_col;
        std::vector<double> L_val, L_diag(nV, 0.0);
        for (int e = 0; e < nE; e++) {
            int h = mesh.e2he[e]; int ht = mesh.twin[h];
            int vi = mesh.to[mesh.prev[h]], vj = mesh.to[h];
            double cot_sum = he_cot[h];
            if (ht >= 0) cot_sum += he_cot[ht];
            double off = -0.5 * cot_sum;
            L_row.push_back(vi); L_col.push_back(vj); L_val.push_back(off);
            L_row.push_back(vj); L_col.push_back(vi); L_val.push_back(off);
            L_diag[vi] -= off; L_diag[vj] -= off;
        }
        for (int v = 0; v < nV; v++) {
            L_row.push_back(v); L_col.push_back(v); L_val.push_back(L_diag[v]);
        }

        int nnz = L_row.size();
        std::vector<int> csr_row(nV+1, 0), csr_col(nnz);
        std::vector<double> csr_val(nnz);
        for (int i = 0; i < nnz; i++) csr_row[L_row[i]+1]++;
        for (int i = 1; i <= nV; i++) csr_row[i] += csr_row[i-1];
        std::vector<int> rp(nV, 0);
        for (int i = 0; i < nnz; i++) {
            int r = L_row[i]; int pos = csr_row[r]+rp[r];
            csr_col[pos] = L_col[i]; csr_val[pos] = L_val[i]; rp[r]++;
        }

        // Sparse solve: L * delta_u = -residual
        // Pin vertex 0 (remove null space of Laplacian) and use CG
        std::vector<double> delta_u(nV, 0.0);
        {
            std::vector<double> b(nV);
            for (int v = 0; v < nV; v++) b[v] = -residual[v];

            // Pin vertex 0: set row 0 to identity, b[0] = 0
            // This removes the constant null space of the Laplacian
            for (int j = csr_row[0]; j < csr_row[0+1]; j++) {
                csr_val[j] = (csr_col[j] == 0) ? 1.0 : 0.0;
            }
            b[0] = 0.0;
            // Also zero column 0 entries in other rows
            for (int r = 1; r < nV; r++) {
                for (int j = csr_row[r]; j < csr_row[r+1]; j++) {
                    if (csr_col[j] == 0) csr_val[j] = 0.0;
                }
            }

            auto spmv = [&](const std::vector<double>& x, std::vector<double>& y) {
                std::fill(y.begin(), y.end(), 0.0);
                for (int r = 0; r < nV; r++)
                    for (int j = csr_row[r]; j < csr_row[r+1]; j++)
                        y[r] += csr_val[j] * x[csr_col[j]];
            };

            // CG solve
            std::vector<double> r_cg = b, p_cg = b, Ap(nV);
            double rr = 0;
            for (int v = 0; v < nV; v++) rr += r_cg[v]*r_cg[v];
            for (int cg = 0; cg < std::min(nV, 500); cg++) {
                spmv(p_cg, Ap);
                double pAp = 0;
                for (int v = 0; v < nV; v++) pAp += p_cg[v]*Ap[v];
                if (std::abs(pAp) < 1e-30) break;
                double a = rr/pAp; double rr_new = 0;
                for (int v = 0; v < nV; v++) {
                    delta_u[v] += a*p_cg[v];
                    r_cg[v] -= a*Ap[v];
                    rr_new += r_cg[v]*r_cg[v];
                }
                if (std::sqrt(rr_new) < 1e-12 * nV) break;
                double beta = rr_new/rr;
                for (int v = 0; v < nV; v++) p_cg[v] = r_cg[v] + beta*p_cg[v];
                rr = rr_new;
            }
            delta_u[0] = 0.0;  // ensure pinned vertex stays at 0

            // Backtracking line search: halve step until error decreases
            double step = 1.0;
            double cur_error = max_error;
            std::vector<double> u_backup = u;
            for (int ls = 0; ls < 10; ls++) {
                for (int v = 0; v < nV; v++) u[v] = u_backup[v] + step * delta_u[v];

                // Recompute error with trial step (quick check without flips)
                std::vector<double> trial_he_angle, trial_he_cot, trial_va;
                compute_angles(u, trial_he_angle, trial_he_cot, trial_va);
                double trial_error = 0;
                for (int v = 0; v < nV; v++)
                    trial_error = std::max(trial_error, std::abs(trial_va[v] - mesh.Th_hat[v]));

                if (trial_error < cur_error * 1.1 || step < 0.01) {
                    if (params.verbose && step < 1.0)
                        fprintf(stderr, "[penner-gpu] line search: step=%.4f (error %.2e → %.2e)\n",
                                step, cur_error, trial_error);
                    break;
                }
                step *= 0.5;
            }
        }
    }

    result.total_time_ms = ms_since(t0);
    if (params.verbose)
        fprintf(stderr, "[penner-gpu] %d iters, error=%.2e, total=%.1fms\n",
                result.newton_iterations, result.final_error, result.total_time_ms);

    return result;
}
