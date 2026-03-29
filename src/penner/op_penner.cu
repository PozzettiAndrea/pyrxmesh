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
// Host-side Newton solve (sparse, using Eigen-like COO assembly + cuSolver)
// For the conformal case, the system is nV×nV — small enough for host solve
// on 10K meshes (~100M ops per iteration). Will migrate to cuSolver later.
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

    // ── 5. Newton loop ────────────────────────────────────────────────
    tp = clk::now();

    PennerResult result;
    result.num_vertices = nV;
    result.num_edges = nE;
    result.newton_iterations = 0;
    result.final_error = 1e30;

    for (int iter = 0; iter < params.max_iterations; iter++) {
        std::vector<double> he_angle, he_cot, vertex_angle;
        compute_angles(u, he_angle, he_cot, vertex_angle);

        // Residual: F = Θ - Θ_target
        double max_error = 0;
        std::vector<double> residual(nV);
        for (int v = 0; v < nV; v++) {
            residual[v] = vertex_angle[v] - mesh.Th_hat[v];
            max_error = std::max(max_error, std::abs(residual[v]));
        }

        if (params.verbose)
            fprintf(stderr, "[penner] iter %d: max_error=%.2e\n", iter, max_error);

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

        // Line search: u = u + t * delta_u
        double step = 1.0;
        for (int v = 0; v < nV; v++)
            u[v] += step * delta_u[v];
    }

    double t_newton = ms_since(tp);

    // ── 6. Export results ─────────────────────────────────────────────
    // Compute final log-lengths per edge
    result.log_lengths.resize(nE);
    for (int e = 0; e < nE; e++) {
        int h = mesh.e2he[e];
        int vi = mesh.to[mesh.prev[h]];
        int vj = mesh.to[h];
        result.log_lengths[e] = log_l0[h] + 0.5 * (u[vi] + u[vj]);
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
