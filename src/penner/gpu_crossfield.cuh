// GPU 4-RoSy cross field computation
// Computes a smooth cross field on a triangle mesh, aligned to feature edges.
// Output: per-face complex representative + per-edge matching + per-vertex singularities
//
// The cross field is the smallest eigenvector of the connection Laplacian,
// or a curvature-aligned field (sparse linear solve).

#pragma once
#include <vector>
#include <complex>
#include <cmath>

// Cross field result
struct CrossFieldResult {
    // Per-face: compressed representation (complex number, one per face)
    std::vector<std::complex<double>> compressed;  // nF

    // Per-edge: matching (0-3, rotation index across edge)
    std::vector<int> matching;  // nE

    // Per-vertex: singularity index (0 = regular, ±1, ±2)
    std::vector<int> singular;  // nV

    // Per-vertex: target cone angle (2π + singularity_index * π/2)
    std::vector<double> cone_angles;  // nV

    int num_singularities;
};

// Compute connection Laplacian for 4-RoSy field
// L[f0, f1] = -w * r^4 where r is the transport rotation across the shared edge
// L[f, f] = sum of weights
// Returns triplets (row, col, val_real, val_imag) for sparse matrix assembly
struct LaplacianTriplet {
    int row, col;
    double val_re, val_im;
};

inline void build_connection_laplacian(
    const double* vertices, int nV,
    const int* faces, int nF,
    const int* he_next, const int* he_twin, const int* he_to, const int* he_face,
    const int* fhe, int nHE,
    std::vector<LaplacianTriplet>& triplets)
{
    // For each edge (pair of twin halfedges), compute transport rotation
    // and add contribution to Laplacian
    triplets.clear();

    // Compute face basis vectors (first edge direction as basisX)
    // For face f with halfedge h: basisX = normalize(v[to[h]] - v[to[prev[h]]])
    //                              normal = cross(edge0, edge1), basisY = cross(normal, basisX)
    // Transport from face f0 to f1 across edge: angle of shared edge in each face's basis

    // Simplified: use the angle that the shared edge makes in each face's local frame
    // r = exp(i * (angle_in_f1 - angle_in_f0))
    // For 4-RoSy: contribution is r^4

    for (int h = 0; h < nHE; h++) {
        int ht = he_twin[h];
        if (ht < 0 || ht < h) continue;  // skip boundary + only process each edge once

        int f0 = he_face[h];
        int f1 = he_face[ht];
        if (f0 < 0 || f1 < 0) continue;

        // Compute edge direction in 3D
        int v_from = he_to[he_next[he_next[h]]]; // prev(h).to = tail of h
        int v_to = he_to[h];

        double ex = vertices[3*v_to] - vertices[3*v_from];
        double ey = vertices[3*v_to+1] - vertices[3*v_from+1];
        double ez = vertices[3*v_to+2] - vertices[3*v_from+2];

        // Face 0 basis: compute from face vertices
        auto face_basis = [&](int f, int fh, double& bx_x, double& bx_y, double& bx_z,
                              double& by_x, double& by_y, double& by_z,
                              double& nx, double& ny, double& nz) {
            int h0 = fh;
            int h1 = he_next[h0];
            int v0 = he_to[he_next[he_next[h0]]];
            int v1 = he_to[h0];
            int v2 = he_to[h1];

            double e0x = vertices[3*v1]-vertices[3*v0], e0y = vertices[3*v1+1]-vertices[3*v0+1], e0z = vertices[3*v1+2]-vertices[3*v0+2];
            double e1x = vertices[3*v2]-vertices[3*v0], e1y = vertices[3*v2+1]-vertices[3*v0+1], e1z = vertices[3*v2+2]-vertices[3*v0+2];

            // Normal
            nx = e0y*e1z - e0z*e1y;
            ny = e0z*e1x - e0x*e1z;
            nz = e0x*e1y - e0y*e1x;
            double nl = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (nl > 1e-15) { nx/=nl; ny/=nl; nz/=nl; }

            // basisX = normalize(e0)
            double bl = std::sqrt(e0x*e0x + e0y*e0y + e0z*e0z);
            if (bl > 1e-15) { bx_x=e0x/bl; bx_y=e0y/bl; bx_z=e0z/bl; }
            else { bx_x=1; bx_y=0; bx_z=0; }

            // basisY = cross(normal, basisX)
            by_x = ny*bx_z - nz*bx_y;
            by_y = nz*bx_x - nx*bx_z;
            by_z = nx*bx_y - ny*bx_x;
        };

        double bx0_x, bx0_y, bx0_z, by0_x, by0_y, by0_z, n0x, n0y, n0z;
        double bx1_x, bx1_y, bx1_z, by1_x, by1_y, by1_z, n1x, n1y, n1z;
        face_basis(f0, fhe[f0], bx0_x, bx0_y, bx0_z, by0_x, by0_y, by0_z, n0x, n0y, n0z);
        face_basis(f1, fhe[f1], bx1_x, bx1_y, bx1_z, by1_x, by1_y, by1_z, n1x, n1y, n1z);

        // Edge angle in each face's local frame
        double angle0 = std::atan2(ex*by0_x + ey*by0_y + ez*by0_z,
                                    ex*bx0_x + ey*bx0_y + ez*bx0_z);
        double angle1 = std::atan2(ex*by1_x + ey*by1_y + ez*by1_z,
                                    ex*bx1_x + ey*bx1_y + ez*bx1_z);

        // Transport rotation: r = exp(i * (angle1 - angle0 + π))
        // The π accounts for the twin halfedge pointing the opposite direction
        double transport_angle = angle1 - angle0 + M_PI;

        // For 4-RoSy: r^4 = exp(i * 4 * transport_angle)
        double r4_angle = 4.0 * transport_angle;
        double r4_re = std::cos(r4_angle);
        double r4_im = std::sin(r4_angle);

        double w = 1.0;  // uniform weight (could use cotangent)

        // L[f0, f0] += w
        triplets.push_back({f0, f0, w, 0.0});
        // L[f1, f1] += w
        triplets.push_back({f1, f1, w, 0.0});
        // L[f0, f1] -= w * r^4
        triplets.push_back({f0, f1, -w * r4_re, -w * r4_im});
        // L[f1, f0] -= w * conj(r^4)
        triplets.push_back({f1, f0, -w * r4_re, w * r4_im});
    }

    // Small regularization for stability
    for (int f = 0; f < nF; f++) {
        triplets.push_back({f, f, 1e-9, 0.0});
    }
}

// Compute matching from cross field (per-edge rotation index 0-3)
inline void compute_matching(
    const std::complex<double>* field,  // per-face complex representative
    const double* vertices, int nV,
    const int* faces, int nF,
    const int* he_next, const int* he_twin, const int* he_to, const int* he_face,
    const int* fhe, const int* e2he, int nE, int nHE,
    std::vector<int>& matching)
{
    matching.assign(nE, 0);

    for (int e = 0; e < nE; e++) {
        int h = e2he[e];
        int ht = he_twin[h];
        if (ht < 0) continue;

        int f0 = he_face[h];
        int f1 = he_face[ht];

        // Find the rotation that best aligns field[f0] to field[f1]
        // after transport across the edge.
        // For simplicity: compare arg(field[f1]) - arg(field[f0]) mod π/2
        std::complex<double> c0 = field[f0];
        std::complex<double> c1 = field[f1];

        // The matching is the integer k ∈ {0,1,2,3} that minimizes
        // |c1 - r^k * transport * c0| where r = exp(iπ/2)
        // Simplified: just round the angle difference to nearest π/2
        double angle_diff = std::arg(c1) - std::arg(c0);
        int m = (int)std::round(angle_diff / (M_PI / 2.0));
        matching[e] = ((m % 4) + 4) % 4;
    }
}

// Compute singularities from matching (per-vertex)
inline void compute_singularities(
    const std::vector<int>& matching,
    const int* he_next, const int* he_twin, const int* he_to,
    const int* he_edge, const int* vhe, int nV, int nE, int nHE,
    const double* angle_defect,  // per-vertex angle defect (2π - sum of corner angles)
    std::vector<int>& singular,
    std::vector<double>& cone_angles)
{
    singular.assign(nV, 0);
    cone_angles.assign(nV, 2.0 * M_PI);

    // Singularity index = round((angle_defect * 4 + Σ signed_matching * π/2) / 2π)
    for (int v = 0; v < nV; v++) {
        // Walk around vertex v using halfedges
        // For each outgoing halfedge from v, accumulate matching
        int h_start = vhe[v];
        if (h_start < 0) continue;

        double sum = angle_defect[v] * 4.0;
        int h = h_start;
        do {
            int e = he_edge[h];
            int m = matching[e];
            // Sign depends on canonical direction
            // If h is the canonical halfedge of edge e (h < twin[h])
            if (h < he_twin[h] || he_twin[h] < 0) {
                sum += m * M_PI / 2.0;
            } else {
                sum -= m * M_PI / 2.0;
            }
            // Move to next outgoing halfedge (CCW around vertex)
            int ht = he_twin[h];
            if (ht < 0) break;  // boundary
            h = he_next[ht];
        } while (h != h_start);

        singular[v] = (int)std::round(sum / (2.0 * M_PI));
        cone_angles[v] = 2.0 * M_PI - singular[v] * M_PI / 2.0;
    }
}
