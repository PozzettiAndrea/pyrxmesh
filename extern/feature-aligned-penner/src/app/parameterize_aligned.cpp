#include <igl/readOBJ.h>
#include "feature/interface.h"
#include "feature/core/io.h"
#include "holonomy/field/frame_field.h"
#include "util/io.h"
#include "util.h"
#include "holonomy/core/viewer.h"
#include "feature/surgery/cut_metric_generator.h"

#include "feature/surgery/refinement.h"
#include <CLI/CLI.hpp>
#include <igl/bounding_box_diagonal.h>
#include <fstream>


using namespace Penner;
using namespace Penner::Optimization;
using namespace Penner::Holonomy;
using namespace Penner::Feature;

// Helper: dump mesh + feature edges as OBJ with 'l' lines
static void dump_mesh_with_features(
    const std::string& path,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<VertexEdge>& features)
{
    std::ofstream out(path);
    for (int i = 0; i < V.rows(); i++)
        out << "v " << V(i,0) << " " << V(i,1) << " " << V(i,2) << "\n";
    for (int i = 0; i < F.rows(); i++)
        out << "f " << F(i,0)+1 << " " << F(i,1)+1 << " " << F(i,2)+1 << "\n";
    for (const auto& e : features)
        out << "l " << e[0]+1 << " " << e[1]+1 << "\n";
    spdlog::info("Wrote {} V, {} F, {} features -> {}", V.rows(), F.rows(), features.size(), path);
}

int main(int argc, char* argv[])
{
    spdlog::set_level(spdlog::level::info);

    // Get command line arguments
    CLI::App app{"Generate a feature aligned parametrization."};
    std::string mesh = "";
    std::string input_dir = "./";
    std::string output_dir = "./";

    // IO Parameters
    bool use_existing_field = false;
    bool show_parameterization = false;
    bool use_erode_dilate = true;
    bool no_erode_dilate = false;
    double dihedral_angle = 35.0;
    app.add_option("--name", mesh, "Mesh name (without obj suffix, e.g., fandisk)")->required();
    app.add_option("-i,--input", input_dir, "Input directory")->check(CLI::ExistingDirectory)->required();
    app.add_option("-o,--output", output_dir, "Output directory");
    app.add_option("--dihedral_angle", dihedral_angle, "Dihedral angle threshold for feature detection (degrees)")->default_val(35.0);
    app.add_flag("--use_existing_field", use_existing_field, "Use precomputed field at the input directory");
    app.add_flag("--show_parameterization", show_parameterization, "Show aligned parameterization");
    app.add_flag("--erode_dilate", use_erode_dilate, "Use erode/dilate for feature cleanup (default: on)");
    app.add_flag("--no_erode_dilate", no_erode_dilate, "Disable erode/dilate feature cleanup");

    // Marked Metric Parameters
    NewtonParameters alg_params;
    add_newton_parameters(app, alg_params);
    CLI11_PARSE(app, argc, argv);
    if (no_erode_dilate) use_erode_dilate = false;

    std::filesystem::create_directory(output_dir);

    // create filepaths for input data
    std::string mesh_filename = join_path(input_dir, mesh + ".obj");
    std::string feature_filename = join_path(input_dir, mesh + "_features");
    std::string hard_feature_filename = join_path(input_dir, mesh + "_hard_features");
    std::string field_filename = join_path(input_dir, mesh + ".ffield");

    // Get input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    spdlog::info("optimizing mesh at {}", mesh_filename);
    igl::readOBJ(mesh_filename, V, uv, N, F, FT, FN);

    // Get features and field
    std::vector<VertexEdge> feature_edges, hard_feature_edges;
    Eigen::MatrixXd reference_field;
    Eigen::VectorXd theta;
    Eigen::MatrixXd kappa;
    Eigen::MatrixXi period_jump;
    if (use_existing_field)
    {
        spdlog::info("loading feature edges");
        feature_edges = load_feature_edges(feature_filename);
        hard_feature_edges = load_feature_edges(hard_feature_filename);

        spdlog::info("loading constraints");
        std::tie(reference_field, theta, kappa, period_jump) = load_frame_field(field_filename);
    }
    else {
        // ── Step 1: Feature detection ──────────────────────────────
        FeatureFinder ff_detect(V, F);
        ff_detect.mark_dihedral_angle_features(dihedral_angle);
        spdlog::info("Step 1 — Feature detection ({}°): {} edges", dihedral_angle, ff_detect.get_features().size());
        dump_mesh_with_features(join_path(output_dir, mesh + "_step01_detect.obj"), V, F, ff_detect.get_features());

        // ── Step 2: Prune small components ─────────────────────────
        ff_detect.prune_small_components(4);
        spdlog::info("Step 2 — After prune_small_components: {} edges", ff_detect.get_features().size());
        dump_mesh_with_features(join_path(output_dir, mesh + "_step02_prune_comp.obj"), V, F, ff_detect.get_features());

        // ── Step 3: Prune small features ───────────────────────────
        ff_detect.prune_small_features(5);
        spdlog::info("Step 3 — After prune_small_features: {} edges", ff_detect.get_features().size());
        dump_mesh_with_features(join_path(output_dir, mesh + "_step03_prune_feat.obj"), V, F, ff_detect.get_features());

        // ── Step 4: Erode/dilate (optional) ────────────────────────
        if (use_erode_dilate) {
            ff_detect.erode_dilate(4);
            spdlog::info("Step 4 — After erode_dilate: {} edges", ff_detect.get_features().size());
            dump_mesh_with_features(join_path(output_dir, mesh + "_step04_erode_dilate.obj"), V, F, ff_detect.get_features());
        }

        // ── Step 5: Refine corner faces ────────────────────────────
        auto [V_ref_f, F_ref_f, feature_edges_f] = refine_corner_feature_faces(ff_detect);
        spdlog::info("Step 5 — After refine_corner_faces: {} V, {} F, {} features", V_ref_f.rows(), F_ref_f.rows(), feature_edges_f.size());
        dump_mesh_with_features(join_path(output_dir, mesh + "_step05_refine_corners.obj"), V_ref_f, F_ref_f, feature_edges_f);

        // ── Step 6: Refine feature components (spanning tree) ──────
        FeatureFinder ff_refined(V_ref_f, F_ref_f);
        ff_refined.mark_features(feature_edges_f);
        std::tie(V, F, feature_edges, hard_feature_edges) = refine_feature_components(ff_refined, false);
        spdlog::info("Step 6 — Refined mesh: {} V, {} F, {} features, {} hard features",
                     V.rows(), F.rows(), feature_edges.size(), hard_feature_edges.size());
        dump_mesh_with_features(join_path(output_dir, mesh + "_step06_refined.obj"), V, F, feature_edges);
        // Also dump the legacy name for compatibility
        dump_mesh_with_features(join_path(output_dir, mesh + "_refined.obj"), V, F, feature_edges);

        // ── Step 7: Generate cut mesh ──────────────────────────────
        FeatureFinder feature_finder(V, F);
        feature_finder.mark_features(feature_edges);
        auto[V_cut, F_cut, V_map, F_is_feature] = feature_finder.generate_feature_cut_mesh();
        spdlog::info("Step 7 — Cut mesh: {} V, {} F (from {} V original)", V_cut.rows(), F_cut.rows(), V.rows());

        {
            std::string cut_file = join_path(output_dir, mesh + "_step07_cut.obj");
            std::ofstream cout(cut_file);
            for (int i = 0; i < V_cut.rows(); i++)
                cout << "v " << V_cut(i,0) << " " << V_cut(i,1) << " " << V_cut(i,2) << "\n";
            for (int i = 0; i < F_cut.rows(); i++)
                cout << "f " << F_cut(i,0)+1 << " " << F_cut(i,1)+1 << " " << F_cut(i,2)+1 << "\n";
            std::map<std::pair<int,int>, int> edge_count;
            for (int fi = 0; fi < F_cut.rows(); fi++) {
                for (int j = 0; j < 3; j++) {
                    int v0 = F_cut(fi, j), v1 = F_cut(fi, (j+1)%3);
                    int a = std::min(v0,v1), b = std::max(v0,v1);
                    edge_count[{a,b}]++;
                }
            }
            std::set<std::pair<int,int>> boundary_edges;
            for (auto& [e, cnt] : edge_count) {
                if (cnt == 1) boundary_edges.insert(e);
            }
            for (const auto& [a,b] : boundary_edges)
                cout << "l " << a+1 << " " << b+1 << "\n";
            spdlog::info("Wrote cut mesh ({} boundary edges) -> {}", boundary_edges.size(), cut_file);
            // Legacy name
            std::filesystem::copy_file(cut_file, join_path(output_dir, mesh + "_cut.obj"),
                std::filesystem::copy_options::overwrite_existing);
        }

        // ── Step 8: Cross field ────────────────────────────────────
        int radius = 5;
        Scalar rel_anisotropy=0.9;
        Scalar abs_anisotropy=0.2;
        Scalar bb_diag = igl::bounding_box_diagonal(V);
        auto [direction, is_fixed_direction] = compute_field_direction(
            V_cut,
            F_cut,
            radius,
            abs_anisotropy / bb_diag,
            rel_anisotropy);
        spdlog::info("Step 8 — Cross field computed: {} faces, {} fixed directions",
                     F_cut.rows(), [&]{ int n=0; for(auto b:is_fixed_direction) if(b) n++; return n; }());

        // Dump direction field as per-face 3D vectors
        {
            std::string dir_file = join_path(output_dir, mesh + "_step08_direction.txt");
            std::ofstream dout(dir_file);
            dout << direction.rows() << "\n";
            for (int i = 0; i < direction.rows(); i++)
                dout << direction(i,0) << " " << direction(i,1) << " " << direction(i,2) << "\n";
            spdlog::info("Wrote direction field -> {}", dir_file);
        }

        // ── Step 9: Build metric generator (rotation form, Th_hat, kappa) ──
        MarkedMetricParameters marked_metric_params;
        marked_metric_params.remove_trivial_torus = false; // FIXME
        marked_metric_params.use_log_length = true;
        marked_metric_params.use_initial_zero = false;
        CutMetricGenerator cut_metric_generator(V_cut, F_cut, marked_metric_params, {});
        cut_metric_generator.generate_fields(V_cut, F_cut, V_map, direction, is_fixed_direction);
        std::tie(reference_field, theta, kappa, period_jump) = cut_metric_generator.get_field();
        spdlog::info("Step 9 — Field: {} theta, {} kappa rows, {} period_jump rows",
                     theta.size(), kappa.rows(), period_jump.rows());

        // Dump per-face component IDs (field computation decomposition)
        {
            const auto& comps = cut_metric_generator.get_components();
            int nc = comps.maxCoeff() + 1;
            std::string comp_file = join_path(output_dir, mesh + "_step09_components.txt");
            std::ofstream cout(comp_file);
            cout << comps.size() << " " << nc << "\n";
            for (int i = 0; i < comps.size(); i++) cout << comps[i] << "\n";
            spdlog::info("Wrote {} face component IDs ({} components) -> {}", comps.size(), nc, comp_file);
            // Log component sizes
            for (int c = 0; c < nc; c++) {
                int count = (comps.array() == c).count();
                if (nc <= 50 || count < 100)
                    spdlog::info("  Field component {}: {} faces", c, count);
            }
        }
        // Dump V_map (cut vertex → original vertex)
        {
            std::string vmap_file = join_path(output_dir, mesh + "_step07_vmap.txt");
            std::ofstream vout(vmap_file);
            vout << V_map.size() << "\n";
            for (int i = 0; i < V_map.size(); i++) vout << V_map[i] << "\n";
        }

        // Dump theta (per-face rotation from reference direction)
        {
            std::string theta_file = join_path(output_dir, mesh + "_step09_theta.txt");
            std::ofstream tout(theta_file);
            for (int i = 0; i < theta.size(); i++) tout << theta[i] << "\n";
            spdlog::info("Wrote theta -> {}", theta_file);
        }
        // Dump period jumps (per-edge integer rotation)
        {
            std::string pj_file = join_path(output_dir, mesh + "_step09_period_jump.txt");
            std::ofstream pout(pj_file);
            for (int i = 0; i < period_jump.rows(); i++)
                pout << period_jump(i,0) << " " << period_jump(i,1) << " " << period_jump(i,2) << "\n";
            spdlog::info("Wrote period_jump -> {}", pj_file);
        }
    }

    // ── Step 10: Newton optimization ───────────────────────────────
    spdlog::info("Step 10 — Newton optimization (projecting to feature constraints)");
    alg_params.output_dir = output_dir;
    alg_params.error_eps = 1e-10;
    alg_params.solver = "ldlt";
    MarkedMetricParameters marked_metric_params;
    AlignedMetricGenerator aligned_metric_generator(
        V,
        F,
        feature_edges,
        hard_feature_edges,
        reference_field,
        theta,
        kappa,
        period_jump,
        marked_metric_params);
    aligned_metric_generator.optimize_full(alg_params);
    spdlog::info("Step 10a — optimize_full done");
    aligned_metric_generator.optimize_relaxed(alg_params);
    spdlog::info("Step 10b — optimize_relaxed done");

    // ── Step 11: Overlay mesh + layout (parameterize) ──────────────
    aligned_metric_generator.parameterize(false);
    spdlog::info("Step 11 — Parameterization (overlay + layout) done");
    auto [V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r] = aligned_metric_generator.get_parameterization();
    auto [feature_face_edges, misaligned_edges] = aligned_metric_generator.get_refined_features();
    auto feature_edges_r = compute_face_edge_endpoints(feature_face_edges, F_r);
    auto [reference_field_r, theta_r, kappa_r, period_jump_r] = aligned_metric_generator.get_refined_field();

    if (show_parameterization) view_seamless_parameterization(V_r, F_r, uv_r, FT_r, "refined mesh", true);

    // ── Step 12: Write output (overlay mesh with UV) ───────────────
    std::string output_filename = join_path(output_dir, mesh+"_step12_opt.obj");
    write_obj_with_uv(output_filename, V_r, F_r, uv_r, FT_r);
    write_mesh_edges(output_filename, feature_edges_r);
    spdlog::info("Step 12 — Wrote overlay mesh with UV: {} V, {} F, {} UV -> {}", V_r.rows(), F_r.rows(), uv_r.rows(), output_filename);
    // Legacy name
    {
        std::string legacy = join_path(output_dir, mesh+"_opt.obj");
        write_obj_with_uv(legacy, V_r, F_r, uv_r, FT_r);
        write_mesh_edges(legacy, feature_edges_r);
    }
    output_filename = join_path(output_dir, mesh+".ffield");
    write_frame_field(output_filename,  reference_field_r, theta_r, kappa_r, period_jump_r);
    output_filename = join_path(output_dir, mesh+"_fn_to_f");
    write_vector(fn_to_f_r, output_filename);

    // ── Dump intermediate data for visualization ──────────────────────
    // Feature edges (on the refined mesh before optimization)
    {
        std::string feat_file = join_path(output_dir, mesh + "_features.txt");
        std::ofstream fout(feat_file);
        fout << feature_edges.size() << "\n";
        for (const auto& e : feature_edges)
            fout << e[0] << " " << e[1] << "\n";
        fout << "hard " << hard_feature_edges.size() << "\n";
        for (const auto& e : hard_feature_edges)
            fout << e[0] << " " << e[1] << "\n";
        spdlog::info("Wrote {} feature edges -> {}", feature_edges.size(), feat_file);
    }

    // Cross field directions + singularities on refined mesh
    {
        int nF_r = F_r.rows();
        int nV_r = V_r.rows();

        // Cross field: per-face center + rotated reference direction
        std::string cf_file = join_path(output_dir, mesh + "_crossfield.txt");
        std::ofstream cfout(cf_file);
        cfout << nF_r << "\n";
        for (int fi = 0; fi < nF_r; fi++) {
            // Face center
            Eigen::Vector3d c = (V_r.row(F_r(fi,0)) + V_r.row(F_r(fi,1)) + V_r.row(F_r(fi,2))) / 3.0;

            // Face normal
            Eigen::Vector3d e0 = V_r.row(F_r(fi,1)) - V_r.row(F_r(fi,0));
            Eigen::Vector3d e1 = V_r.row(F_r(fi,2)) - V_r.row(F_r(fi,0));
            Eigen::Vector3d n = e0.cross(e1);
            double nl = n.norm();
            if (nl > 1e-15) n /= nl;

            // Reference direction (from ffield) rotated by theta in tangent plane
            Eigen::Vector3d ref(reference_field_r(fi, 0), reference_field_r(fi, 1), reference_field_r(fi, 2));
            double ref_norm = ref.norm();
            if (ref_norm > 1e-15) ref /= ref_norm;

            // Rodrigues rotation by theta_r around normal
            double ct = std::cos(theta_r(fi)), st = std::sin(theta_r(fi));
            Eigen::Vector3d nxr = n.cross(ref);
            double ndotr = n.dot(ref);
            Eigen::Vector3d d0 = ref * ct + nxr * st + n * ndotr * (1 - ct);
            double d0n = d0.norm();
            if (d0n > 1e-15) d0 /= d0n;

            // Second direction: 90° in tangent plane
            Eigen::Vector3d d1 = n.cross(d0);
            double d1n = d1.norm();
            if (d1n > 1e-15) d1 /= d1n;

            cfout << c.x() << " " << c.y() << " " << c.z()
                  << " " << d0.x() << " " << d0.y() << " " << d0.z()
                  << " " << d1.x() << " " << d1.y() << " " << d1.z()
                  << " " << (-d0).x() << " " << (-d0).y() << " " << (-d0).z()
                  << " " << (-d1).x() << " " << (-d1).y() << " " << (-d1).z()
                  << "\n";
        }
        spdlog::info("Wrote cross field: {} faces -> {}", nF_r, cf_file);

        // Singularities from Th_hat (cone angles)
        // Access through the internal metric's Th_hat
        std::string sing_file = join_path(output_dir, mesh + "_singularities.txt");
        std::ofstream sout(sing_file);
        const auto& Th_hat_vec = aligned_metric_generator.Th_hat;
        const auto& vtx_reindex_vec = aligned_metric_generator.vtx_reindex;
        const auto& V_map_vec = aligned_metric_generator.V_map;
        const auto& V_ref = aligned_metric_generator.V;

        int n_sing = 0;
        // Count real singularities (not double-cover artifacts)
        // For closed meshes with double cover: Th_hat = 4π for regular interior vertices
        // For single sheet: Th_hat = 2π for regular vertices
        // Singularity = vertex where Th_hat is NOT a multiple of 2π that's "regular"
        double flat_angle = 2.0 * M_PI;
        // Detect double cover: if most Th_hat values are near 4π
        int n_near_4pi = 0;
        for (int i = 0; i < (int)Th_hat_vec.size(); i++)
            if (std::abs(Th_hat_vec[i] - 4.0 * M_PI) < 0.1) n_near_4pi++;
        if (n_near_4pi > (int)Th_hat_vec.size() / 2)
            flat_angle = 4.0 * M_PI;

        // Singularity = vertex where Th_hat deviates from flat_angle by more than π/4
        // (i.e., the cone angle offset is a non-zero multiple of π/2)
        // Use a tolerance that catches real singularities (offset ≥ π/2 ≈ 1.57)
        // but ignores numerical noise and boundary artifacts
        std::set<int> seen_orig;  // avoid duplicates from double cover
        std::vector<std::tuple<double,double,double,int>> sing_data;

        for (int i = 0; i < (int)Th_hat_vec.size(); i++) {
            double th = Th_hat_vec[i];
            double offset = std::abs(th - flat_angle);
            // Real singularity: offset is close to a non-zero multiple of π/2
            double nearest_half_pi = std::round(offset / (M_PI / 2.0));
            if (nearest_half_pi < 0.5) continue;  // regular vertex
            if (std::abs(offset - nearest_half_pi * M_PI / 2.0) > 0.3) continue;  // not clean

            int orig_v = (i < (int)vtx_reindex_vec.size()) ? V_map_vec[vtx_reindex_vec[i]] : i;
            if (orig_v >= (int)V_ref.rows()) continue;
            if (seen_orig.count(orig_v)) continue;  // skip double cover duplicate
            seen_orig.insert(orig_v);

            int index = (int)std::round((flat_angle - th) / (M_PI / 2.0));
            // Only real cross-field singularities: index ±1
            // (±2 are typically seam boundary artifacts in the double cover)
            if (std::abs(index) != 1) continue;
            sing_data.push_back({V_ref(orig_v, 0), V_ref(orig_v, 1), V_ref(orig_v, 2), index});
        }

        n_sing = sing_data.size();
        sout << n_sing << "\n";
        for (auto& [x, y, z, idx] : sing_data)
            sout << x << " " << y << " " << z << " " << idx << "\n";
        spdlog::info("Singularities: flat_angle={:.4f}, {} total (of {} Th_hat values)", flat_angle, n_sing, Th_hat_vec.size());

        // Also dump raw Th_hat for debugging
        {
            std::string th_file = join_path(output_dir, mesh + "_th_hat.txt");
            std::ofstream tout(th_file);
            tout << Th_hat_vec.size() << "\n";
            for (int i = 0; i < (int)Th_hat_vec.size(); i++)
                tout << Th_hat_vec[i] << "\n";
        }
        spdlog::info("Wrote {} singularities -> {}", n_sing, sing_file);
    }

    // Export intrinsic mesh with UV (connectivity after Ptolemy flips, original 3D positions)
    auto [V_intr, F_intr, uv_intr, FT_intr] = aligned_metric_generator.get_intrinsic_mesh();
    output_filename = join_path(output_dir, mesh+"_intrinsic.obj");
    write_obj_with_uv(output_filename, V_intr, F_intr, uv_intr, FT_intr);
    spdlog::info("Wrote intrinsic mesh: {} V, {} F, {} UV -> {}", V_intr.rows(), F_intr.rows(), uv_intr.rows(), output_filename);

    //std::string output_filename = join_path(output_dir, "optimized_corner_coords");
    //write_matrix(opt_corner_coords, output_filename, " ");

}
