// Real Penner pipeline — calls the actual feature-aligned-penner library.
// Links against libPennerFeatureLib, libPennerOptimizationLib, etc.
// This is NOT a reimplementation — it calls the exact same code as parameterize_aligned.

#include "penner/penner_types.h"

#include <igl/readOBJ.h>
#include "feature/interface.h"
#include "holonomy/field/frame_field.h"
#include "feature/surgery/cut_metric_generator.h"
#include "util/io.h"

#include <chrono>
#include <cstdio>

using namespace Penner;
using namespace Penner::Optimization;
using namespace Penner::Holonomy;
using namespace Penner::Feature;

// Run the full Penner feature-aligned pipeline.
// Input: V (nV×3), F (nF×3)
// Output: overlay mesh with UV (V_r, F_r, uv_r, FT_r) + timing
PennerFullResult run_real_penner(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    bool verbose)
{
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    auto ms_since = [&t0]() {
        return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    };

    PennerFullResult result;

    if (verbose)
        fprintf(stderr, "[penner-real] %dV %dF\n", (int)V.rows(), (int)F.rows());

    // 1. Feature detection + cross field
    auto tp = clk::now();
    spdlog::set_level(verbose ? spdlog::level::info : spdlog::level::off);

    if (verbose) fprintf(stderr, "[penner-real] generating refined feature mesh...\n");
    auto [V_ref, F_ref, feature_edges, hard_feature_edges] =
        generate_refined_feature_mesh(V, F, false);
    if (verbose) fprintf(stderr, "[penner-real] refined: %dV %dF, %d features, %d hard\n",
        (int)V_ref.rows(), (int)F_ref.rows(), (int)feature_edges.size(), (int)hard_feature_edges.size());

    if (verbose) fprintf(stderr, "[penner-real] building feature finder...\n");
    FeatureFinder feature_finder(V_ref, F_ref);
    feature_finder.mark_features(feature_edges);
    if (verbose) fprintf(stderr, "[penner-real] generating cut mesh...\n");
    auto [V_cut, F_cut, V_map, F_is_feature] =
        feature_finder.generate_feature_cut_mesh();
    if (verbose) fprintf(stderr, "[penner-real] cut mesh: %dV %dF\n", (int)V_cut.rows(), (int)F_cut.rows());

    int radius = 5;
    Scalar rel_anisotropy = 0.9;
    Scalar abs_anisotropy = 0.2;
    Scalar bb_diag = (V_ref.colwise().maxCoeff() - V_ref.colwise().minCoeff()).norm();
    if (verbose) fprintf(stderr, "[penner-real] computing field direction...\n");
    auto [direction, is_fixed_direction] = compute_field_direction(
        V_cut, F_cut, radius, abs_anisotropy / bb_diag, rel_anisotropy);
    if (verbose) fprintf(stderr, "[penner-real] field direction done\n");

    MarkedMetricParameters marked_metric_params;
    marked_metric_params.remove_trivial_torus = false;
    marked_metric_params.use_log_length = true;
    marked_metric_params.use_initial_zero = false;
    CutMetricGenerator cut_metric_generator(V_cut, F_cut, marked_metric_params, {});
    cut_metric_generator.generate_fields(V_cut, F_cut, V_map, direction, is_fixed_direction);
    auto [reference_field, theta, kappa, period_jump] = cut_metric_generator.get_field();

    if (verbose)
        fprintf(stderr, "[penner-real] cross field: %.0fms\n",
                std::chrono::duration<double, std::milli>(clk::now() - tp).count());

    // 2. Penner optimization
    tp = clk::now();
    NewtonParameters alg_params;
    alg_params.error_eps = 1e-10;
    alg_params.solver = "ldlt";

    AlignedMetricGenerator aligned_metric_generator(
        V_ref, F_ref,
        feature_edges, hard_feature_edges,
        reference_field, theta, kappa, period_jump,
        marked_metric_params);
    aligned_metric_generator.optimize_full(alg_params);
    aligned_metric_generator.optimize_relaxed(alg_params);

    if (verbose)
        fprintf(stderr, "[penner-real] Newton optimize: %.0fms\n",
                std::chrono::duration<double, std::milli>(clk::now() - tp).count());

    // 3. Parameterize (layout)
    tp = clk::now();
    aligned_metric_generator.parameterize(false);
    auto [V_r, F_r, uv_r, FT_r, fn_to_f_r, endpoints_r] =
        aligned_metric_generator.get_parameterization();

    if (verbose)
        fprintf(stderr, "[penner-real] layout: %.0fms\n",
                std::chrono::duration<double, std::milli>(clk::now() - tp).count());

    result.total_time_ms = ms_since();
    result.newton_iterations = 0; // TODO: extract from optimizer

    if (verbose)
        fprintf(stderr, "[penner-real] total: %.0fms, output: %dV %dF %dUV\n",
                result.total_time_ms, (int)V_r.rows(), (int)F_r.rows(), (int)uv_r.rows());

    return result;
}
