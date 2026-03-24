// UV Parameterization wrapper (Tutte + symmetric Dirichlet energy optimization).

#include "pipeline.h"
#include <filesystem>
#include <chrono>
#include <cstdio>

#include "rxmesh/rxmesh_static.h"
#include "rxmesh/attribute.h"
#include "rxmesh/algo/tutte_embedding.h"
#include "rxmesh/diff/diff_scalar_problem.h"
#include "rxmesh/diff/newton_solver.h"
#include "rxmesh/matrix/cholesky_solver.h"
#include "rxmesh/util/import_obj.h"

#include "glm_compat.h"

using namespace rxmesh;

static char* s_param_argv[] = {(char*)"pyrxmesh", nullptr};
static struct arg {
    std::string obj_file_name;
    std::string output_folder   = "/tmp";
    std::string uv_file_name    = "";
    std::string solver          = "chol";
    uint32_t    device_id       = 0;
    float       cg_abs_tol      = 1e-6f;
    float       cg_rel_tol      = 0.0f;
    uint32_t    cg_max_iter     = 10;
    uint32_t    newton_max_iter = 100;
    char**      argv            = s_param_argv;
    int         argc            = 1;
} Arg;


template <typename T, typename ProblemT, typename SolverT>
static void run_parameterize(RXMeshStatic& rx, ProblemT& problem, SolverT& solver)
{
    NetwtonSolver newton_solver(problem, &solver);

    auto coordinates = *rx.get_input_vertex_coordinates();

    auto rest_shape = *rx.add_face_attribute<Eigen::Matrix<T, 2, 2>>("fRestShape", 1);

    // Tutte embedding as initial UV
    tutte_embedding(rx, coordinates, *problem.objective);

    constexpr uint32_t blockThreads = 256;

    // Compute rest shapes
    rx.run_query_kernel<Op::FV, blockThreads>(
        [=] __device__(const FaceHandle& fh, const VertexIterator& iter) {
            const VertexHandle v0 = iter[0];
            const VertexHandle v1 = iter[1];
            const VertexHandle v2 = iter[2];

            Eigen::Vector3<T> ar_3d = coordinates.to_eigen<3>(v0);
            Eigen::Vector3<T> br_3d = coordinates.to_eigen<3>(v1);
            Eigen::Vector3<T> cr_3d = coordinates.to_eigen<3>(v2);

            Eigen::Vector3<T> n  = (br_3d - ar_3d).cross(cr_3d - ar_3d);
            Eigen::Vector3<T> b1 = (br_3d - ar_3d).normalized();
            Eigen::Vector3<T> b2 = n.cross(b1).normalized();

            Eigen::Vector2<T> ar_2d(T(0.0), T(0.0));
            Eigen::Vector2<T> br_2d((br_3d - ar_3d).dot(b1), T(0.0));
            Eigen::Vector2<T> cr_2d((cr_3d - ar_3d).dot(b1),
                                    (cr_3d - ar_3d).dot(b2));

            Eigen::Matrix<T, 2, 2> fout = col_mat(br_2d - ar_2d, cr_2d - ar_2d);
            rest_shape(fh) = fout;
        });

    // Add symmetric Dirichlet energy term
    problem.template add_term<Op::FV, true>(
        [=] __device__(const auto& fh, const auto& iter, auto& objective) {
            assert(iter.size() == 3);

            using ActiveT = ACTIVE_TYPE(fh);

            Eigen::Vector2<ActiveT> a = iter_val<ActiveT, 2>(fh, iter, objective, 0);
            Eigen::Vector2<ActiveT> b = iter_val<ActiveT, 2>(fh, iter, objective, 1);
            Eigen::Vector2<ActiveT> c = iter_val<ActiveT, 2>(fh, iter, objective, 2);

            Eigen::Matrix<ActiveT, 2, 2> M = col_mat(b - a, c - a);

            if (M.determinant() <= 0.0) {
                using PassiveT = PassiveType<ActiveT>;
                return ActiveT(std::numeric_limits<PassiveT>::max());
            }

            const Eigen::Matrix<T, 2, 2> Mr = rest_shape(fh);
            const T A = T(0.5) * Mr.determinant();

            Eigen::Matrix<ActiveT, 2, 2> J = M * Mr.inverse();
            ActiveT res = A * (J.squaredNorm() + J.inverse().squaredNorm());
            return res;
        });

    T convergence_eps = 1e-2;

    for (uint32_t iter = 0; iter < Arg.newton_max_iter; ++iter) {
        problem.eval_terms();
        newton_solver.compute_direction();

        if (0.5f * problem.grad.dot(newton_solver.dir) < convergence_eps)
            break;

        newton_solver.line_search();
    }

    problem.objective->move(DEVICE, HOST);
}

AttributeResult pipeline_param(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int newton_iterations,
    bool verbose)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    // Write temp OBJ
    auto tmp = std::filesystem::temp_directory_path() / "pyrxmesh_param_in.obj";
    FILE* fp = fopen(tmp.string().c_str(), "w");
    for (int i = 0; i < num_vertices; ++i)
        fprintf(fp, "v %f %f %f\n", vertices[i*3], vertices[i*3+1], vertices[i*3+2]);
    for (int i = 0; i < num_faces; ++i)
        fprintf(fp, "f %d %d %d\n", faces[i*3]+1, faces[i*3+1]+1, faces[i*3+2]+1);
    fclose(fp);

    Arg.obj_file_name = tmp.string();
    Arg.newton_max_iter = static_cast<uint32_t>(newton_iterations);

    RXMeshStatic rx(tmp.string());
    std::filesystem::remove(tmp);

    if (rx.is_closed())
        throw std::runtime_error("Param requires a mesh with boundaries (not closed).");

    using T = float;
    constexpr int VariableDim = 2;
    using ProblemT = DiffScalarProblem<T, VariableDim, VertexHandle, true>;

    ProblemT problem(rx, true);  // assemble Hessian

    using HessMatT = typename ProblemT::HessMatT;
    constexpr int Order = ProblemT::DenseMatT::OrderT;

    CholeskySolver<HessMatT, Order> solver(problem.hess.get());
    run_parameterize<T>(rx, problem, solver);

    // Extract UV coordinates (2 per vertex)
    AttributeResult result;
    result.num_elements = static_cast<int>(rx.get_num_vertices());
    result.num_cols = 2;
    result.data.resize(result.num_elements * 2);

    rx.for_each_vertex(HOST, [&](const VertexHandle& vh) {
        uint32_t v_id = rx.map_to_global(vh);
        result.data[v_id * 2 + 0] = static_cast<double>((*problem.objective)(vh, 0));
        result.data[v_id * 2 + 1] = static_cast<double>((*problem.objective)(vh, 1));
    });

    if (verbose) {
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "[pyrxmesh] param: %d verts, %d newton_iters, %.1f ms\n",
                num_vertices, newton_iterations, ms);
    }
    return result;
}
