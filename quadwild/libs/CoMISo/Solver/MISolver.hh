/*===========================================================================*\
 *                                                                           *
 *                               CoMISo                                      *
 *      Copyright (C) 2008-2009 by Computer Graphics Group, RWTH Aachen      *
 *                           www.rwth-graphics.de                            *
 *                                                                           *
 *---------------------------------------------------------------------------*
 *  This file is part of CoMISo.                                             *
 *                                                                           *
 *  CoMISo is free software: you can redistribute it and/or modify           *
 *  it under the terms of the GNU General Public License as published by     *
 *  the Free Software Foundation, either version 3 of the License, or        *
 *  (at your option) any later version.                                      *
 *                                                                           *
 *  CoMISo is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 *  GNU General Public License for more details.                             *
 *                                                                           *
 *  You should have received a copy of the GNU General Public License        *
 *  along with CoMISo.  If not, see <http://www.gnu.org/licenses/>.          *
 *                                                                           *
\*===========================================================================*/


//=============================================================================
//
//  CLASS MISolver
//
//=============================================================================


#ifndef COMISO_MISOLVER_HH
#define COMISO_MISOLVER_HH

//== INCLUDES =================================================================
#include <CoMISo/Config/CoMISoDefines.hh>
#include <CoMISo/Config/config.hh>

#include <CoMISo/Solver/Eigen_Tools.hh>

#include <vector>

//== NAMESPACES ===============================================================

namespace COMISO
{

class MISolverDialog;

//== CLASS DEFINITION =========================================================

/** \class MISolver MISolver.hh

    Mixed-Integer Solver.
    Approximates the solution of a (mixed-)integer problem
    by iteratively computing a continuous(real) minimizer x followed by a
    rounding of the one variable x_i which is subsequently eliminated from the
    system, and the system is solved again ...
*/
class COMISODLLEXPORT MISolver
{
public:
  typedef std::vector<int> Veci;
  typedef std::vector<double> Vecd;
  typedef std::vector<unsigned int> Vecui;

  typedef Eigen::SparseMatrix<double, Eigen::ColMajor> Matrix;
  typedef Eigen::VectorXd Vector;

  enum class RoundingType
  {
    NONE,     // no rounding at all, use with caution
    DIRECT,   /* simple method that round all integer variables in one pass,
                 fast, but solutions are far from optimal. */
    MULTIPLE, /* greedy method that rounds several variables at once,
                 controlled by set_multiple_rounding_threshold(). */
    GUROBI,   // only available if you have the Gurobi solver configured
    CPLEX,    // only available if you have the CPLEX solver configured
    DEFAULT = MULTIPLE // default setting
  };

  /// default Constructor
  MISolver();

  /// delete copy constructor
  MISolver(const MISolver&) = delete;

  /// delete assignment operator
  MISolver& operator=(const MISolver& _rhs) = delete;

  // destructor
  ~MISolver();

  /// Compute greedy approximation to a mixed integer problem.
  /** @param _A symmetric positive semi-definite CSC matrix (Will be \b
   * destroyed!)
   *  @param _x vector holding solution at the end
   *  @param _rhs right hand side of system Ax=rhs (Will be \b destroyed!)
   *  @param _to_round vector with variable indices to round to integers
   *  */
  void solve(Matrix& _A, Vector& _x, Vector& _rhs, const Veci& _to_round);

  //! Resolve using the direct solver
  void resolve(Vector& _x, Vector& _rhs);

  /// Compute greedy approximation to a mixed integer problem.
  /** @param _B mx(n+1) matrix with (still non-squared) equations of the energy,
   * including the right hand side (Will be \b destroyed!)
   *  @param _x vector holding solution at the end
   *  @param _to_round vector with variable indices to round to integers
   *  @param _fixed_order specifies if _to_round indices shall be rounded in the
   *  given order (\b true) or be greedily selected (\b false)
   *  */
  // TODO: Missing function??

  /// show Qt-Options-Dialog for setting algorithm parameters
  /** Requires a Qt Application running and COMISO_GUI to be defined */
  void show_options_dialog();

  /// Set the solve type
  void set_rounding_type(const RoundingType _rt) { rounding_type_ = _rt; }

  /// Get the solve type
  RoundingType get_rounding_type() const { return rounding_type_; }

  /// Shall no rounding be performed?
  void set_no_rounding() { rounding_type_ = RoundingType::NONE; }

  /// Shall direct (or greedy) rounding be used?
  void set_direct_rounding() { rounding_type_ = RoundingType::DIRECT; }

  /// Shall multiple rounding be performed?
  void set_multiple_rounding() { rounding_type_ = RoundingType::MULTIPLE; }

  /// Shall Gurobi solver be used?
  void set_gurobi_rounding() { rounding_type_ = RoundingType::GUROBI; }

  /// Shall CPLEX solver be used?
  void set_cplex_rounding() { rounding_type_ = RoundingType::CPLEX; }

  /** @name Get/Set functions for algorithm parameters
   * Besides being used by the Qt-Dialog these can also be called explicitly
   * to set parameters of the algorithm. */
  /*@{*/
  /// Shall an initial full solution be computed?
  void set_inital_full(bool _b) { initial_full_solution_ = _b; }
  /// Will an initial full solution be computed?
  bool get_inital_full() const { return initial_full_solution_; }

  /// Shall an full solution be computed if iterative methods did not converged?
  void set_iter_full(bool _b) { iter_full_solution_ = _b; }
  /// Will an full solution be computed if iterative methods did not converged?
  bool get_iter_full() const { return iter_full_solution_; }

  /// Shall a final full solution be computed?
  void set_final_full(bool _b) { final_full_solution_ = _b; }
  /// Will a final full solution be computed?
  bool get_final_full() const { return final_full_solution_; }

  /// Set number of maximum Gauss-Seidel iterations
  void set_local_iters(unsigned int _i) { max_local_iters_ = _i; }
  /// Get number of maximum Gauss-Seidel iterations
  unsigned int get_local_iters() { return max_local_iters_; }

  /// Set error threshold for Gauss-Seidel solver
  void set_local_error(double _d) { max_local_error_ = _d; }
  /// Get error threshold for Gauss-Seidel solver
  double get_local_error() { return max_local_error_; }

  /// Set number of maximum Conjugate Gradient iterations
  void set_cg_iters(unsigned int _i) { max_cg_iters_ = _i; }
  /// Get number of maximum Conjugate Gradient iterations
  unsigned int get_cg_iters() { return max_cg_iters_; }

  /// Set error threshold for Conjugate Gradient
  void set_cg_error(double _d) { max_cg_error_ = _d; }
  /// Get error threshold for Conjugate Gradient
  double get_cg_error() { return max_cg_error_; }

  /// Set multiple rounding threshold (upper bound of rounding performed in each
  /// iteration)
  void set_multiple_rounding_threshold(double _d)
  {
    multiple_rounding_threshold_ = _d;
  }
  /// Get multiple rounding  threshold (upper bound of rounding performed in
  /// each iteration)
  double get_multiple_rounding_threshold()
  {
    return multiple_rounding_threshold_;
  }

  /// Set time limit for gurobi solver (in seconds)
  void set_gurobi_max_time(double _d) { max_time_ = _d; }
  /// Get time limit for gurobi solver (in seconds)
  double get_gurobi_max_time() { return max_time_; }
  /*@}*/

private:
  void solve_no_rounding      (Matrix& _A, Vector& _x, Vector& _rhs);
  void solve_direct_rounding  (Matrix& _A, Vector& _x, Vector& _rhs, const Veci& _to_round);
  void solve_multiple_rounding(Matrix& _A, Vector& _x, Vector& _rhs, const Veci& _to_round);
  void solve_gurobi           (Matrix& _A, Vector& _x, Vector& _rhs, const Veci& _to_round);
  void solve_cplex            (Matrix& _A, Vector& _x, Vector& _rhs, const Veci& _to_round);

  // return true if the solution has been improved only by local iterations
  bool update_solution_is_local(
      const Matrix& _A, Vector& _x, const Vector& _rhs, const Vecui& _neigh_i);

private:
  RoundingType rounding_type_;

  // parameters used by the MiSo
  bool initial_full_solution_;
  bool iter_full_solution_;
  bool final_full_solution_;

  unsigned int max_local_iters_;
  double       max_local_error_;
  unsigned int max_cg_iters_;
  double       max_cg_error_;

  double multiple_rounding_threshold_; // control solve_multiple_rounding()
  double max_time_; // control the time limit for Gurobi and CPLEX (in seconds)

  // the actual solver declarations are hidden in the implementation code
  class DirectSolver;
  class IterativeSolver;

  DirectSolver* direct_solver_;
  IterativeSolver* iter_solver_;

  bool factorization_done_; // indicate if system factorization has been done

  // statistics
  unsigned int n_local_;
  unsigned int n_cg_;
  unsigned int n_full_;

#if(COMISO_QT_AVAILABLE)
  friend class COMISO::MISolverDialog;
#endif
};

//=============================================================================
} // namespace COMISO
//=============================================================================
//=============================================================================
#endif // COMISO_MISOLVER_HH defined
//=============================================================================
