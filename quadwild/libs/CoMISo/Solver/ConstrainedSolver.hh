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
//  CLASS ConstrainedSolver
//
//=============================================================================


#ifndef COMISO_CONSTRAINEDSOLVER_HH
#define COMISO_CONSTRAINEDSOLVER_HH


//== INCLUDES =================================================================
#include <CoMISo/Config/config.hh>
#include <CoMISo/Config/CoMISoDefines.hh>

#include "GMM_Tools.hh"
#include "MISolver.hh"
#include <CoMISo/Config/StdTypes.hh>
#include <vector>

//== NAMESPACES ===============================================================

namespace COMISO
{
//== CLASS DEFINITION =========================================================

/** \class ConstrainedSolver ConstrainedSolver.hh <COMISO/.../ConstrainedSolver.hh>

  Takes a linear (symmetric) system of equations and a set of linear constraints and solves it.
 */


//#define COMISO_CONSTRAINEDSOLVER_DUMP_SYSTEMS


class COMISODLLEXPORT ConstrainedSolver
{
public:
  typedef Eigen::SparseMatrix<double, Eigen::ColMajor> ColMatrix;
  typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RowMatrix;
  typedef Eigen::VectorXd                              Vector;
  typedef Eigen::SparseVector<double>                  SparseVector;
  typedef COMISO_EIGEN::HalfSparseColMatrix<double>    HalfSparseColMatrix;
  typedef COMISO_EIGEN::HalfSparseRowMatrix<double>    HalfSparseRowMatrix;


  /// default Constructor
  /** _do_gcd specifies if a greatest common divisor correction should be used when no (+-)1-coefficient is found*/
  ConstrainedSolver(bool _do_gcd = true): do_gcd_(_do_gcd) { epsilon_ = 1e-8; }

/** @name Constrained solvers
 * Functions to solve constrained linear systems of the form Ax=b (stemming from quadratic energies).
 * The constraints can be linear constraints of the form \f$ x_1*c_1+ \cdots +x_n*c_n=c \f$ as well as integer constraints \f$x_i\in \mathbf{Z}\f$.
 * The system is then solved with these constraints in mind. For solving the system the Mixed-Integer Solver \a MISolver is used.
 */
/*@{*/

/// Quadratic matrix constrained solver
/**
  *  Takes a system of the form Ax=b, a constraint matrix C and a set of variables _to_round to be rounded to integers. \f$ A\in \mathbf{R}^{n\times n}\f$
  *  @param _constraints row matrix with rows of the form \f$ [ c_1, c_2, \cdots, c_n, c_{n+1} ] \f$ corresponding to the linear equation \f$ c_1*x_1+\cdots+c_n*x_n + c_{n+1}=0 \f$.
  *  @param _A nxn-dimensional column matrix of the system
  *  @param _x n-dimensional variable vector
  *  @param _rhs n-dimensional right hand side.
  *  @param _idx_to_round indices i of variables x_i that shall be rounded
  *  @param _reg_factor regularization factor. Helps unstable, low rank system to become solvable. Adds \f$ \_reg\_factor*mean(trace(_A))*Id \f$ to A.
  *  @param _show_miso_settings should the (QT) dialog of the Mixed-Integer solver pop up?
  */
  void solve(
      const RowMatrix&        _constraints,
      const ColMatrix&        _A,
            Vector&           _x,
            Vector&           _rhs,
            std::vector<int>& _idx_to_round,
      const double            _reg_factor = 0.0,
      const bool              _show_miso_settings = true);

  // const version of above
  void solve_const(
      const RowMatrix&        _constraints,
      const ColMatrix&        _A,
            Vector&           _x,
      const Vector&           _rhs,
      const std::vector<int>& _idx_to_round,
      const double            _reg_factor = 0.0,
      const bool              _show_miso_settings = true);

  // same as above but HalfSparse matrices as input
  void solve(
      HalfSparseRowMatrix& _constraints,
      HalfSparseColMatrix& _A,
      Vector&              _x,
      Vector&              _rhs,
      std::vector<int>&    _idx_to_round,
      double               _reg_factor = 0.0,
      bool                 _show_miso_settings = true);

#if COMISO_GMM_AVAILABLE
  // same as above but gmm matrices as input.
  // This is deprecated. Please use Eigen interface
  template<class RMatrixT, class CMatrixT, class VectorT, class VectorIT >
  void solve(
      const RMatrixT& _constraints,
      const CMatrixT& _A,
            VectorT&  _x,
      const VectorT&  _rhs,
            VectorIT& _idx_to_round,
      const double    _reg_factor = 0.0,
      const bool      _show_miso_settings = true);

  // const version of above function
  template<class RMatrixT, class CMatrixT, class VectorT, class VectorIT >
  void solve_const(
      const RMatrixT& _constraints,
      const CMatrixT& _A,
            VectorT&  _x,
      const VectorT&  _rhs,
      const VectorIT& _idx_to_round,
      const double    _reg_factor = 0.0,
      const bool      _show_miso_settings = true);
#endif // COMISO_GMM_AVAILABLE


  // efficient re-solve with modified _constraint_rhs and/or _rhs (if not
  //  modified use 0 pointer) by keeping previous _constraints and _A fixed
  // _constraint_rhs and _rhs are constant, i.e. not changed
  void resolve(
            Vector&  _x,
      const Vector*  _constraint_rhs = nullptr,
      const Vector*  _rhs            = nullptr);

#if COMISO_GMM_AVAILABLE
  // same as above but gmm interface.
  // This is deprecated. Please use Eigen interface
  template <class VectorT>
  void resolve(
            VectorT& _x,
      const VectorT* _constraint_rhs = nullptr,
      const VectorT* _rhs            = nullptr);
#endif // COMISO_GMM_AVAILABLE


/// Non-Quadratic matrix constrained solver
/**
  *  Same as above, but performs the elimination of the constraints directly on the B matrix of \f$ x^\top B^\top Bx \f$, where B has m rows (equations) and (n+1) columns \f$ [ x_1, x_2, \cdots, x_n, -rhs ] \f$.
  *  \note This function might be more efficient in some cases, but generally the solver for the quadratic matrix above is a safer bet. Needs further testing.
  *  \note Internally the \f$ A=B^\top B \f$ matrix is formed.
  *  @param _constraints row matrix with rows of the form \f$ [ c_1, c_2, \cdots, c_n, c_{n+1} ] \f$ corresponding to the linear equation \f$ c_1*x_1+\cdots+c_n*x_n + c_{n+1}=0 \f$.
  *  @param _B mx(n+1)-dimensional column matrix of the system
  *  @param _x n-dimensional variable vector
  *  @param _idx_to_round indices i of variables x_i that shall be rounded
  *  @param _reg_factor regularization factor. Helps unstable, low rank system to become solvable.
  *  @param _show_miso_settings should the (QT) dialog of the Mixed-Integer solver pop up?
  */
  void solve(
      const RowMatrix&        _constraints,
      const RowMatrix&        _B,
            Vector&           _x,
            std::vector<int>& _idx_to_round,
      const double            _reg_factor = 0.0,
      const bool              _show_miso_settings = true);

  // same as above but work on HalfSparse matrices
  void solve(
      HalfSparseRowMatrix& _constraints,
      HalfSparseColMatrix& _B,
      Vector&              _x,
      std::vector<int>&    _idx_to_round,
      const double         _reg_factor = 0.0,
      const bool           _show_miso_settings = true);

#if COMISO_GMM_AVAILABLE
  // Same as above but deprecated gmm interface
  template<class RMatrixT, class VectorT, class VectorIT >
  void solve(
      const RMatrixT& _constraints,
      const RMatrixT& _B,
            VectorT&  _x,
            VectorIT& _idx_to_round,
      const double    _reg_factor = 0.0,
      const bool      _show_miso_settings = true);
#endif // COMISO_GMM_AVAILABLE


  // const version of above function
  void solve_const(
      const RowMatrix&        _constraints,
      const RowMatrix&        _B,
            Vector&           _x,
      const std::vector<int>& _idx_to_round,
      const double            _reg_factor = 0.0,
      const bool              _show_miso_settings = true);


#if COMISO_GMM_AVAILABLE
  // Same as above but deprecated gmm interface
  template<class RMatrixT, class VectorT, class VectorIT >
  void solve_const(
      const RMatrixT& _constraints,
      const RMatrixT& _B,
            VectorT&  _x,
      const VectorIT& _idx_to_round,
      const double    _reg_factor = 0.0,
      const bool      _show_miso_settings = true);
#endif // COMISO_GMM_AVAILABLE

  // efficient re-solve with modified _rhs by keeping previous _constraints and _A fixed
  // ATTENTION: only the rhs resulting from B^TB can be changed!!! otherwise use solve
  void resolve(
    const RowMatrix& _B,
          Vector&    _x,
    const Vector*    _constraint_rhs = 0);

#if COMISO_GMM_AVAILABLE
  // Same as above but deprecated gmm interface
  template<class RMatrixT, class VectorT >
  void resolve(
    const RMatrixT& _B,
          VectorT&  _x,
    const VectorT*  _constraint_rhs = 0);
#endif // COMISO_GMM_AVAILABLE

/*@}*/

  /// Set numerical epsilon for valid constraint coefficient
  void set_epsilon(double _epsilon) { epsilon_ = _epsilon; }

  // Set whether the constraint reordering is used (default true)
  void set_use_constraint_reordering(bool _use)
  {
    use_constraint_reordering_ = _use;
  }

  /// Access the MISolver (e.g. to change settings)
  COMISO::MISolver& misolver() { return miso_; }

private:

  /** @name Eliminate constraints
 * Functions to eliminate (or integrate) linear constraints from an equation system. These functions are used internally by the \a solve functions.
 */
/*@{*/

/// Make constraints independent
/**
  *  This function performs a Gauss elimination on the constraint matrix making the constraints easier to eliminate.
  *  \note A certain amount of independence of the constraints is assumed.
  *  \note contradicting constraints will be ignored.
  *  \warning care must be taken when non-trivial constraints occur where some of the variables contain integer-variables (to be rounded) as the optimal result might not always occur.
  *  @param _constraints  row matrix with constraints
  *  @param _idx_to_round indices of variables to be rounded (these must be considered.)
  *  @param _c_elim the "returned" vector of variable indices and the order in which the can be eliminated.
  */
  void make_constraints_independent(
            HalfSparseRowMatrix& _constraints,
			const std::vector<int>&    _idx_to_round,
			      std::vector<int>&    _c_elim );


/// Eliminate constraints on a factored matrix B
/**
  *  \note Constraints are assumed to have been made independent by \a make_constraints_independent.
  *  @param _constraints row matrix with constraints (n+1 columns)
  *  @param _B system row matrix mx(n+1)
  *  @param _idx_to_round indices to be rounded
  *  @param _c_elim the indices of the variables to be eliminated.
  *  @param _new_idx the created re-indexing map. new_idx[i] = -1 means x_i eliminated, new_idx[i] = j means x_i is now at index j.
  *  @param _Bcol resulting (smaller) column matrix to be used for future computations. (e.g. convert to CSC and solve)
  */
  void eliminate_constraints(
      const HalfSparseRowMatrix& _constraints,
            HalfSparseColMatrix& _Bcol,
			      std::vector<int>&    _idx_to_round,
			const std::vector<int>&    _c_elim,
			      std::vector<int>&    _new_idx);

/// Eliminate constraints on a quadratic matrix A
/**
  *  \note Constraints are assumed to have been made independent by \a make_constraints_independent.
  *  \note _x must have correct size (same as _rhs)
  *  @param _constraints row matrix with constraints (n+1 columns)
  *  @param _A system row matrix nxn)
  *  @param _x variable vector
  *  @param _rhs right hand side
  *  @param _idx_to_round indices to be rounded
  *  @param _c_elim the indices of the variables to be eliminated.
  *  @param _new_idx the created re-indexing map. new_idx[i] = -1 means x_i eliminated, new_idx[i] = j means x_i is now at index j.
  *  @param _Acsc resulting (smaller) column (csc) matrix to be used for future computations.
  */
  void eliminate_constraints(
      const HalfSparseRowMatrix& _constraints,
            HalfSparseColMatrix& _A,
            Vector&              _x,
            Vector&              _rhs,
            std::vector<int>&    _idx_to_round,
      const std::vector<int>&    _c_elim,
            std::vector<int>&    _new_idx,
            ColMatrix&           _Acsc);

/// Restore a solution vector to the un-eliminated size
/**
  *  @param _constraints row matrix with constraints (n+1 columns)
  *  @param _x solution vector to reduced/eliminated system (result will also be written here)
  *  @param _c_elim vector of eliminated indices
  *  @param _new_idx re-indexing vector
  */
  void restore_eliminated_vars(
        const HalfSparseRowMatrix& _constraints,
				      Vector&               _x,
				const std::vector<int>&    _c_elim,
				const std::vector<int>&    _new_idx);


  // add _coeff * (_source_row of _source_mat)  to _target_row of _target_mat
  void add_row(
          Eigen::Index  _target_row,
          double        _coeff,
    const RowMatrix&    _source_mat,
          Eigen::Index  _source_row,
          ColMatrix&    _target_mat );

  // add _coeff * (_source_row of _source_mat)  to _target_row of _target_rmat and target_cmat.
  // set element in _zero_col to 0 if it exists
  void add_row_simultaneously(
    const Eigen::Index         _target_row,
    const double               _coeff,
    const HalfSparseRowMatrix& _source_mat,
    const Eigen::Index         _source_row,
          HalfSparseRowMatrix& _target_rmat,
          HalfSparseColMatrix& _target_cmat,
    const Eigen::Index         _zero_col = -1);

  /// Support changing the RHS of the constraint system in resolve(). Enabled by default.
  /// If this is needed, it must be enabled for the initial solve, not just before the resolve!
  /// Warning: This can impose substantial memory overhead for large sparse constraint systems.
  void set_support_constraint_rhs_resolve(bool _val)
  {
    support_constraint_rhs_resolve_ = _val;
    if (!_val)
    {
      // Disabling support means we don't need the content of D_ anymore.
      this->rhs_update_table_.D_ = {};
    }
  }


  // warning: order of replacement not the same as in _columns (internal sort)
  void eliminate_columns(HalfSparseColMatrix& _M, const std::vector<int>& _columns);

  inline int gcd( int _a, int _b)
  {
    while( _b != 0)
    {
      int t(_b);
      _b = _a%_b;
      _a = t;
    }
    return _a;
  }


private:

  /// Copy constructor (not used)
  ConstrainedSolver(const ConstrainedSolver& _rhs);

  /// Assignment operator (not used)
  ConstrainedSolver& operator=(const ConstrainedSolver& _rhs);

  // MISO solver
  COMISO::MISolver miso_;

  double epsilon_;
  bool   do_gcd_;
  bool   use_constraint_reordering_ = true;

#ifdef COMISO_CONSTRAINEDSOLVER_DUMP_SYSTEMS
  // count number of resolves
  int n_resolves_ = 0;
#endif

  // User-configurable, whether to store information for constraint-rhs resolve:
  bool   support_constraint_rhs_resolve_ = true;

  // --------------- Update by Marcel to enable efficient re-solve with changed rhs ----------------------
    // Store for symbolic elimination information for rhs
  class RHSUpdateTable
  {
  public:
    RHSUpdateTable()
        : c_elim_(), new_idx_(), constraints_(), D_(), cur_rhs_(),
          cur_constraint_rhs_()
    {}

    void append(int _i, double _f, int _j, bool _flag)
    {
      if (_f != 0.0)
        table_.emplace_back(_i, _j, _f, _flag);
    }

    void add_elim_id(int _i) { elim_var_ids_.push_back(_i); }

    void clear()
    {
      table_.clear();
      elim_var_ids_.clear();
    }

    // apply stored transformations to _rhs
    void apply(Vector& _constraint_rhs, Vector& _rhs);

    // remove eliminated elements from _rhs
    void eliminate(Vector& _rhs);

    // store transformed constraint matrix and index map to allow for later
    // re-substitution
    void store(
      const HalfSparseRowMatrix& _constraints,
      const std::vector<int>&    _c_elim,
      const std::vector<int>&    _new_idx)
    {
      constraints_ = _constraints;
      c_elim_ = _c_elim;
      new_idx_ = _new_idx;
    }

  private:
    class RHSUpdateTableEntry
    {
    public:
      RHSUpdateTableEntry(int _i, int _j, double _f, bool _rhs_flag)
          : i(_i), j(_j), f(_f), rhs_flag(_rhs_flag)
      {
      }

      int i;
      int j;
      double f;
      bool rhs_flag;
    };

    std::vector<RHSUpdateTableEntry> table_;
    std::vector<int> elim_var_ids_;

  public:
    std::vector<int> c_elim_;
    std::vector<int> new_idx_;
    HalfSparseRowMatrix constraints_;

    // cache current rhs_ and constraint_rhs_ and linear transformation of
    // constraint_rhs_ D_
    RowMatrix D_;
    Vector cur_rhs_;
    // constraint_rhs after Gaussian elimination update D*constraint_rhs_orig_
    Vector cur_constraint_rhs_;

  } rhs_update_table_;
};


//=============================================================================
} // namespace COMISO
//=============================================================================
#if defined(INCLUDE_TEMPLATES) && !defined(COMISO_CONSTRAINEDSOLVER_C)
#define COMISO_CONSTRAINEDSOLVER_TEMPLATES
#include "ConstrainedSolverT_impl.hh"
#endif
//=============================================================================
#endif // COMISO_CONSTRAINEDSOLVER_HH defined
//=============================================================================

