// Copyright 2022 Autodesk, Inc. All rights reserved.


//=============================================================================
//
//  CLASS MultiDimConstrainedSolver
//
//=============================================================================


#ifndef COMISO_SUBSYSTEMSOLVERT_HH
#define COMISO_SUBSYSTEMSOLVERT_HH


//== INCLUDES =================================================================
#include <CoMISo/Config/CoMISoDefines.hh>
#include <CoMISo/Config/StdTypes.hh>
#include <CoMISo/Solver/SolverBaseT.hh>
#include <CoMISo/Solver/MultiDimConstrainedSolverT.hh>
#include <CoMISo/Utils/ProblemSubsetMapT.hh>

#include <vector>

//== NAMESPACES ===============================================================

namespace COMISO
{


// Class to solve linear systems of the form:
// Minimize ||Ax - b||^2, subject to linear constraints Cx=d,
// fix point constraints, and integer constraints.
// This class handles efficiently the case of solving a system with many unused
// variables. For solving many such systems its worth keeping the solver object
// alive as its construction is in O(n) where n is largest variable index.
template <int DIM>
class COMISODLLEXPORT SubProblemSolverT
{
public:
  using Point            = typename SolverBaseT<DIM>::Point;
  using PointVector      = typename SolverBaseT<DIM>::PointVector;
  using LinearEquation   = typename SolverBaseT<DIM>::LinearEquation;
  using ValueVector      = typename SolverBaseT<DIM>::ValueVector;
  using IndexVector      = std::vector<int>;
  using Result           = ValueVector;


  SubProblemSolverT(
      size_t _max_n_vars, const ValueVector& _fixed_values = ValueVector());

  /// delete copy constructor
  SubProblemSolverT(const SubProblemSolverT<DIM>&) = delete;

  /// delete assignment operator
  SubProblemSolverT& operator=(const SubProblemSolverT& _rhs) = delete;

  // Add an equation to the system. solve() will minimize sum of the quadratic
  // errors of all equations
  void add_equation(LinearEquation _eq);

  // Add a linear constraint to the system
  void add_constraint(LinearEquation _eq);

  // Add multiple linear constraints to the system
  void add_constraints(std::vector<LinearEquation> _eqs);

  // Set the fix point constraints which are special linear constraints for
  // fixing individual variables to given values. They are handled more
  // efficiently than the more general constraints specified with
  // add_constraints() by being removed from the system immediatly.
  // Also clears all equations, linear constraints, and integer constraints.
  void reset(ValueVector _fixed_values = {});

  // Set the integer constraints
  void set_integers(IndexVector _int_var_indcs);

  // Solve the system that has been setup with the calls above.
  void solve(Result& _result);


  // Update const term of the _eq_idx'th equation added via add_equation().
  void update_equation_const_term(size_t _eq_idx, const Point& _const_term);

  // Update const term of the _cnstrnt_idx'th constraint added via
  // add_constraint().
  void update_constraint_const_term(
      size_t _cnstrnt_idx, const Point& _const_term);

  void resolve(Result& _result);

  // Return the list of fixed values, without duplications
  const ValueVector& fixed_values() const;

private:

  // Transform result into value vector with indices mapped back to original
  // problem
  void to_values(typename MultiDimConstrainedSolverT<DIM>::Result& _points,
      Result& _values) const;

  ProblemSubsetMapT<DIM> sbst_map_;
  MultiDimConstrainedSolverT<DIM> solver_; // Solver used to solve subproblem

  PointVector eq_cnst_term_diff_; // Effect of fixed values on rhs of equations
  PointVector cnstrnt_cnst_term_diff_; // Effect of fixed values on rhs of
                                       // constraints
};


//=============================================================================
} // namespace COMISO
//=============================================================================
#if defined(INCLUDE_TEMPLATES) && !defined(COMISO_SUBSYSTEMSOLVERT_C)
#define COMISO_SUBSYSTEMSOLVERT_TEMPLATES
#include "SubProblemSolverT_impl.hh"
#endif
//=============================================================================
#endif // COMISO_CONSTRAINEDSOLVER_HH defined
//=============================================================================

