// Copyright 2022 Autodesk, Inc. All rights reserved.


//=============================================================================
//
//  CLASS MultiDimConstrainedSolver
//
//=============================================================================


#ifndef COMISO_MULTIDIMCONSTRAINEDSOLVERT_HH
#define COMISO_MULTIDIMCONSTRAINEDSOLVERT_HH


//== INCLUDES =================================================================
#include <CoMISo/Config/CoMISoDefines.hh>
#include <CoMISo/Config/StdTypes.hh>
#include <CoMISo/Solver/SolverBaseT.hh>

#include <vector>

//== NAMESPACES ===============================================================

namespace COMISO
{

// Class to solve linear systems of the form:
// Minimize ||Ax - b||^2, subject to linear constraints Cx=d, and integer
// constraints.
template <int DIM>
class COMISODLLEXPORT MultiDimConstrainedSolverT
{
public:
  using Point            = typename SolverBaseT<DIM>::Point;
  using PointVector      = typename SolverBaseT<DIM>::PointVector;
  using LinearTerm       = typename SolverBaseT<DIM>::LinearTerm;
  using LinearTermVector = typename SolverBaseT<DIM>::LinearTermVector;
  using LinearEquation   = typename SolverBaseT<DIM>::LinearEquation;
  using IndexVector      = std::vector<int>;
  using Result           = PointVector;


  MultiDimConstrainedSolverT();

  /// delete copy constructor
  MultiDimConstrainedSolverT(const MultiDimConstrainedSolverT<DIM>&) = delete;

  /// delete assignment operator
  MultiDimConstrainedSolverT& operator=(
      const MultiDimConstrainedSolverT& _rhs) = delete;

  ~MultiDimConstrainedSolverT();

  // Add an equation to the system. solve() will minimize sum of the quadratic
  // errors of all equations
  void add_equation(const LinearEquation& _eq);

  // Add a linear constraint to the system
  void add_constraint(const LinearEquation& _eq);

  // Set the integer constraints
  void set_integers(IndexVector _int_var_indcs);

  // Solve the system that has been setup with the calls above.
  void solve(Result& _result);

  // Clear all equations, constraints, and integer constraints so that a new
  // system can be solved.
  void reset();

  // Update const term of the _eq_idx'th equation added via add_equation().
  void update_equation_const_term(size_t _eq_idx, const Point& _const_term);

  // Update const term of the _cnstrnt_idx'th constraint added via
  // add_constraint().
  void update_constraint_const_term(size_t _cnstrnt_idx, const Point& _const_term);

  // Resolve problem with changed right hand sides. You need to ensure that
  // solve has been called before calling this function.
  void resolve(Result& _result);

private:

  class Impl;
  Impl* impl_;
};


//=============================================================================
} // namespace COMISO
//=============================================================================
#if defined(INCLUDE_TEMPLATES) && !defined(COMISO_MULTIDIMCONSTRAINEDSOLVERT_C)
#define COMISO_MULTIDIMCONSTRAINEDSOLVERT_TEMPLATES
#include "MultiDimConstrainedSolverT_impl.hh"
#endif
//=============================================================================
#endif // COMISO_CONSTRAINEDSOLVER_HH defined
//=============================================================================

