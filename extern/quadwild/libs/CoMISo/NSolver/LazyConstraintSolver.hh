//=============================================================================
//
//  CLASS LazyConstraintSolver
//
//=============================================================================


#ifndef COMISO_LAZYCONSTRAINTSOLVER_HH
#define COMISO_LAZYCONSTRAINTSOLVER_HH


//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
//#if COMISO_OSQP_AVAILABLE // TODO

//== INCLUDES =================================================================

#include <CoMISo/Config/CoMISoDefines.hh>
#include "NProblemInterface.hh"
#include "NConstraintInterface.hh"

#include <Base/Debug/DebOut.hh>
#include <vector>

//== FORWARDDECLARATIONS ======================================================


//== NAMESPACES ===============================================================

namespace COMISO {

//== CLASS DEFINITION =========================================================

namespace LazySolveImpl
{
enum FeasibilityType
{
  FT_FEASIBLE,
  FT_INFEASIBLE,
  FT_ALMOST_INFEASIBLE
};

inline FeasibilityType get_feasibility(NConstraintInterface* c,
    const double* const x, const double _acceptable_tolerance,
    const double _almost_infeasible_threshold)
{
  auto v = c->eval_constraint(x);

  if (c->constraint_type() == NConstraintInterface::NC_EQUAL)
  {
    if (std::abs(v) < _acceptable_tolerance)
      return FT_FEASIBLE;
    else
      return FT_INFEASIBLE;
  }
  else if (c->constraint_type() == NConstraintInterface::NC_LESS_EQUAL)
  {
    if (v >= -_acceptable_tolerance)
      return FT_INFEASIBLE;
    else if (v >= -_almost_infeasible_threshold)
      return FT_ALMOST_INFEASIBLE;
    else
      return FT_FEASIBLE;
  }
  else if (c->constraint_type() == NConstraintInterface::NC_GREATER_EQUAL)
  {
    if (v <= _acceptable_tolerance)
      return FT_INFEASIBLE;
    else if (v <= _almost_infeasible_threshold)
      return FT_ALMOST_INFEASIBLE;
    else
      return FT_FEASIBLE;
  }
  else
  {
    DEB_error("Unknown constraint type");
    return FT_INFEASIBLE;
  }
}
} // namespace LazySolveImpl

/// SolveFunction should be callable with two arguments: NProblemInterface* and
/// const std::vector<NConstraintInterface*> Result function should return
/// double* to solution
template <typename SolveFunction, typename ResultFunction>
void solve_with_lazy_constraints(SolveFunction& _solve,
    ResultFunction& _get_result, NProblemInterface* _problem,
    const std::vector<NConstraintInterface*>& _initial_constraints,
    const std::vector<NConstraintInterface*>& _lazy_constraints,
    double _acceptable_tolerance = 1e-8,
    double _almost_infeasible_threshold = 0.5, int _max_passes = 5,
    bool _final_step_with_all_constraints = false)
{
  DEB_enter_func;

  if (_max_passes <= 0 && !_final_step_with_all_constraints)
  {
    DEB_warning(2, "Asking for "
                       << _max_passes
                       << " passes and no final step with all constraints. "
                       << "Will perform 1 pass instead.");
    _max_passes = 1;
  }

  std::vector<NConstraintInterface*> constraints = _initial_constraints;
  std::vector<bool> added(_lazy_constraints.size(), false);

  std::vector<int> n_infeasible;
  n_infeasible.reserve(_max_passes);
  std::vector<int> n_almost_infeasible;
  n_almost_infeasible.reserve(_max_passes);

  for (int pass = 0; pass < _max_passes; ++pass)
  {
    _solve(_problem, constraints);

    n_infeasible.push_back(0);
    n_almost_infeasible.push_back(0);

    const auto* solution_x = _get_result();

    for (size_t i = 0; i < _lazy_constraints.size(); ++i)
    {
      if (added[i])
        continue; // we already added this constraint

      auto f = LazySolveImpl::get_feasibility(_lazy_constraints[i], solution_x, _acceptable_tolerance, _almost_infeasible_threshold);
      if (f == LazySolveImpl::FT_INFEASIBLE)
        ++n_infeasible.back();
      else if (f == LazySolveImpl::FT_ALMOST_INFEASIBLE)
        ++n_almost_infeasible.back();

      if (f != LazySolveImpl::FT_FEASIBLE)
      {
        constraints.push_back(_lazy_constraints[i]);
        added[i] = true;
      }
    }

    if (n_infeasible.back() == 0)
      break; // if nothing is infeasible we are done
  }

  DEB_only(bool did_final_step = false);
  if (n_infeasible
          .empty() || // no initial step without constraints was performed or
      (n_infeasible.back() != 0 && // (the last step was not successful and
          _final_step_with_all_constraints)) // we want to do a pass with all
                                             // constraints)
  {
    DEB_only(did_final_step = true);

    // add remaining constraints
    for (size_t i = 0; i < _lazy_constraints.size(); ++i)
    {
      if (!added[i])
        constraints.push_back(_lazy_constraints[i]);
    }

    _solve(_problem, constraints);
  }

  // Retrieve some statistics about the solve
  DEB_line(4, "############# lazy constraints statistics ###############");
  DEB_line(4, _lazy_constraints.size() << " lazy constraints in input.");
  DEB_line(4, "#passes     : " << n_infeasible.size() << "( of " << _max_passes << ")");
  for(size_t i=0; i<n_infeasible.size(); ++i)
    DEB_line(5, "pass " << i+1 << " induced " << n_infeasible[i]
      << " infeasible and " << n_almost_infeasible[i] << " almost infeasible");
  DEB_line_if(did_final_step, 4, "Did final step with all lazy constraints");
}


//=============================================================================
} // namespace COMISO

//=============================================================================
//#endif // COMISO_OSQP_AVAILABLE
//=============================================================================
#endif // COMISO_OSQPSOLVER_HH defined
//=============================================================================

