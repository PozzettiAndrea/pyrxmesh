//=============================================================================
//
//  CLASS OSQPSolver
//
//=============================================================================

#ifndef COMISO_OSQPSOLVER_HH
#define COMISO_OSQPSOLVER_HH

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
#if COMISO_OSQP_AVAILABLE

//== INCLUDES =================================================================

#include <CoMISo/Config/CoMISoDefines.hh>
#include <vector>
#include <string>
#include "NProblemInterface.hh"
#include "NConstraintInterface.hh"

//== FORWARDDECLARATIONS ======================================================

//== NAMESPACES ===============================================================

namespace COMISO
{

//== CLASS DEFINITION =========================================================

/** \class OSQP Solver OSQPSolver.hh

    Solver for quadratic problem with linear equality and linear inequality
   constraints based on OSQP.
*/
class COMISODLLEXPORT OSQPSolver
{
public:
  using ContraintVector = std::vector<NConstraintInterface*>;

  // ********** SOLVE **************** //
  void solve(NProblemInterface* _problem, // problem instance
      const ContraintVector& _constraints // linear constraints
  );

  // same as above with additional lazy constraints that are only added
  // iteratively to the problem if not satisfied
  void solve(NProblemInterface* _problem, const ContraintVector& _constraints,
      const ContraintVector& _lazy_constraints,
      double _acceptable_tolerance = 1e-8,
      double _almost_infeasible_threshold = 0.5, int _max_passes = 5,
      bool _final_step_with_all_constraints = true);

  // Return the primal objective value
  double energy() const { return obj_val_; }

private:
  double obj_val_ = 0.0;

};

//=============================================================================
} // namespace COMISO

//=============================================================================
#endif // COMISO_OSQP_AVAILABLE
//=============================================================================
#endif // COMISO_OSQPSOLVER_HH defined
//=============================================================================
