// (C) Copyright 2015 by Autodesk, Inc.

//=============================================================================
//
//  CLASSLPSolverSolver
//
//=============================================================================
#ifndef COMISO_LPSOLVESOLVER_HH
#define COMISO_LPSOLVESOLVER_HH

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>

#if COMISO_LPSOLVE_AVAILABLE

//== INCLUDES =================================================================

#include <CoMISo/Config/CoMISoDefines.hh>
#include <vector>
#include <string>
#include <CoMISo/NSolver/NProblemInterface.hh>
#include <CoMISo/NSolver/NConstraintInterface.hh>
#include <CoMISo/NSolver/VariableType.hh>

//== FORWARDDECLARATIONS ======================================================


//== NAMESPACES ===============================================================

namespace COMISO {

//== CLASS DEFINITION =========================================================

/**
    Solver interface for LPSolve.
*/
class COMISODLLEXPORT LPSolveSolver
{
public:
  // By default variables have a lower bound of 0.
  // By choosing unbounded variables LPSolve will double the number of variables
  // which leads to a more expensive solve.
  LPSolveSolver(bool _unbounded_variables = false)
    :
      unbounded_variables_(_unbounded_variables)
  {}

  // ********** SOLVE **************** //
  //! \throws Outcome
  bool solve(
    NProblemInterface* _problem, // problem instance
    const std::vector<NConstraintInterface*>& _constraints, // linear constraints
    const std::vector<PairIndexVtype>& _discrete_constraints, // discrete constraints
    const double _time_limit = 60); // time limit in seconds

  //! \throws Outcome
 bool solve(
    NProblemInterface* _problem, // problem instance
    const std::vector<NConstraintInterface*>& _constraints, // linear constraints
    const double _time_limit = 60) // time limit in seconds
  {
    std::vector<PairIndexVtype> dc;
    return solve(_problem, _constraints, dc, _time_limit);
  }

private:

 bool unbounded_variables_;
};


//=============================================================================
} // namespace COMISO

//=============================================================================
#endif // COMISO_CBC_AVAILABLE
//=============================================================================
#endif // COMISO_CBCSolver_HH
//=============================================================================

