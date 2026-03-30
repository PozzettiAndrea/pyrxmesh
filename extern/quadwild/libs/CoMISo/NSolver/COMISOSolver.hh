//=============================================================================
//
//  CLASS COMISOSolver
//
//=============================================================================


#ifndef COMISO_COMISOSOLVER_HH
#define COMISO_COMISOSOLVER_HH


//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
#if COMISO_EIGEN3_AVAILABLE

//== INCLUDES =================================================================

#include <CoMISo/Config/CoMISoDefines.hh>
#include <CoMISo/Solver/ConstrainedSolver.hh>
#include <vector>
#include "NProblemInterface.hh"
#include "NConstraintInterface.hh"
#include "VariableType.hh"


//== FORWARDDECLARATIONS ======================================================

//== NAMESPACES ===============================================================

namespace COMISO {

//== CLASS DEFINITION =========================================================



/** \class NewtonSolver GUROBISolver.hh

    Brief Description.

    A more elaborate description follows.
*/
class COMISODLLEXPORT COMISOSolver
{
public:

  typedef std::pair<unsigned int, VariableType> PairUiV;

  // ********** SOLVE **************** //
  void solve(NProblemInterface*                  _problem,                      // problem instance
             std::vector<NConstraintInterface*>& _constraints,                  // linear constraints
             std::vector<PairUiV>&               _discrete_constraints,         // discrete constraint
             double                              _reg_factor = 0.0,             // regularization factor
             bool                                _show_miso_settings = false);  // show settings dialog


  // get reference to ConstrainedSolver to manipulate parameters
  ConstrainedSolver& solver() { return cs_;}

protected:
  double* P(ConstrainedSolver::Vector& _v)
  {
    return _v.data();
  }

private:
  ConstrainedSolver cs_;
};



//=============================================================================
} // namespace COMISO
//=============================================================================
#endif // COMISO_EIGEN3_AVAILABLE
//=============================================================================
#endif // COMISO_GUROBISOLVER_HH defined
//=============================================================================

