//=============================================================================
//
//  CLASS COMISOSolver - IMPLEMENTATION
//
//=============================================================================

//== INCLUDES =================================================================

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
#if COMISO_EIGEN3_AVAILABLE

//=============================================================================

#include <vector>
#include "COMISOSolver.hh"

//== NAMESPACES ===============================================================

namespace COMISO {

//== IMPLEMENTATION ==========================================================


// ********** SOLVE **************** //
void
COMISOSolver::
solve(NProblemInterface*                  _problem,
      std::vector<NConstraintInterface*>& _constraints,
      std::vector<PairUiV>&               _discrete_constraints,
      double                              _reg_factor,
      bool                                _show_miso_settings)
{

  //----------------------------------------------
  // 1. identify integer variables
  //----------------------------------------------

  // identify integer variables
  std::vector<int> round_idxs;
  for(unsigned int i=0; i<_discrete_constraints.size(); ++i)
    switch(_discrete_constraints[i].second)
    {
      case Binary :
      case Integer:
        round_idxs.push_back(_discrete_constraints[i].first); break;
      default     : break;
    }


  //----------------------------------------------
  // 2. setup constraints
  //----------------------------------------------
  std::size_t n = _problem->n_unknowns();
  std::vector<Eigen::Triplet<double>> triplets;
  int n_constraints = 0;

  // get zero vector
  ConstrainedSolver::Vector x(n);
  x.setZero();

  for(unsigned int i=0; i<_constraints.size();  ++i)
  {
    if(_constraints[i]->constraint_type() == NConstraintInterface::NC_EQUAL)
    {
      DEB_error_if(!_constraints[i]->is_linear(),
        "COMISOSolver received a problem with non-linear constraints!!!");

      // get linear part
      NConstraintInterface::SVectorNC gc;
      _constraints[i]->eval_gradient(P(x), gc);

      NConstraintInterface::SVectorNC::InnerIterator v_it(gc);
      for(; v_it; ++v_it)
        triplets.emplace_back(n_constraints, v_it.index(), v_it.value());

      // get constant part
      triplets.emplace_back(
          n_constraints, (int)n, _constraints[i]->eval_constraint(P(x)));

      // move to next constraint
      ++n_constraints;
    }
    else
    {
      DEB_error(
          "COMISOSolver received a problem with non-equality constraints!!!");
    }
  }

  // resize matrix to final number of constraints
  ConstrainedSolver::RowMatrix C(n_constraints, n + 1);
  C.setFromTriplets(triplets.begin(), triplets.end());

  //----------------------------------------------
  // 3. setup energy
  //----------------------------------------------

  DEB_error_if(!_problem->constant_hessian(),
    "COMISOSolver received a problem with non-constant hessian!!!");


  // get hessian matrix
  NProblemInterface::SMatrixNP H;
  _problem->eval_hessian(P(x), H);


  // get negative gradient
  ConstrainedSolver::Vector rhs(_problem->n_unknowns());
  _problem->eval_gradient(P(x), P(rhs));
  for(unsigned int i=0; i<rhs.size(); ++i)
    rhs[i] = -rhs[i];

//  // add constant part
//  objective += _problem->eval_f(P(x));

  //----------------------------------------------
  // 4. solve problem
  //----------------------------------------------

  cs_.solve(C,H,x,rhs,round_idxs,
            _reg_factor, _show_miso_settings);

  //  void solve(
  //      RMatrixT& _constraints,
  //      CMatrixT& _A,
  //      VectorT&  _x,
  //      VectorT&  _rhs,
  //      VectorIT& _idx_to_round,
  //      double    _reg_factor = 0.0,
  //      bool      _show_miso_settings = true );

  //----------------------------------------------
  // 5. store result
  //----------------------------------------------

  _problem->store_result(P(x));

//  std::cout << "COMISO Objective: " << model.get(GRB_DoubleAttr_ObjVal) << std::endl;
}


//=============================================================================
} // namespace COMISO
//=============================================================================
#endif // COMISO_EIGEN3_AVAILABLE
