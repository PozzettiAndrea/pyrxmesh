//=============================================================================
//
//  CLASS LPSolverSolver - IMPLEMENTATION
//
//=============================================================================

//== INCLUDES =================================================================

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
#if COMISO_LPSOLVE_AVAILABLE

//=============================================================================
#include "LPSolveSolver.hh"
#include <CoMISo/Utils/CoMISoError.hh>

#include <Base/Debug/DebTime.hh>
#include <Base/Code/Quality.hh>

#include <lp_lib.h>

#include <stdexcept>

//== NAMESPACES ===============================================================

namespace COMISO {

//== IMPLEMENTATION ==========================================================

namespace {

int getLPSolverRowType(NConstraintInterface* c)
{
  switch (c->constraint_type())
  {
  case NConstraintInterface::NC_EQUAL: return EQ;
  case NConstraintInterface::NC_LESS_EQUAL: return LE;
  case NConstraintInterface::NC_GREATER_EQUAL: return GE;
  }
}

std::vector<double> get_linear_energy_coefficients(NProblemInterface* _problem)
{
  std::vector<double> zero(_problem->n_unknowns(), 0);
  std::vector<double> q;
  q.resize(_problem->n_unknowns());
  _problem->eval_gradient(zero.data(), q.data());
  return q;
}

bool solve_impl(
    NProblemInterface*                        _problem,
    const std::vector<NConstraintInterface*>& _constraints,
    const std::vector<PairIndexVtype>&        _discrete_constraints,
    bool                                      _unbounded_variables,
    const double                              _time_limit)
{
  lprec* lp = nullptr;
  int Ncol = _problem->n_unknowns();
  int ret = 0;

  /* We will build the model row by row */
  lp = make_lp(0, Ncol);
  if(lp == nullptr)
    ret = 1; /* couldn't construct a new model... */

  set_timeout(lp, _time_limit);

  std::vector<int> col_idxs;
  col_idxs.reserve(Ncol);
  std::vector<REAL> row_coefficients;
  row_coefficients.reserve(Ncol);

  auto reset_vectors = [&col_idxs, &row_coefficients]()
  {
    col_idxs.clear();
    row_coefficients.clear();
    // LPSolve ignores first entry
    col_idxs.push_back(0);
    row_coefficients.push_back(0);
  };

  if (_unbounded_variables)
    for (size_t i = 1; i < Ncol+1; ++i)
      set_unbounded(lp, i);


  // setup objective function
  if(ret == 0)
  {
    auto coeffs = get_linear_energy_coefficients(_problem);
    reset_vectors();
    for (size_t i = 0; i < coeffs.size(); ++i)
    {
      col_idxs.push_back(i+1);
      row_coefficients.push_back(coeffs[i]);
    }

    /* set the objective in lpsolve */
    if(!set_obj_fnex(lp, row_coefficients.size(), row_coefficients.data(), col_idxs.data()))
      ret = 4;
  }


  // add constraints
  {
    set_add_rowmode(lp, TRUE);  /* makes building the model faster if it is done rows by row */
    std::vector<REAL> all_row_coefficients(Ncol); // including zeros
    NConstraintInterface::SVectorNC gc;

    for (auto c : _constraints)
    {
      reset_vectors();

      if (!c->is_linear())
      {
        DEB_error("LPSolve: non-linear constraints are not supported and thus ignored.");
        continue;
      }

      c->eval_gradient(all_row_coefficients.data(), gc);
      for (NConstraintInterface::SVectorNC::InnerIterator v_it(gc); v_it; ++v_it)
      {
        col_idxs.push_back(v_it.index()+1);
        row_coefficients.push_back(v_it.value());
      }

      const auto b = c->eval_constraint(all_row_coefficients.data());
      if(!add_constraintex(lp, row_coefficients.size(), row_coefficients.data(), col_idxs.data(), getLPSolverRowType(c), -b))
        ret = 3;
    }
    set_add_rowmode(lp, FALSE); /* rowmode should be turned off again when done building the model */
  }

  // setup variable types
  if (ret == 0)
  {
    for (auto c : _discrete_constraints)
    {
      switch (c.second)
      {
      case Real:
        // real is default
        break;
      case Integer:
        set_int(lp, c.first+1, true);
        break;
      case Binary:
        set_binary(lp, c.first+1, true);
        break;
      }
    }
  }

  if(ret == 0)
  {
    /* set the object direction to maximize */
    set_minim(lp);

    /* just out of curioucity, now show the model in lp format on screen */
    /* this only works if this is a console application. If not, use write_lp and a filename */
    //write_LP(lp, stdout);
    /* write_lp(lp, "model.lp"); */

    /* I only want to see important messages on screen while solving */
    set_verbose(lp, SEVERE);

    /* Now let lpsolve calculate a solution */
    ret = solve(lp);
    if(ret == OPTIMAL)
      ret = 0;
    else
      ret = 5;
  }

  if(ret == 0)
  {
    /* variable values */
    get_variables(lp, row_coefficients.data());
    _problem->store_result(row_coefficients.data());
    /* we are done now */
  }

  if(lp != nullptr)
  {
    /* clean up such that all used memory by lpsolve is freed */
    delete_lp(lp);
  }

  return ret == 0;
}

}//namespace 

bool LPSolveSolver::solve(
    NProblemInterface*                        _problem,
    const std::vector<NConstraintInterface*>& _constraints,
    const std::vector<PairIndexVtype>&        _discrete_constraints,
    const double                              _time_limit )
{
  DEB_enter_func;
  bool valid_solution = false;
  try
  {
    valid_solution = solve_impl(_problem, _constraints, _discrete_constraints, unbounded_variables_, _time_limit);
  }
  catch (...)
  {
    DEB_warning(1, "Caught an error");
    //COMISO_THROW(UNSPECIFIED_CBC_EXCEPTION);
  }
  return valid_solution;
}


//=============================================================================
} // namespace COMISO
//=============================================================================

#endif // COMISO_CBC_AVAILABLE
//=============================================================================

