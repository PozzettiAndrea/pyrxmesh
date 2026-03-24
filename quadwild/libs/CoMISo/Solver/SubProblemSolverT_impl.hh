// Copyright 2022 Autodesk, Inc. All rights reserved.

#define COMISO_SUBSYSTEMSOLVERT_C

#include <CoMISo/Solver/SubProblemSolverT.hh>
#include <CoMISo/Utils/ProblemSubsetMapT.hh>


namespace COMISO
{

template <int DIM>
SubProblemSolverT<DIM>::SubProblemSolverT(
    size_t _max_var_num, const ValueVector& _fixed_values)
    : sbst_map_(_max_var_num), solver_()
{
  reset(_fixed_values);
}


template <int DIM>
void
SubProblemSolverT<DIM>::add_equation(LinearEquation _eq)
{
  // Apply transformation due to fixed values and store change for resolve.
  eq_cnst_term_diff_.push_back(sbst_map_.map(_eq));
  solver_.add_equation(_eq);
}

template <int DIM>
void
SubProblemSolverT<DIM>::add_constraint(LinearEquation _eq)
{
  // Apply transformation due to fixed values and store change for resolve.
  cnstrnt_cnst_term_diff_.push_back(sbst_map_.map(_eq));
  solver_.add_constraint(_eq);
}

template <int DIM>
void SubProblemSolverT<DIM>::add_constraints(std::vector<LinearEquation> _eqs)
{
  for (auto& eq : _eqs)
    add_constraint(std::move(eq));
}

template <int DIM>
void
SubProblemSolverT<DIM>::reset(ValueVector _fixed_values)
{
  sbst_map_.reset_fixed_values(std::move(_fixed_values));
  solver_.reset();
  eq_cnst_term_diff_.clear();
  cnstrnt_cnst_term_diff_.clear();
}

template <int DIM>
void
SubProblemSolverT<DIM>::set_integers(IndexVector _int_var_indcs)
{
  sbst_map_.map(_int_var_indcs);
  solver_.set_integers(std::move(_int_var_indcs));
}

template <int DIM>
void
SubProblemSolverT<DIM>::solve(Result& _result)
{
  // Compute dense result
  typename MultiDimConstrainedSolverT<DIM>::Result res;
  solver_.solve(res);

  to_values(res, _result);
}

template <int DIM>
void SubProblemSolverT<DIM>::update_equation_const_term(
    size_t _eq_idx, const Point& _const_term)
{
  DEB_error_if(
      _eq_idx >= eq_cnst_term_diff_.size(), "Equation index out of range.");
  const auto const_term = _const_term + eq_cnst_term_diff_[_eq_idx];
  solver_.update_equation_const_term(_eq_idx, const_term);
}

template <int DIM>
void SubProblemSolverT<DIM>::update_constraint_const_term(
    size_t _cnstrnt_idx, const Point& _const_term)
{
  DEB_error_if(_cnstrnt_idx >= cnstrnt_cnst_term_diff_.size(),
      "Equation index out of range.");
  const auto const_term = _const_term + cnstrnt_cnst_term_diff_[_cnstrnt_idx];
  solver_.update_constraint_const_term(_cnstrnt_idx, const_term);
}


template <int DIM>
void
SubProblemSolverT<DIM>::resolve(Result& _result)
{
  // Compute dense result
  typename MultiDimConstrainedSolverT<DIM>::Result res;
  solver_.resolve(res);

  to_values(res, _result);
}

template <int DIM>
const typename SubProblemSolverT<DIM>::ValueVector&
SubProblemSolverT<DIM>::fixed_values() const
{
  return sbst_map_.fixed_values();
}


template <int DIM>
void
SubProblemSolverT<DIM>::to_values(
    typename MultiDimConstrainedSolverT<DIM>::Result& _points,
    Result& _values) const
{
  const auto size = _points.size();
  _values.clear();
  _values.reserve(size);
  for (size_t i = 0; i < size; ++i)
    _values.emplace_back(sbst_map_.mapped_back(i), _points[i]);
}

}//namespace COMISO


