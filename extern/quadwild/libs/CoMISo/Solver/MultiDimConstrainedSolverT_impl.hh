// Copyright 2022 Autodesk, Inc. All rights reserved.

#define COMISO_MULTIDIMCONSTRAINEDSOLVERT_C

#include "MultiDimConstrainedSolverT.hh"

#include <CoMISo/Solver/ConstrainedSolver.hh>

#include <Eigen/Sparse>

namespace COMISO
{

template <int DIM>
class MultiDimConstrainedSolverT<DIM>::Impl
{
  using RowMatrix = ConstrainedSolver::RowMatrix;
  using Vector = ConstrainedSolver::Vector;

  using TripletVector = std::vector<Eigen::Triplet<double>>;

public:
  Impl()
      : var_nmbr_(0)
  {
  }

  void add_equation(const LinearEquation& _eq) { add(_eq, A_triplets_, b_); }
  void add_constraint(const LinearEquation& _eq) { add(_eq, C_triplets_, d_); }

  void set_integers(IndexVector _int_var_indcs)
  {
    int_var_indcs_ = std::move(_int_var_indcs);
  }

  // Solve problem.
  void solve(Result& _result);

  // Update const term of the _eq_idx'th equation added via add_equation().
  void update_equation_const_term(size_t _eq_idx, const Point& _const_term)
  {
    DEB_error_if(_eq_idx >= b_.size(), "Index out of range.");
    b_[_eq_idx] = _const_term;
  }

  // Update const term of the _cnstrnt_idx'th constraint added via
  // add_constraint().
  void update_constraint_const_term(size_t _cnstrnt_idx, const Point& _const_term)
  {
    DEB_error_if(_cnstrnt_idx >= d_.size(), "Index out of range.");
    d_[_cnstrnt_idx] = _const_term;
  }

  // Resolve problem with changed right hand sides. You need to ensure that
  // solve has been called before calling this function.
  void resolve(Result& _result);

  // Clear all equations, constraints, and integer constraints so that a new
  // system can be solved.
  void reset();

private:

  // Create matrix from triplets with first dim of rhs in last column
  static void to_matrix(TripletVector& _triplets, size_t _col_num,
      const PointVector& _rhs, RowMatrix& _mat)
  {
    const auto row_num = _rhs.size();
    const auto triplets_nmbr = _triplets.size();
    // temporarily add rhs to triplets
    for (size_t i = 0; i < row_num; ++i)
      _triplets.emplace_back((int)i, (int)_col_num, -_rhs[i][0]);
    _mat.resize(row_num, _col_num + 1);
    _mat.reserve(_triplets.size());
    _mat.setFromTriplets(_triplets.begin(), _triplets.end());
    _triplets.resize(triplets_nmbr);// Remove the added triplets
  }

  // Extract simple one dimensional vector for column _col of a point vector
  static void to_vector(const PointVector& _point_vec, int _col, Vector& _vec)
  {
    _vec.resize(_point_vec.size());
    for (size_t i = 0; i < _point_vec.size(); ++i)
      _vec[i] = _point_vec[i][_col];
  }

  // Exchange rhs (last column) of _mat with column _col of _rhs
  static void update_rhs(const PointVector& _rhs, int _col, RowMatrix& _mat)
  {
    for (int i = 0; i < _mat.rows(); ++i)
      _mat.coeffRef(i, _mat.cols() - 1) = -_rhs[i][_col];
  }

  // Add a linear equation into triplets and rhs vector
  void add(const LinearEquation& _eq, TripletVector& _M, PointVector& _rhs)
  {
    const auto row_idx = _rhs.size();
    for (const auto& term : _eq.linear_terms)
    {
      _M.emplace_back((int)row_idx, (int)term.var_name, term.coeff);
      var_nmbr_ = std::max(var_nmbr_, term.var_name+1);
    }
    _rhs.push_back(_eq.const_term);
  }

  // Store one dimensional solution in dimension _dim_idx of _result
  void store_result(const ConstrainedSolver::Vector& _solution, int _dim_idx,
    PointVector& _result)
  {
    DEB_error_if(_solution.size() != (int)var_nmbr_,
        "Unexpected solution size of " << (int)_solution.size()
                                       << " instead of expected " << var_nmbr_);
    _result.resize(var_nmbr_);
    for (size_t i = 0; i < var_nmbr_; ++i)
      _result[i][_dim_idx] = _solution[i];
  }

  // Resolve for a given dimension
  void resolve(int _dim_idx, PointVector& _result);

  RowMatrix     A_; // Matrix A defining the minimization objective ||Ax-b||^2
  TripletVector A_triplets_; // Triplets for matrix A_
  PointVector   b_; // b of the minimization objective ||Ax-b||^2
  RowMatrix     C_; // Matrix C defining the linear equality constraints Cx=d
  TripletVector C_triplets_; // Triplets for matrix C_
  PointVector   d_; // rhs of equality constraints Cx = d
  IndexVector   int_var_indcs_; // List of variables which should be rounded to
                                // integers.

  size_t var_nmbr_; // Number of variables in the problem

  ConstrainedSolver solver_; // Solver for one dimensional problems.
};

template <int DIM>
void
MultiDimConstrainedSolverT<DIM>::Impl::solve(Result& _result)
{
  DEB_enter_func;

  // create systems with first dimension of _b,_d as rhs in last column of A,C
  to_matrix(A_triplets_, var_nmbr_, b_, A_);
  to_matrix(C_triplets_, var_nmbr_, d_, C_);

  ConstrainedSolver::Vector solution(var_nmbr_);

  // solve system for first dimension
  solver_.solve(C_, A_, solution, int_var_indcs_, 0.0, false);
  store_result(solution, 0, _result);

  // solve system for remaining dimensions
  for (int i = 1; i < DIM; ++i)
    resolve(i, _result);
}


template <int DIM>
void
MultiDimConstrainedSolverT<DIM>::Impl::resolve(Result& _result)
{
  DEB_error_if(A_.rows() == 0,
    "Equation matrix is empty. solve() needs to be called before resolve().");
  // resolve system for all dimensions
  for (int i = 0; i < DIM; ++i)
    resolve(i, _result);
}

template <int DIM>
void
MultiDimConstrainedSolverT<DIM>::Impl::resolve(int _dim_idx, PointVector& _result)
{
  DEB_enter_func;

  ConstrainedSolver::Vector solution(var_nmbr_);

  // exchange right hand sides
  update_rhs(b_, _dim_idx, A_);
  Vector rhs;
  to_vector(d_, _dim_idx, rhs);

  // resolve for updated rhs
  solver_.resolve(A_, solution, &rhs);

  store_result(solution, _dim_idx, _result);
}

template <int DIM>
void
COMISO::MultiDimConstrainedSolverT<DIM>::Impl::reset()
{
  A_triplets_.clear();
  b_.clear();
  C_triplets_.clear();
  d_.clear();
  int_var_indcs_.clear();
  var_nmbr_ = 0;
}

template <int DIM>
MultiDimConstrainedSolverT<DIM>::MultiDimConstrainedSolverT()
    : impl_(new Impl())
{
}

template <int DIM>
MultiDimConstrainedSolverT<DIM>::~MultiDimConstrainedSolverT()
{
  if (impl_ != nullptr)
    delete impl_;
}

template <int DIM>
void
MultiDimConstrainedSolverT<DIM>::add_equation(const LinearEquation& _eq)
{
  impl_->add_equation(_eq);
}

template <int DIM>
void
MultiDimConstrainedSolverT<DIM>::add_constraint(const LinearEquation& _eq)
{
  impl_->add_constraint(_eq);
}

template <int DIM>
void
MultiDimConstrainedSolverT<DIM>::set_integers(IndexVector _int_var_indcs)
{
  impl_->set_integers(std::move(_int_var_indcs));
}

template <int DIM>
void
MultiDimConstrainedSolverT<DIM>::solve(Result& _result)
{
  return impl_->solve(_result);
}

template <int DIM>
void
MultiDimConstrainedSolverT<DIM>::reset()
{
  impl_->reset();
}

template <int DIM>
void MultiDimConstrainedSolverT<DIM>::update_equation_const_term(
    size_t _eq_idx, const Point& _const_term)
{
  impl_->update_equation_const_term(_eq_idx, _const_term);
}

template <int DIM>
void MultiDimConstrainedSolverT<DIM>::update_constraint_const_term(
    size_t _cnstrnt_idx, const Point& _const_term)
{
  impl_->update_constraint_const_term(_cnstrnt_idx, _const_term);
}

template <int DIM>
void
MultiDimConstrainedSolverT<DIM>::resolve(Result& _result)
{
  return impl_->resolve(_result);
}


}//namespace COMISO


