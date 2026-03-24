// Copyright 2021 Autodesk, Inc. All rights reserved.

#include "EigenLSQSolver.hh"
#include <CoMISo/Utils/CoMISoError.hh>

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#if (COMISO_EIGEN3_AVAILABLE)
//== INCLUDES =================================================================

#include <Base/Code/Quality.hh>
LOW_CODE_QUALITY_SECTION_BEGIN
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCholesky>
LOW_CODE_QUALITY_SECTION_END

namespace COMISO
{

template <size_t DIM>
void EigenLSQSolverT<DIM>::solve(Result& _result)
{
  using SparseMatrix = Eigen::SparseMatrix<double>;
  using ColumnMatrix = Eigen::Matrix<double, Eigen::Dynamic, DIM>;
  std::vector<Eigen::Triplet<double, size_t>> A_coeff;
  A_coeff.reserve(lin_eqs_.size() * 5); // Estimation of 5 coefficient per row
  ColumnMatrix B(lin_eqs_.size(), DIM);
  size_t max_col = 0;
  for (size_t i = 0; i < lin_eqs_.size(); ++i)
  {
    const auto& lin_eq = lin_eqs_[i];
    for (size_t j = DIM; j-- > 0;)
      B(i, j) = lin_eq.const_term[j];
    for (const auto& term : lin_eq.linear_terms)
    {
      A_coeff.emplace_back(i, term.var_name, term.coeff);
      max_col = std::max(term.var_name, max_col);
    }
  }
  // Automatically finds the size of the system to solve
  const size_t col_size = max_col + 1;
  SparseMatrix A(lin_eqs_.size(), col_size);
  A.setFromTriplets(A_coeff.begin(), A_coeff.end());
  // SimplicialLDLT needs only the lower triangular matrix
  SparseMatrix M = (A.transpose() * A).triangularView<Eigen::Lower>();
  Eigen::SimplicialLDLT<SparseMatrix> solver(M);
  ColumnMatrix X = solver.solve(A.transpose() * B);
  if (solver.info() != Eigen::Success)
  {
    // We may want to try a different factorization here, for example SparseLU
    // is to be able to find a result. Nevertheless if we are here the
    // minimization problem has multiple solutions and randomly picking one of
    // them is dangerous, for example the solution may have unpleasant
    // oscillations.
    COMISO_THROW(LSQC_SINGULAR);
  }
  _result.resize(col_size);
  for (size_t i = 0; i < _result.size(); ++i)
  {
    for (size_t j = 0; j < DIM; ++j)
      _result[i][j] = X(i, j);
  }
}

} // namespace COMISO

//=============================================================================
#endif // COMISO_EIGEN3_AVAILABLE
//=============================================================================
