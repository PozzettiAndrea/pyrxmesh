//=============================================================================
//
//  CLASS IterativeSolverT
//
//=============================================================================

#ifndef COMISO_ITERATIVESOLVERT_HH
#define COMISO_ITERATIVESOLVERT_HH

//== INCLUDES =================================================================

#include <Eigen/Sparse>
#include <deque>

//== FORWARDDECLARATIONS ======================================================

//== NAMESPACES ===============================================================

namespace COMISO
{

//== CLASS DEFINITION =========================================================

/** \class IterativeSolverT IterativeSolverT.hh <COMISO/.../IterativeSolverT.hh>

    Brief Description.

    A more elaborate description follows.
*/

template <class RealT> class IterativeSolverT
{
public:
  typedef unsigned int uint;
  typedef RealT Real;
  typedef std::vector<uint> IndexVector;
  typedef Eigen::SparseMatrix<Real, Eigen::ColMajor> Matrix;
  typedef Eigen::Matrix<Real, Eigen::Dynamic, 1> Vector;

  // local Gauss-Seidel
  bool gauss_seidel_local(const Matrix& _A, Vector& _x, const Vector& _rhs,
      const IndexVector& _idxs, const int _max_iter, const Real& _tolerance);

  // get the indices of any variables updated during the last local Gauss-Seidel
  const IndexVector& updated_variable_indices() const
  {
    return updt_vrbl_indcs_;
  }

  // conjugate gradient
  bool conjugate_gradient(const Matrix& _A, Vector& _x, const Vector& _rhs,
      int& _max_iter, Real& _tolerance);

private:
  // compute relative norm
  Real vect_norm_rel(const Vector& _v, const Vector& _diag) const;

private:
  // context  for Conjugate Gradient
  Vector p_;
  Vector q_;
  Vector r_;
  Vector d_;

  // context for local Gauss-Seidel
  IndexVector indx_temp_;
  std::deque<uint> indx_queue_;
  IndexVector updt_vrbl_indcs_; // updated variable indices
};

//=============================================================================
} // namespace COMISO
//=============================================================================
#if defined(INCLUDE_TEMPLATES) && !defined(COMISO_ITERATIVESOLVERT_C)
#define COMISO_ITERATIVESOLVERT_TEMPLATES
#include "IterativeSolverT_impl.hh"
#endif
//=============================================================================
#endif // COMISO_ITERATIVESOLVERT_HH defined
//=============================================================================
