//=============================================================================
//
//  CLASS IterativeSolverT - IMPLEMENTATION
//
//=============================================================================

#define COMISO_ITERATIVESOLVERT_C

//== INCLUDES =================================================================

#include "IterativeSolverT.hh"
#include <Base/Debug/DebOut.hh>
#include <CoMISo/Solver/Eigen_Tools.hh>

//== NAMESPACES ===============================================================

namespace COMISO
{

//== IMPLEMENTATION ==========================================================


//-----------------------------------------------------------------------------

template <class RealT>
bool IterativeSolverT<RealT>::gauss_seidel_local(const Matrix& _A,
    Vector& _x, const Vector& _rhs, const IndexVector& _idxs,
    const int _max_iter, const Real& _tolerance)
{
  if (_max_iter == 0)
    return false;

  updt_vrbl_indcs_.clear();
  indx_queue_.clear();

  for (size_t i = 0; i < _idxs.size(); ++i)
    indx_queue_.push_back(_idxs[i]);

  int it_count = 0;

  while (!indx_queue_.empty() && it_count < _max_iter)
  {
    ++it_count;
    const auto i = indx_queue_.front();
    indx_queue_.pop_front();
    indx_temp_.clear();

    double res_i = -_rhs[i];
    double x_i_new = _rhs[i];
    double diag = 1.0;

    for (typename Matrix::InnerIterator it(_A, i); it; ++it)
    {
      const auto j = static_cast<unsigned>(it.row());
      res_i += it.value() * _x[j];
      x_i_new -= it.value() * _x[j];
      if (j != i)
        indx_temp_.push_back(j);
      else
        diag = it.value();
    }

    // take inverse of diag
    diag = 1.0 / diag;

    // compare relative residuum normalized by diagonal entry
    if (std::abs(res_i * diag) > _tolerance)
    {
      _x[i] += x_i_new * diag;
      updt_vrbl_indcs_.push_back(i);
      for (size_t j = 0; j < indx_temp_.size(); ++j)
        indx_queue_.push_back(indx_temp_[j]);
    }
  }

  return indx_queue_.empty(); // converged?
}

//-----------------------------------------------------------------------------


template <class RealT>
bool IterativeSolverT<RealT>::conjugate_gradient(const Matrix& _A,
    Vector& _x, const Vector& _rhs, int& _max_iter, Real& _tolerance)
{
  DEB_enter_func;
  Real rho, rho_1(0), a;

  // initialize vectors
  p_.resize(_x.size());
  q_.resize(_x.size());
  r_.resize(_x.size());
  d_.resize(_x.size());
  p_ = _x; // gets overwritten before being used?

  // initialize diagonal (for relative norm)
  d_ = _A.diagonal().cwiseInverse();

  // start with iteration 0
  int cur_iter(0);

  r_ = _A * -_x + _rhs;
  rho = r_.dot(r_);
  p_ = r_;

  bool not_converged = true;
  Real res_norm(0);

  // while not converged
  while ((not_converged = ((res_norm = vect_norm_rel(r_, d_)) > _tolerance)) &&
         cur_iter < _max_iter)
  {
    DEB_line(11, "iter " << cur_iter << "  res " << res_norm);

    if (cur_iter != 0)
    {
      rho = r_.dot(r_);
      p_ = r_ + rho / rho_1 * p_;
    }

    q_ = _A * p_;

    a = rho / q_.dot(p_);
    _x += a * p_;
    r_ -= a * q_;
    rho_1 = rho;

    ++cur_iter;
  }

  _max_iter = cur_iter;
  _tolerance = res_norm;

  return !not_converged;
}

//-----------------------------------------------------------------------------

template <class RealT>
typename IterativeSolverT<RealT>::Real IterativeSolverT<RealT>::vect_norm_rel(
    const Vector& _v, const Vector& _diag) const
{
  // compute component wise product
  const auto cwise_product = _v.array() * _diag.array();
  // return largest coefficient
  return cwise_product.abs().maxCoeff();
}


//=============================================================================
} // namespace COMISO
//=============================================================================
