/*===========================================================================*\
 *                                                                           *
 *                               CoMISo                                      *
 *      Copyright (C) 2008-2022 by Computer Graphics Group, RWTH Aachen      *
 *                           www.rwth-graphics.de                            *
 *                                                                           *
 *---------------------------------------------------------------------------*
 *  This file is part of CoMISo.                                             *
 *                                                                           *
 *  CoMISo is free software: you can redistribute it and/or modify           *
 *  it under the terms of the GNU General Public License as published by     *
 *  the Free Software Foundation, either version 3 of the License, or        *
 *  (at your option) any later version.                                      *
 *                                                                           *
 *  CoMISo is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 *  GNU General Public License for more details.                             *
 *                                                                           *
 *  You should have received a copy of the GNU General Public License        *
 *  along with CoMISo.  If not, see <http://www.gnu.org/licenses/>.          *
 *                                                                           *
\*===========================================================================*/

//=============================================================================
//
//  CLASS MISolver - IMPLEMENTATION
//
//=============================================================================

#define COMISO_CONSTRAINEDSOLVER_C

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>

//== INCLUDES =================================================================
#include "ConstrainedSolver.hh"
#include <CoMISo/Utils/gmm.hh>
#include <CoMISo/Solver/GMM_Tools.hh>
#include <CoMISo/Solver/Eigen_Tools.hh>
#include <float.h>
#include <CoMISo/Utils/MutablePriorityQueueT.hh>

#include <Base/Utils/StopWatch.hh>
#include <Base/Debug/DebOut.hh>

#include <type_traits>

//== NAMESPACES ===============================================================

#if COMISO_GMM_AVAILABLE

namespace COMISO
{

//== IMPLEMENTATION ==========================================================

// cf. issue #3 - gmm5.2 compat
template <typename T>
using linalg_traits = typename gmm::linalg_traits<
    typename std::remove_const<typename std::remove_reference<T>::type>::type>;

template <class RMatrixT, class CMatrixT, class VectorT, class VectorIT>
void
ConstrainedSolver::solve_const(
  const RMatrixT& _constraints,
  const CMatrixT& _A,
        VectorT&  _x,
  const VectorT&  _rhs,
  const VectorIT& _idx_to_round,
  const double    _reg_factor,
  const bool      _show_miso_settings)
{
  // copy and vectors
  VectorIT idx_to_round(_idx_to_round);

  // call non-const function
  solve(_constraints,
    _A,
    _x,
    _rhs,
    idx_to_round,
    _reg_factor,
    _show_miso_settings);
}

//-----------------------------------------------------------------------------

template <class RMatrixT, class VectorT, class VectorIT>
void
ConstrainedSolver::solve_const(
  const RMatrixT& _constraints,
  const RMatrixT& _B,
        VectorT&  _x,
  const VectorIT& _idx_to_round,
  const double    _reg_factor,
  const bool      _show_miso_settings)
{
  // copy vector
  VectorIT idx_to_round(_idx_to_round);

  // call non-const function
  solve(_constraints,
    _B,
    _x,
    idx_to_round,
    _reg_factor,
    _show_miso_settings);
}

//-----------------------------------------------------------------------------

template <class RMatrixT, class VectorT, class VectorIT>
void
ConstrainedSolver::solve(
  const RMatrixT& _constraints,
  const RMatrixT& _B,
        VectorT&  _x,
        VectorIT& _idx_to_round,
  const double    _reg_factor,
  const bool      _show_miso_settings)
{
  RowMatrix constraints;
  RowMatrix B;
  Vector x;
  COMISO_EIGEN::gmm_to_eigen(_constraints, constraints);
  COMISO_EIGEN::gmm_to_eigen(_B, B);
  COMISO_EIGEN::to_eigen_vec(_x, x);

  solve(constraints, B, x, _idx_to_round, _reg_factor, _show_miso_settings);

  COMISO_EIGEN::from_eigen_vec(x, _x);
}

//-----------------------------------------------------------------------------

template <class RMatrixT, class CMatrixT, class VectorT, class VectorIT>
void
ConstrainedSolver::solve(
  const RMatrixT& _constraints,
  const CMatrixT& _A,
        VectorT&  _x,
  const VectorT&  _rhs,
        VectorIT& _idx_to_round,
  const double    _reg_factor,
  const bool      _show_miso_settings)
{
  RowMatrix constraints;
  ColMatrix A;
  COMISO_EIGEN::gmm_to_eigen(_constraints, constraints);
  COMISO_EIGEN::gmm_to_eigen(_A, A);
  Vector x;
  Vector rhs;
  COMISO_EIGEN::to_eigen_vec(_x, x);
  COMISO_EIGEN::to_eigen_vec(_rhs, rhs);

  solve(
      constraints, A, x, rhs, _idx_to_round, _reg_factor, _show_miso_settings);

  COMISO_EIGEN::from_eigen_vec(x, _x);
}

//-----------------------------------------------------------------------------

template <class RMatrixT, class VectorT>
void
ConstrainedSolver::resolve(
  const RMatrixT& _B,
        VectorT&  _x,
  const VectorT*  _constraint_rhs)
{
  RowMatrix B;
  Vector x;
  Vector constraint_rhs;
  Vector* constraint_rhs_ptr = nullptr;
  COMISO_EIGEN::gmm_to_eigen(_B, B);
  COMISO_EIGEN::to_eigen_vec(_x, x);
  if (_constraint_rhs != nullptr)
  {
    COMISO_EIGEN::to_eigen_vec(*_constraint_rhs, constraint_rhs);
    constraint_rhs_ptr = &constraint_rhs;
  }
  resolve(B, x, constraint_rhs_ptr);
  COMISO_EIGEN::from_eigen_vec(x, _x);
}

//-----------------------------------------------------------------------------

template <class VectorT>
void
ConstrainedSolver::resolve(
        VectorT& _x,
  const VectorT* _constraint_rhs,
  const VectorT* _rhs)
{
  Vector x;
  Vector constraint_rhs;
  Vector rhs;
  Vector* constraint_rhs_ptr = nullptr;
  Vector* rhs_ptr = nullptr;
  COMISO_EIGEN::to_eigen_vec(_x, x);
  if (_constraint_rhs != nullptr)
  {
    COMISO_EIGEN::to_eigen_vec(*_constraint_rhs, constraint_rhs);
    constraint_rhs_ptr = &constraint_rhs;
  }
  if (_rhs != nullptr)
  {
    COMISO_EIGEN::to_eigen_vec(*_rhs, rhs);
    rhs_ptr = &rhs;
  }

  resolve(x, constraint_rhs_ptr, rhs_ptr);

  COMISO_EIGEN::from_eigen_vec(x, _x);
}

//=============================================================================
} // namespace COMISO
//=============================================================================

#endif // COMISO_GMM_AVAILABLE
