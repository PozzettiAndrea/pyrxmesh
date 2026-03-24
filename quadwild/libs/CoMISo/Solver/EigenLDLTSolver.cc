/*===========================================================================*\
 *                                                                           *
 *                               CoMISo                                      *
 *      Copyright (C) 2008-2009 by Computer Graphics Group, RWTH Aachen      *
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

#include "EigenLDLTSolverT_impl.hh"

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#if (COMISO_EIGEN3_AVAILABLE)
//== INCLUDES =================================================================

namespace COMISO {

//-----------------------------------------------------------------------------


bool EigenLDLTSolver::calc_system( const std::vector<int>&    _colptr,
				 const std::vector<int>&    _rowind,
				 const std::vector<double>& _values)
{
  DEB_error("EigenLDLTSolver::calc_system( const std::vector<int> ...) not "
            "implemented yet...");
  return false;
}


//-----------------------------------------------------------------------------


bool EigenLDLTSolver::update_system( const std::vector<int>& _colptr,
				     const std::vector<int>& _rowind,
				     const std::vector<double>& _values )
{
  DEB_error("EigenLDLTSolver::update_system( const std::vector<int> ...) not "
            "implemented yet...");
  return false;
}

//-----------------------------------------------------------------------------

bool EigenLDLTSolver::solve(double* _x, double* _b)
{
  // map arrays to Eigen-Vectors
  Eigen::Map<Eigen::VectorXd> x(_x, n_);
  Eigen::Map<Eigen::VectorXd> b(_b, n_);

  // solve for another right hand side:
  x = ldlt_.solve(b);

  return ldlt_.info() == Eigen::Success;
}

//-----------------------------------------------------------------------------

bool EigenLDLTSolver::solve(Eigen::VectorXd& _x, const Eigen::VectorXd& _b)
{
  // solve for another right hand side:
  _x = ldlt_.solve(_b);

  return ldlt_.info() == Eigen::Success;
}


//-----------------------------------------------------------------------------

int EigenLDLTSolver::dimension()
{
  return n_;
}

//-----------------------------------------------------------------------------

bool EigenLDLTSolver::
solve ( std::vector<double>& _x0, std::vector<double>& _b)
{
  return solve(&(_x0[0]), &(_b[0]));
}

//-----------------------------------------------------------------------------

bool& EigenLDLTSolver::
show_timings()
{
  return show_timings_;
}


}//namespace COMISO

//////////////////////////////////////////////////////////////////////////
// explicit instantiation

namespace COMISO
{

#if COMISO_GMM_AVAILABLE
template bool EigenLDLTSolver::update_system_gmm(const gmm::csc_matrix<double>&);
template bool EigenLDLTSolver::calc_system_gmm(const gmm::csc_matrix<double>&);
#endif // COMISO_GMM_AVAILABLE

template bool EigenLDLTSolver::update_system_eigen(const Eigen::SparseMatrix<double>&);
template bool EigenLDLTSolver::calc_system_eigen(const Eigen::SparseMatrix<double>&);

}//namespace COMISO

//=============================================================================
#endif // COMISO_EIGEN3_AVAILABLE
//=============================================================================
