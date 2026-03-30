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



//=============================================================================
//
//  CLASS GMM_Tools - IMPLEMENTATION
//
//=============================================================================

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
#if COMISO_GMM_AVAILABLE

//== INCLUDES =================================================================

#include "GMM_ToolsT_impl.hh"

#include <CoMISo/Config/GmmTypes.hh>
#include <CoMISo/Config/StdTypes.hh>

// explicit instantiation

namespace COMISO_GMM
{
using namespace COMISO_STD;

template void factored_to_quadratic(const WSRowMatrix&, WSColMatrix&, DoubleVector&);
template void factored_to_quadratic(const WSRowMatrix&, RSColMatrix&, DoubleVector&);
template void factored_to_quadratic(const RSRowMatrix&, RSColMatrix&, DoubleVector&);

template void factored_to_quadratic_eigen(const WSRowMatrix&, WSColMatrix&, DoubleVector&);
template void factored_to_quadratic_eigen(const WSRowMatrix&, RSColMatrix&, DoubleVector&);
template void factored_to_quadratic_eigen(const RSRowMatrix&, RSColMatrix&, DoubleVector&);

template void eliminate_csc_vars(const IntVector&, const DoubleVector&,
  CSCMatrix&, DoubleVector&, DoubleVector&);

template void eliminate_csc_vars2(const IntVector&, const DoubleVector&,
  CSCMatrix&, DoubleVector&, DoubleVector&);

template void eliminate_csc_vars2(const UIntVector&, const DoubleVector&,
  CSCMatrix&, DoubleVector&, DoubleVector&);

template double residuum_norm(CSCMatrix&, DoubleVector&, DoubleVector&);

template double residuum_norm(WSColMatrix&, DoubleVector&, DoubleVector&);

template void fix_var_csc_symmetric(
    const unsigned int, const double, CSCMatrix&, DoubleVector&, DoubleVector&);


template void write_matrix_ascii(
    const std::string& _filename, const WSRowMatrix& _m);

template void write_matrix_ascii(
    const std::string& _filename, const WSColMatrix& _m);

template void write_matrix_ascii(
    const std::string& _filename, const RSRowMatrix& _m);

template void write_matrix_ascii(
    const std::string& _filename, const RSColMatrix& _m);

template void read_matrix_ascii(
    const std::string& _filename, WSRowMatrix& _m);

template void write_vector_ascii(
    const std::string& _filename, const std::vector<double>& _v);

template void write_vector_ascii(
    const std::string& _filename, const std::vector<int>& _v);

template void read_vector_ascii(
    const std::string& _filename, std::vector<double>& _v);

template void read_vector_ascii(
    const std::string& _filename, std::vector<int>& _v);


template void write_matrix(
    const std::string& _filename, const WSRowMatrix& _m);

template void write_matrix(
    const std::string& _filename, const WSColMatrix& _m);

template void write_matrix(
    const std::string& _filename, const RSRowMatrix& _m);

template void write_matrix(
    const std::string& _filename, const RSColMatrix& _m);

template void read_matrix(const std::string& _filename, WSRowMatrix& _m);

template void read_matrix(const std::string& _filename, WSColMatrix& _m);

template void write_vector(
    const std::string& _filename, const std::vector<double>& _v);

template void write_vector(
    const std::string& _filename, const std::vector<int>& _v);

template void read_vector(
    const std::string& _filename, std::vector<double>& _v);

template void read_vector(
    const std::string& _filename, std::vector<int>& _v);

template void read_vector(
    const std::string& _filename, std::vector<unsigned int>& _v);



// specializations

template <>
void write_matrix_ascii(
    const std::string& _filename, const CSCMatrix& _m)
{
  if (_m.ncols() == 0 || _m.nrows() == 0)
    return;
  gmm::MatrixMarket_save(_filename.c_str(), _m);
}


template <>
void read_matrix_ascii(
    const std::string& _filename, WSColMatrix& _m)
{
  gmm::MatrixMarket_load(_filename.c_str(), _m);
}

template <>
void read_matrix(const std::string& _filename, CSCMatrix& _m)
{
  using Scalar = typename gmm::linalg_traits<CSCMatrix>::value_type;

  Eigen::SparseMatrix<Scalar> m;
  COMISO_EIGEN::read_matrix(_filename, m);
  COMISO_EIGEN::eigen_to_gmm_csc(m, _m);
}


}//namespace COMISO_GMM


#endif // COMISO_GMM_AVAILABLE
