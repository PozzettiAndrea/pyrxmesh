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


#ifndef COMISO_GMM_TOOLS_HH
#define COMISO_GMM_TOOLS_HH


//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
#if COMISO_GMM_AVAILABLE

//== INCLUDES =================================================================

#include <string>
#include <iostream>
#include <vector>
#include <algorithm>

#include <CoMISo/Config/GmmTypes.hh>
#include <CoMISo/Utils/gmm.hh>

#if COMISO_SUITESPARSE_AVAILABLE
#include <cholmod.h>
#endif


//== FORWARDDECLARATIONS ======================================================

//== NAMESPACES ===============================================================

namespace COMISO_GMM
{

/** \class GMMTools GMM_Tools.hh

    A collection of helper functions for manipulating (gmm) matrices.
*/

//== FUNCTION DEFINITION ======================================================

/** @name Variable elimination
 * These functions are used to eliminate (one or more) variables x_i from an
 * equation system Ax=b. Elimination meaning that x_i has been assigned a value
 * x_i = c and is considered a constant, this changes entries of the matrix
 * which depend on i and finally "eliminates" the ith row and column and
 * updates the rhs. */
/*@{*/

/// Eliminate multiple variables from a CSC matrix.
/**
 *  \note eliminate_csc_vars2 is probably more efficient
 *  @param _elmn_vars indices of variables to be eliminated
 *  @param _elmn_vals values c_i of x_i to be eliminated, x_i = c_i
 *  @param _A CSC Matrix of the equation system
 *  @param _x variable vector of equation system
 *  @param _rhs right-hand side vector of equation system  */
template<class ScalarT, class VectorT, class RealT, class IntegerT>
void eliminate_csc_vars(
    const std::vector<IntegerT>&     _elmn_vars,
    const std::vector<ScalarT>&      _elmn_vals,
    typename gmm::csc_matrix<RealT>&  _A,
    VectorT&                          _x,
    VectorT&                          _rhs );

/// Eliminate variables from a CSC matric.
template<class ScalarT, class VectorT, class RealT, class IntegerT>
void eliminate_csc_vars2(
    const std::vector<IntegerT>&     _elmn_vars,
    const std::vector<ScalarT>&      _elmn_vals,
    typename gmm::csc_matrix<RealT>&  _A,
    VectorT&                          _x,
    VectorT&                          _rhs );

/// Eliminate variables from a CSC matric.
template<class ScalarT, class VectorT, class RealT, class IntegerT>
void eliminate_csc_vars2_eigen(
    const std::vector<IntegerT>&     _elmn_vars,
    const std::vector<ScalarT>&      _elmn_vals,
    typename gmm::csc_matrix<RealT>&  _A,
    VectorT&                          _x,
    VectorT&                          _rhs );

/// Eliminate only one variable x_i = c (CSC matrices)
/** Specialization to eliminating one varaible
 *  @param _j index of variable to be eliminated
 *  @param _value_j value c of x_i to be eliminated, x_i = c
 *  @param _A CSC Matrix of the equation system
 *  @param _x variable vector of equation system
 *  @param _rhs right-hand side vector of equation system */
template<class ScalarT, class VectorT, class RealT>
void eliminate_var(
    const unsigned int                _j,
    const ScalarT                     _value_j,
    typename gmm::csc_matrix<RealT>&  _A,
    VectorT&                          _x,
    VectorT&                          _rhs );



/// eliminate multiple variables from a (NON CSC) linear system by fixin x[j] = _value_j
/**
 *  @param _elmn_vars indices of variables to be eliminated
 *  @param _elmn_vals values c_i of x_i to be eliminated, x_i = c_i
 *  @param _A (non-CSC) Matrix of the equation system
 *  @param _x variable vector of equation system
 *  @param _rhs right-hand side vector of equation system */
template<class IntegerT, class ScalarT, class VectorT, class MatrixT>
void eliminate_vars(
    const std::vector<IntegerT>& _elmn_vars,
    const std::vector<ScalarT>&  _elmn_vals,
    MatrixT&                     _A,
    VectorT&                     _x,
    VectorT&                     _rhs );

/// Eliminate only one variable x_i = c (non-CSC matrices)
/** Specialization to eliminating one varaible
 *  @param _j index of variable to be eliminated
 *  @param _value_j value c of x_i to be eliminated, x_i = c
 *  @param _A (non-CSC) Matrix of the equation system
 *  @param _x variable vector of equation system
 *  @param _rhs right-hand side vector of equation system */
template<class ScalarT, class VectorT, class SVT>
void eliminate_var(
    const unsigned int     _j,
    const ScalarT          _value_j,
    gmm::col_matrix<SVT>&  _A,
    VectorT&               _x,
    VectorT&               _rhs );


/// do in-place elimination in CSC format by setting row and column to zero and
/// diagonal entry to one
/**
 *  @param _j index of variable to be eliminated
 *  @param _value_j value c of x_i to be eliminated, x_i = c
 *  @param _A (non-CSC) Matrix of the equation system
 *  @param _x variable vector of equation system
 *  @param _rhs right-hand side vector of equation system */
template<class ScalarT, class VectorT, class RealT>
void fix_var_csc_symmetric( const unsigned int                _j,
			    const ScalarT                     _value_j,
			    typename gmm::csc_matrix<RealT>&  _A,
			    VectorT&                          _x,
			    VectorT&                          _rhs );

// Same as above but use Eigen internally
template<class ScalarT, class VectorT, class RealT>
void fix_var_csc_symmetric_eigen( const unsigned int                _j,
			    const ScalarT                     _value_j,
			    typename gmm::csc_matrix<RealT>&  _A,
			    VectorT&                          _x,
			    VectorT&                          _rhs );


/*@}*/


/// Get matrix data (CSC matrix format) from matrix
/** Used by Cholmod wrapper
 *  @param _mat matrix
 *  @param _c uplo parameter (l, L, u, U, c, C)
 *  @param _values values vector
 *  @param _rowind row indices
 *  @param _colptr column pointer  */
template<class MatrixT, class REALT, class INTT>
void get_ccs_symmetric_data( const MatrixT&      _mat,
                             const char          _c,
                             std::vector<REALT>& _values,
                             std::vector<INTT>&  _rowind,
                             std::vector<INTT>&  _colptr );

/// Regularize matrix
/**  Makes matrices with rank(_mat)<n solvable.
  *  Add factor*avg(trace(_mat))*Identity to _mat.
 *  @param _mat Matrix to regularize
 *  @param _v factor in factor*avg(trace(_mat))*Identity  */
template<class MatrixT>
void regularize_hack( MatrixT& _mat, double _v = 1e-6 );


/// Local Gauss Seidel update of lin. equation system.
/**
 *  Add factor*avg(trace(_mat))*Identity to _mat.
 *  @param _A Matrix of linear system
 *  @param _x variable vector of linear system
 *  @param _rhs right hand side of linear system
 *  @param _max_iter maximum number of iterations
 *  @param _tolerance error tolerance threshold
 *  @return number of iterations performed */
template<class MatrixT, class VectorT>
int gauss_seidel_local(
    MatrixT&                  _A,
    VectorT&                  _x,
    VectorT&                  _rhs,
    std::vector<unsigned int> _idxs,
    int                       _max_iter = 10000,
    double                    _tolerance = 1e-6 );


/// Residuum norm of linear system
/** residuum = Ax-b
  * @param _A Matrix
  * @param _x Variables
  * @param _rhs right hand side
  * @return norm Ax-rhs */
template<class MatrixT, class VectorT>
double residuum_norm( MatrixT& _A, VectorT& _x, VectorT& _rhs );


/// Convert factored LSE to quadratic representation
/** Conversion is done by computing _F^t _F where the last column is the _rhs
  * @param _F Factored Matrix (input)
  * @param _Q Quadratic Matrix (output)
  * @param _rhs right hand side (output) */
template <class MatrixT, class MatrixT2, class VectorT>
void factored_to_quadratic(const MatrixT& _F, MatrixT2& _Q, VectorT& _rhs);

// same as above but internally uses Eigen
template <class MatrixT, class MatrixT2, class VectorT>
void factored_to_quadratic_eigen(const MatrixT& _F, MatrixT2& _Q, VectorT& _rhs);

template<class MatrixT, class VectorT>
  void factored_to_quadratic_rhs_only( MatrixT& _F, VectorT& _rhs);


/// Inspect the matrix (print)
/** Prints useful matrix informations such as, dimension, symmetry, zero_rows, zero_cols, nnz, max, min, max_abs, min_abs, NAN, INF
  * @param _A matrix */
template<class MatrixT>
void inspect_matrix( const MatrixT& _A);

/// Print the matrix as dense matrix
template<class MatrixT>
void print_dense( const MatrixT& _A);


#if COMISO_SUITESPARSE_AVAILABLE

/// GMM to Cholmod_sparse interface
template<class MatrixT>
void cholmod_to_gmm( const cholmod_sparse& _AC, MatrixT& _A);

template<class MatrixT>
void gmm_to_cholmod( const MatrixT&  _A,
                     cholmod_sparse* &_AC,
                     cholmod_common* _common,
                     int             _sparsity_type = 0,
                     bool            _long_int      = false);
#endif


// GMM used to put operator<< for std::vector<T> into namespace std which was used in the CoMISo examples
// Newer versions do not do that anymore.
// The functions below allows to still use operator<< both with the older and newer GMM by putting
// using COMISO_GMM::operator<<; at the start of the scope.
inline std::ostream& operator <<(std::ostream &o, const std::vector<double>& m) { gmm::write(o,m); return o; }
inline std::ostream& operator <<(std::ostream &o, const std::vector<size_t>& m) { gmm::write(o,m); return o; }
inline std::ostream& operator <<(std::ostream &o, const std::vector<int>& m)    { gmm::write(o,m); return o; }


// Write a gmm matrix in MatrixMarket format
template <typename MatrixT>
void write_matrix_ascii(const std::string& _filename, const MatrixT& _m);

// Specialization of method above for csc matrices
template <>
void write_matrix_ascii(const std::string& _filename, const CSCMatrix& _m);

// Load a gmm matrix from MatrixMarket format
template <typename MatrixT>
void read_matrix_ascii(const std::string& _filename, MatrixT& _m);

// Specialization of method above for WSColMatrix
template <>
void read_matrix_ascii(const std::string& _filename, WSColMatrix& _m);

// Write a gmm vector as a nx1 matrix in MatrixMarket format
template <typename VectorT>
void write_vector_ascii(const std::string& _filename, const VectorT& _v);

// Load a vector (or rather a nx1 matrix, see write_gmm_vector_ascii) from
// MatrixMarket format
template <typename T>
void read_vector_ascii(const std::string& _filename, std::vector<T>& _v);


// Write a gmm matrix in our custom binary format
template <typename MatrixT>
void write_matrix(const std::string& _filename, const MatrixT& _m);

// Load a gmm matrix from our custom binary format
template <typename MatrixT>
void read_matrix(const std::string& _filename, MatrixT& _m);

template <>
void read_matrix(const std::string& _filename, CSCMatrix& _m);

// Write a gmm vector as a nx1 matrix in our custom binary format
template <typename VectorT>
void write_vector(const std::string& _filename, const VectorT& _v);

// Load a vector (or rather a nx1 matrix, see write_eigen_vector) from
// our custom binary format
template <typename T>
void read_vector(const std::string& _filename, std::vector<T>& _v);


//=============================================================================
} // namespace COMISO_GMM
//=============================================================================
#if defined(INCLUDE_TEMPLATES) && !defined(COMISO_GMM_TOOLS_C)
#define COMISO_GMM_TOOLS_TEMPLATES
#include "GMM_ToolsT_impl.hh"
#endif

#endif // COMISO_GMM_AVAILABLE

//=============================================================================
#endif // GMM_GMM_TOOLS_HH defined
//=============================================================================

