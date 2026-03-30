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


#ifndef COMISO_Eigen_TOOLS_HH
#define COMISO_Eigen_TOOLS_HH


//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
#if COMISO_EIGEN3_AVAILABLE

//== INCLUDES =================================================================

#include <Base/Code/Quality.hh>
#include <Base/Debug/DebError.hh>
#include <Base/Debug/DebOut.hh>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

LOW_CODE_QUALITY_SECTION_BEGIN
#include <Eigen/Dense>
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Sparse>
LOW_CODE_QUALITY_SECTION_END

#if COMISO_SUITESPARSE_AVAILABLE
#include <cholmod.h>
#endif


//== FORWARDDECLARATIONS ======================================================

//== NAMESPACES ===============================================================

namespace COMISO_EIGEN
{

/** \class EigenTools Eigen_Tools.hh

    A collection of helper functions for manipulating (Eigen) matrices.
*/


// A matrix class based on Eigen that stores a matrix as an std::vector of
// Eigen::SparseVector. This class is meant to be used when one works on a
// sparse matrix but wants to often change elements, e.g. performing gaussian
// elimination. One Eigen::SparseMatrix classes such operations are slow as
// elements are ordered and element insertion thus requires frequent memory
// reallocation and shifting of elements. With this class, these effects are
// limited to individual rows/columns and is therefore a lot faster.
// This class supports some of the functions to access elements that
// that Eigen::SparseMatrix also offers, but it does not support many other
// operations such as multiplication, transpose, etc.
template <typename ScalarT, int OPTIONS, typename StorageT>
class HalfSparseMatrixBase
{
public:
  static constexpr int ORDERING = OPTIONS;
  static constexpr int OTHER_ORDERING = 1 - ORDERING;
  using Matrix      = Eigen::SparseMatrix<ScalarT, ORDERING, StorageT>;
  using OtherMatrix = Eigen::SparseMatrix<ScalarT, OTHER_ORDERING, StorageT>;
  using SparseVector = Eigen::SparseVector<ScalarT>;
  using Index = Eigen::Index;

  HalfSparseMatrixBase() { inner_size_ = 0; }
  HalfSparseMatrixBase(size_t _outer_size, size_t _inner_size);
  HalfSparseMatrixBase(const Matrix& _mat);
  HalfSparseMatrixBase(const OtherMatrix& _mat);

  template <typename Scalar2T, typename Storage2T>
  HalfSparseMatrixBase(
      const HalfSparseMatrixBase<Scalar2T, OTHER_ORDERING, Storage2T>& _mat);

  HalfSparseMatrixBase operator=(const Matrix& _mat)
  {
    std::swap(HalfSparseMatrixBase(_mat), *this);
    return *this;
  }

  operator Matrix() const;

  const SparseVector& innerVector(Index _index) const { return mat_[_index]; }

  int outerSize() const { return (int)mat_.size(); }
  int innerSize() const { return inner_size_; }

  void outerResize(size_t _size)
  {
    mat_.resize(_size, SparseVector(inner_size_));
  }

  void innerResize(size_t _size)
  {
    inner_size_ = static_cast<int>(_size);
    for (auto& v : mat_)
      v.resize(_size);
  }

  void innerConservativeResize(size_t _size)
  {
    inner_size_ = static_cast<int>(_size);
    for (auto& v : mat_)
      v.conservativeResize(_size);
  }

  void prune(double _eps)
  {
    for (auto& vec : mat_)
      vec.prune(_eps);
  }

protected:
  std::vector<SparseVector> mat_;
  int inner_size_;
};

// forward declaration
template <typename ScalarT> class HalfSparseRowMatrix;

template <typename ScalarT>
class HalfSparseColMatrix: public HalfSparseMatrixBase<ScalarT, Eigen::ColMajor, int>
{
public:

  using Base = HalfSparseMatrixBase<ScalarT, Eigen::ColMajor, int>;
  using SparseVector = typename Base::SparseVector;
  using Index = typename Base::Index;
  using Matrix = Eigen::SparseMatrix<ScalarT, Eigen::ColMajor, int>;
  using OtherMatrix = Eigen::SparseMatrix<ScalarT, Eigen::RowMajor, int>;

  HalfSparseColMatrix() : Base() {}
  HalfSparseColMatrix(int _rows, int _cols) : Base(_cols, _rows) {}
  HalfSparseColMatrix(const Matrix& _m) : Base(_m) {}
  HalfSparseColMatrix(const OtherMatrix& _m) : Base(_m) {}
  HalfSparseColMatrix(const HalfSparseRowMatrix<ScalarT>& _m) : Base(_m) {}

        SparseVector& col(Index _col)       { return this->mat_[_col]; }
  const SparseVector& col(Index _col) const { return this->mat_[_col]; }

  ScalarT& coeffRef(Index _row, Index _col)       { return this->mat_[_col].coeffRef(_row); }
  ScalarT  coeff   (Index _row, Index _col) const { return this->mat_[_col].coeff   (_row); }

  int cols() const { return (int)this->mat_.size(); }
  int rows() const { return this->inner_size_; }
};


template <typename ScalarT>
class HalfSparseRowMatrix : public HalfSparseMatrixBase<ScalarT, Eigen::RowMajor, int>
{
public:
  using Base = HalfSparseMatrixBase<ScalarT, Eigen::RowMajor, int>;
  using SparseVector = typename Base::SparseVector;
  using Index = typename Base::Index;
  using Matrix = Eigen::SparseMatrix<ScalarT, Eigen::RowMajor, int>;
  using OtherMatrix = Eigen::SparseMatrix<ScalarT, Eigen::ColMajor, int>;

  HalfSparseRowMatrix() : Base() {}
  HalfSparseRowMatrix(size_t _rows, size_t _cols) : Base(_rows, _cols) {}
  HalfSparseRowMatrix(const Matrix& _m) : Base(_m) {}
  HalfSparseRowMatrix(const OtherMatrix& _m) : Base(_m) {}
  HalfSparseRowMatrix(const HalfSparseColMatrix<ScalarT>& _m) : Base(_m) {}

  Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> col(Index _col) const
  {
    const auto size = this->mat_.size();
    Eigen::Matrix<ScalarT, Eigen::Dynamic, 1> vec(size);
    for (size_t i = 0; i < size; ++i)
      vec[i] = this->mat_[i].coeff(_col);
    return vec;
  }

        SparseVector& row(Index _row)       { return this->mat_[_row]; }
  const SparseVector& row(Index _row) const { return this->mat_[_row]; }

  ScalarT& coeffRef(Index _row, Index _col)       { return this->mat_[_row].coeffRef(_col); }
  ScalarT  coeff   (Index _row, Index _col) const { return this->mat_[_row].coeff   (_col); }

  int cols() const { return this->inner_size_; }
  int rows() const { return (int)this->mat_.size(); }
};



//== FUNCTION DEFINITION ======================================================

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

/// Inspect the matrix (print)
/** Prints useful matrix informations such as, dimension, symmetry, zero_rows, zero_cols, nnz, max, min, max_abs, min_abs, NAN, INF
  * @param _A matrix */
template<class MatrixT>
void inspect_matrix( const MatrixT& _A);

/** checks for symmetry
  * @param _A matrix
  * @return symmetric? (bool)*/
template<class MatrixT>
bool is_symmetric( const MatrixT& _A);

template< class Eigen_MatrixT, class IntT >
void permute( const Eigen_MatrixT& _QR, const std::vector< IntT>& _Pvec, Eigen_MatrixT& _A);

// Count non zero elements in row of a matrix.
// This operation is linear in time, but ignores all zero values.
// In contrast, Eigen's nonZeros method is constant time, but includes all
// entries stored in the matrix, also zero entries (e.g. after adding and
// subtracting 1 from a coefficient)
// _ignore_last_element allows to ignore the last element which in CoMISo
// often represents the rhs of an equation and may not need to be included in
// the count.
template <class ScalarT, class StorageT>
int count_non_zeros_in_row(
    const Eigen::SparseMatrix<ScalarT, Eigen::RowMajor, StorageT>& _mat,
    int _row, bool _ignore_last_element);

// same as above but on a sparse vector
template <class ScalarT>
int count_non_zeros(
    const Eigen::SparseVector<ScalarT>& _vec, bool _ignore_last_element);


/// Residuum norm of linear system
/** residuum = Ax-b
 * @param _A Matrix
 * @param _x Variables
 * @param _rhs right hand side
 * @return norm Ax-rhs */
template <class MatrixT, class VectorT>
double residuum_norm(const MatrixT& _A, const VectorT& _x, const VectorT& _rhs);

/// Convert factored LSE to quadratic representation
/** Conversion is done by computing _F^t _F where the last column is the _rhs
 * @param _F Factored Matrix (input)
 * @param _Q Quadratic Matrix (output)
 * @param _rhs right hand side (output) */
template <class ScalarT, int OPTIONS1, class Storage1T, int OPTIONS2,
    class Storage2T>
void factored_to_quadratic(
    const Eigen::SparseMatrix<ScalarT, OPTIONS1, Storage1T>& _F,
          Eigen::SparseMatrix<ScalarT, OPTIONS2, Storage2T>& _Q,
          Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>& _rhs);

/// Eliminate multiple variables from a CSC matrix.
/**
 *  @param _elmn_vars indices of variables to be eliminated
 *  @param _elmn_vals values c_i of x_i to be eliminated, x_i = c_i
 *  @param _A CSC Matrix of the equation system
 *  @param _x variable vector of equation system
 *  @param _rhs right-hand side vector of equation system  */
template <class ScalarT, class IntegerT>
void eliminate_csc_vars(const std::vector<IntegerT>& _elmn_vars,
    const std::vector<ScalarT>& _elmn_vals,
    Eigen::SparseMatrix<ScalarT, Eigen::ColMajor>& _A,
    Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>& _x,
    Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>& _rhs);

/// Same as above but operate on data buffers
template <class ScalarT, class Integer1T, class Integer2T>
void eliminate_csc_vars(const std::vector<Integer1T>& _elmn_vars,
    const int _rows, const ScalarT* const _val_src,
    const Integer2T* const _rows_src, const Integer2T* const _cols_src,
    ScalarT* const _val_dst, Integer2T* const _rows_dst,
    Integer2T* const _cols_dst);

/// Same as above but input and output buffers are the same, i.e. work in-place.
template <class ScalarT, class Integer1T, class Integer2T>
void eliminate_csc_vars(const std::vector<Integer1T>& _elmn_vars,
    const int _n_rows, ScalarT* const _val, Integer2T* const _rows,
    Integer2T* const _cols);

/// Create a map for _n_vars variables to their new position if variables in
/// _elmn_vars are eliminated
template <class IntegerT>
std::vector<int> make_new_index_map(
    const std::vector<IntegerT>& _elmn_vars, int _n_vars);


/// do in-place elimination in CSC format by setting row and column to zero and
/// diagonal entry to one
/**
 *  @param _i index of variable to be eliminated
 *  @param _xi value the eliminated variable to set to
 *  @param _A CSC Matrix of the equation system
 *  @param _x variable vector of equation system
 *  @param _rhs right-hand side vector of equation system */
template <class ScalarT, class RealT>
void fix_var_csc_symmetric(const unsigned int _i, const ScalarT _xi,
    Eigen::SparseMatrix<RealT, Eigen::ColMajor>& _A,
    Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>& _x,
    Eigen::Matrix<ScalarT, Eigen::Dynamic, 1>& _rhs);

// same as above but operate directly on csc storage buffers
// See https://en.wikipedia.org/wiki/Sparse_matrix
// for description of csc format.
template <class ScalarT, class IntegerT, class RealT>
void fix_var_csc_symmetric(const unsigned int _i, const ScalarT _xi,
    RealT* const _val, IntegerT* const _rows, IntegerT* const _cols,
    ScalarT* const _x, ScalarT* const _rhs);

#if COMISO_SUITESPARSE_AVAILABLE

/// Eigen to Cholmod_sparse interface
template<class MatrixT>
void cholmod_to_eigen( const cholmod_sparse& _AC, MatrixT& _A);

template<class MatrixT>
void eigen_to_cholmod( const MatrixT&  _A,
                     cholmod_sparse* &_AC,
                     cholmod_common* _common,
                     int             _sparsity_type = 0,
                     bool            _long_int      = false);
#endif

#if COMISO_GMM_AVAILABLE
// convert a gmm column-sparse matrix into an Eigen sparse matrix
template <class GMM_MatrixT, class EIGEN_MatrixT>
void gmm_to_eigen(const GMM_MatrixT& _G, EIGEN_MatrixT& _E);

template <class GMM_VectorT, class EIGEN_VectorT>
void to_eigen_vec(const GMM_VectorT& _G, EIGEN_VectorT& _E);
#endif // COMISO_GMM_AVAILABLE

template <class ScalarT, class EIGEN_VectorT>
void to_eigen_vec(const std::vector<ScalarT>& _G, EIGEN_VectorT& _E);

#if COMISO_GMM_AVAILABLE
// convert an Eigen sparse matrix into a gmm sparse matrix
template <class EIGEN_MatrixT, class GMM_MatrixT>
void eigen_to_gmm(const EIGEN_MatrixT& _E, GMM_MatrixT& _G);

// convert an Eigen sparse matrix into a gmm csc matrix
template <class EIGEN_MatrixT, class GMM_CSC_MatrixT>
void eigen_to_gmm_csc(const EIGEN_MatrixT& _E, GMM_CSC_MatrixT& _G);

// convert an Eigen csc matrix into a gmm csc matrix
// Expects that the buffers of the gmm csc matrix are already big enough
template <class ScalarT, class GMM_CSC_MatrixT>
void eigen_to_gmm_csc(const Eigen::SparseMatrix<ScalarT, Eigen::ColMajor>& _E, GMM_CSC_MatrixT& _G);

template <class EIGEN_VectorT, class GMM_VectorT>
void from_eigen_vec(const EIGEN_VectorT& _E, const GMM_VectorT& _G);
#endif // COMISO_GMM_AVAILABLE

template <class EIGEN_VectorT, class ScalarT>
void from_eigen_vec(const EIGEN_VectorT& _E, std::vector<ScalarT>& _v);


inline void fill_random(std::vector<double>& _x)
{
  from_eigen_vec(Eigen::VectorXd::Random(_x.size()), _x);
}

// Write a matrix in MatrixMarket format
template <typename MatrixT>
void write_matrix_ascii(const std::string& _filename, const MatrixT& _m);

// Load a matrix from MatrixMarket format
template <typename MatrixT>
void read_matrix_ascii(const std::string& _filename, MatrixT& _m);

// Load a sparse matrix from MatrixMarket format
template <typename ScalarT, int OPTIONS, typename StorageIndexT>
void read_matrix_ascii(const std::string& _filename,
    Eigen::SparseMatrix<ScalarT, OPTIONS, StorageIndexT>& _m);

// Write a vector in MatrixMarket format
template <typename VectorT>
void write_vector_ascii(const std::string& _filename, const VectorT& _v);

// Load a vector from MatrixMarket format
template <typename VectorT>
void read_vector_ascii(const std::string& _filename, VectorT& _v);

// Write a sparse matrix in our custom compact storage file format
template <typename ScalarT, int OPTIONS, typename StorageIndexT>
void write_matrix(const std::string& _filename,
    const Eigen::SparseMatrix<ScalarT, OPTIONS, StorageIndexT>& _m);

// Load a sparse matrix from our custom compact storage file format
template <typename ScalarT, int OPTIONS, typename StorageIndexT>
void read_matrix(const std::string& _filename,
    Eigen::SparseMatrix<ScalarT, OPTIONS, StorageIndexT>& _m);

// Write a dense matrix in our custom file format
template <typename ScalarT, int ROWS, int COLS>
void write_matrix(const std::string& _filename,
    const Eigen::Matrix<ScalarT, ROWS, COLS>& _m);

// Load a dense matrix in our custom file format
template <typename ScalarT, int ROWS, int COLS>
void read_matrix(const std::string& _filename,
    Eigen::Matrix<ScalarT, ROWS, COLS>& _m);


// Write a matrix in MatrixMarket format
template <typename MatrixT>
void write_matrix_ascii(const std::string& _filename, const MatrixT& _m);

// Load a matrix from MatrixMarket format
template <typename MatrixT>
void read_matrix_ascii(const std::string& _filename, MatrixT& _m);

// Load a sparse matrix from MatrixMarket format
template <typename ScalarT, int OPTIONS, typename StorageIndexT>
void read_matrix_ascii(const std::string& _filename,
    Eigen::SparseMatrix<ScalarT, OPTIONS, StorageIndexT>& _m);

// Write a vector in MatrixMarket format
template <typename VectorT>
void write_vector_ascii(const std::string& _filename, const VectorT& _v);

// Load a vector from MatrixMarket format
template <typename VectorT>
void read_vector_ascii(const std::string& _filename, VectorT& _v);

// Write a sparse matrix in our custom compact storage file format
template <typename ScalarT, int OPTIONS, typename StorageIndexT>
void write_matrix(const std::string& _filename,
    const Eigen::SparseMatrix<ScalarT, OPTIONS, StorageIndexT>& _m);

// Load a sparse matrix from our custom compact storage file format
template <typename ScalarT, int OPTIONS, typename StorageIndexT>
void read_matrix(const std::string& _filename,
    Eigen::SparseMatrix<ScalarT, OPTIONS, StorageIndexT>& _m);

// Write a dense matrix in our custom file format
template <typename ScalarT, int ROWS, int COLS>
void write_matrix(const std::string& _filename,
    const Eigen::Matrix<ScalarT, ROWS, COLS>& _m);

// Load a dense matrix in our custom file format
template <typename ScalarT, int ROWS, int COLS>
void read_matrix(const std::string& _filename,
    Eigen::Matrix<ScalarT, ROWS, COLS>& _m);



//=============================================================================
} // namespace COMISO_Eigen
//=============================================================================
#define COMISO_Eigen_TOOLS_TEMPLATES
#include "Eigen_ToolsT_impl.hh"

//=============================================================================
#endif // COMISO_EIGEN3_AVAILABLE
//=============================================================================
#endif // Eigen_TOOLS_HH defined
//=============================================================================

