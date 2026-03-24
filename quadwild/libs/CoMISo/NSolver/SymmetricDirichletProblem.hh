/*===========================================================================*\
 *                                                                           *
 *                               CoMISo                                      *
 *      Copyright (C) 2008-2019 by Computer Graphics Group, RWTH Aachen      *
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
//  CLASS SymmetricDirichletProblem
//
//=============================================================================


#ifndef COMISO_SYMMETRICDIRICHLETTPROBLEM_HH
#define COMISO_SYMMETRICDIRICHLETTPROBLEM_HH


//== INCLUDES =================================================================

#include <CoMISo/Config/config.hh>

#if (COMISO_ADOLC_AVAILABLE && COMISO_EIGEN3_AVAILABLE)

#include <CoMISo/Config/CoMISoDefines.hh>
#include "FiniteElementProblem.hh"

#include <adolc/adolc.h>
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/sparse/sparsedrivers.h>
#include <adolc/taping.h>
#include <adolc/drivers/taylor.h>
#include <CoMISo/NSolver/TapeIDSingleton.hh>


//== FORWARDDECLARATIONS ======================================================

//== NAMESPACES ===============================================================

namespace COMISO { 

//== CLASS DEFINITION =========================================================

class SymmetricDirichletElement
{
public:

  // define dimensions
  const static int NV = 6; // the three u and v coordinates. u1, v1, u2, v2, u3, v3
  const static int NC = 6; // the three reference positions of the triangle, x1, y1, x2, y2, x3, y3

  typedef Eigen::Matrix<size_t,NV,1> VecI;
  typedef Eigen::Matrix<double,NV,1> VecV;
  typedef Eigen::Matrix<double,NC,1> VecC;
  typedef Eigen::Triplet<double> Triplet;

  typedef Eigen::Matrix<double,12,1> Vector12;
  typedef Eigen::Matrix<adouble,2,2> Matrix2x2ad;
  typedef Eigen::Matrix<double,6,6> Matrix6x6;

  SymmetricDirichletElement()
    :
      tape_(-1),
      tape_available_(false)
  {
    init_tape();
  }

  double eval_f       (const VecV& _x, const VecC& _c);
  void   eval_gradient(const VecV& _x, const VecC& _c, VecV& _g);
  void   eval_hessian (const VecV& _x, const VecC& _c, std::vector<Triplet>& _triplets);

  void project_hessian(Eigen::MatrixXd& H_orig, Eigen::MatrixXd& H_spd, double eps);

  double max_feasible_step(const VecV& _x, const VecV& _v, const VecC& /*_c*/);

  adouble f_adouble( const adouble* _x);

  void retape();
  void init_tape();

private:

  // index of tape
  short int tape_;

  // taping already done?
  bool tape_available_;
};


/** \class SymmetricDirichletProblem

    A problem that allows you to add triangles with reference positions for which
    the symmetric dirichlet energy should be minimized
*/

class COMISODLLEXPORT SymmetricDirichletProblem : public FiniteElementProblem
{
public:

  typedef FiniteElementSet<SymmetricDirichletElement> SymmetricDirichletElementSet;

  typedef Eigen::Matrix<size_t,3,1> IndexVector;
  typedef Eigen::Matrix<double,3,2> ReferencePositionVector2D;
  typedef Eigen::Matrix<double,3,3> ReferencePositionVector3D;

  typedef Eigen::VectorXd             VectorD;
  typedef Eigen::SparseMatrix<double> SMatrixD;


  /// Default constructor
  SymmetricDirichletProblem(const unsigned int _n_vertices);

  void add_triangle(const IndexVector& _vertex_indices, const ReferencePositionVector2D& _reference_positions);

  /// add fix point constraint. Note that the final constraints still need to be passed
  /// to the solver by you after retrieving them via get_constraints
  void add_fix_point_constraint(int _vertex_index, double _fix_u, double _fix_v);

  /// add fix point constraint for only one of the two coordinates of a vertex. Note that the
  /// final constraints still need to be passed to the solver by you after retrieving them via get_constraints
  void add_fix_coordinate_constraint(int _vertex_index, int _coordinate, double _fix_coordinate);

  /// Retrieve the constraint matrix and right hand side representing the added
  /// fix point and fix coordinate constraints for use e.g. with the NewtonSolver
  void get_constraints(SMatrixD& _A, VectorD& _b);

  /// remove all constraints
  void clear_constraints() { fix_points.clear();}

  /// get reference positions for an equilateral triangle with the given area
  ReferencePositionVector2D get_equilateral_refernce_positions(double _area = 1.0);

private:
  SymmetricDirichletElementSet element_set;

  std::vector<std::pair<int, double>> fix_points;
};


class SymmetricDirichletOneVertexElement
{
public:

  // define dimensions
  const static int NV = 2; // the one u and v coordinate of the single vertex that is optimized. u1, v1
  const static int NC = 10; // the two other positions of the triangle u2, v2, u3, and v3, and the three reference positions of the triangle, x1, y1, x2, y2, x3, y3

  typedef Eigen::Matrix<size_t,NV,1> VecI;
  typedef Eigen::Matrix<double,NV,1> VecV;
  typedef Eigen::Matrix<double,NC,1> VecC;
  typedef Eigen::Triplet<double> Triplet;

  typedef Eigen::Matrix<double,12,1> Vector12;
  typedef Eigen::Matrix<adouble,2,2> Matrix2x2ad;
  typedef Eigen::Matrix<double,2,2> Matrix2x2d;
  typedef Eigen::Matrix<double,6,6> Matrix6x6;

  SymmetricDirichletOneVertexElement(){}

  double eval_f       (const VecV& _x, const VecC& _c);
  void   eval_gradient(const VecV& _x, const VecC& _c, VecV& _g);
  void   eval_hessian (const VecV& _x, const VecC& _c, std::vector<Triplet>& _triplets);

  void project_hessian(Eigen::MatrixXd& H_orig, Eigen::MatrixXd& H_spd, double eps);

  double max_feasible_step(const VecV& _x, const VecV& _v, const VecC& /*_c*/);
};


/** \class SymmetricDirichletOneRingProblem

    A problem that allows you to add triangles with reference positions for which
    the symmetric dirichlet energy should be minimized. Secial case of SymmetricDirichletProblem
    that optimizes only a single vertex for which the user inputs all adjacent triangles.
*/

class COMISODLLEXPORT SymmetricDirichletOneRingProblem : public FiniteElementProblem
{
public:

  typedef FiniteElementSet<SymmetricDirichletOneVertexElement> SymmetricDirichletOneVertexElementSet;

  typedef Eigen::Matrix<size_t,3,1> IndexVector;
  typedef Eigen::Matrix<double,3,2> InputPositionVector2D;
  typedef Eigen::Matrix<double,3,2> ReferencePositionVector2D;
  typedef Eigen::Matrix<double,3,3> ReferencePositionVector3D;

  typedef Eigen::VectorXd             VectorD;
  typedef Eigen::SparseMatrix<double> SMatrixD;


  /// Default constructor
  SymmetricDirichletOneRingProblem();

  void add_triangle(const InputPositionVector2D& _current_positions, const ReferencePositionVector2D& _reference_positions);

  static ReferencePositionVector2D get_equilateral_refernce_positions(double _area = 1.0);

private:
  SymmetricDirichletOneVertexElementSet element_set;

  std::vector<std::pair<int, double>> fix_points;
};


//=============================================================================
} // namespace COMISO
//=============================================================================

#endif //(COMISO_ADOLC_AVAILABLE && COMISO_EIGEN3_AVAILABLE)

//=============================================================================
#endif // COMISO_SYMMETRICDIRICHLETPROBLEM_HH defined
//=============================================================================

