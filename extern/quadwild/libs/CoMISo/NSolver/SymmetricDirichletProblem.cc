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

#include <CoMISo/Config/config.hh>

#if (COMISO_ADOLC_AVAILABLE && COMISO_EIGEN3_AVAILABLE)

#include "SymmetricDirichletProblem.hh"


namespace COMISO { 

double SymmetricDirichletElement::eval_f(const VecV& _x, const VecC& _c)
{
  double y = 0.0;

  Vector12 x;
  x << _x[0], _x[1], _x[2], _x[3], _x[4], _x[5],
       _c[0], _c[1], _c[2], _c[3], _c[4], _c[5];

  if(tape_available_)
  {
    int ec = function(tape_, 1, 12, const_cast<double*>(x.data()), &y);

#ifdef ADOLC_RET_CODES
    std::cout << "Info: function() returned code " << ec << std::endl;
#endif

    // tape not valid anymore? retape and evaluate again
    if(ec < 0)
    {
      retape();
      function(tape_, 1, 12, const_cast<double*>(x.data()), &y);
    }
  }
  else
  {
    retape();
    function(tape_, 1, 12, const_cast<double*>(x.data()), &y);
  }

  return y;
}

void SymmetricDirichletElement::eval_gradient(const VecV& _x, const VecC& _c, VecV& _g)
{
  Vector12 x;
  x << _x[0], _x[1], _x[2], _x[3], _x[4], _x[5],
       _c[0], _c[1], _c[2], _c[3], _c[4], _c[5];

  Vector12 grad;
  grad.setZero();

  // evaluate gradient
  int ec = gradient(tape_, 12, x.data(), grad.data());

  // check if retaping is required
  if(ec < 0)
  {
#ifdef ADOLC_RET_CODES
  std::cout << __FUNCTION__ << " invokes retaping of function due to discontinuity! Return code: " << ec << std::endl;
#endif
    retape();
    gradient(tape_, 12, x.data(), grad.data());
  }

  _g = grad.block(0,0,6,1);
}

void SymmetricDirichletElement::eval_hessian(const VecV& _x, const VecC& _c, std::vector<Triplet>& _triplets)
{
  Vector12 x;
  x << _x[0], _x[1], _x[2], _x[3], _x[4], _x[5],
       _c[0], _c[1], _c[2], _c[3], _c[4], _c[5];

  // dense hessian
  {
    auto dense_hessian = new double*[12];
    for(int i = 0; i < 12; ++i)
      dense_hessian[i] = new double[i+1];

    int ec = hessian(tape_, 12, const_cast<double*>(x.data()), dense_hessian);

    if(ec < 0) {
#ifdef ADOLC_RET_CODES
      std::cout << __FUNCTION__ << " invokes retaping of function due to discontinuity! Return code: " << ec << std::endl;
#endif
      // Retape function if return code indicates discontinuity
      retape();
      ec = hessian(tape_, 12, const_cast<double*>(x.data()), dense_hessian);
    }

#ifdef ADOLC_RET_CODES
    std::cout << "Info: hessian() returned code " << ec << std::endl;
#endif

    Eigen::MatrixXd H(6,6);
    for (int i = 0; i < 6; ++i)
    {
      H(i,i) = dense_hessian[i][i];
      for (int j = 0; j < i; ++j)
      {
        H(i,j) = dense_hessian[i][j];
        H(j,i) = dense_hessian[i][j];
      }
    }

    Eigen::MatrixXd Hspd(6,6);
    project_hessian(H, Hspd, 1e-6);
//    Hspd = H;

    _triplets.reserve(6*6);
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j)
        _triplets.push_back(Triplet(i,j,Hspd(i,j)));

    for (int i = 0; i < 6; ++i)
      delete dense_hessian[i];
    delete[] dense_hessian;
  }
}

void SymmetricDirichletElement::project_hessian(Eigen::MatrixXd& H_orig, Eigen::MatrixXd& H_spd, double eps)
{
  // Compute eigen-decomposition (of symmetric matrix)
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H_orig);
  Eigen::MatrixXd V = eig.eigenvectors();
  Eigen::MatrixXd D = eig.eigenvalues().asDiagonal();

  // Clamp all eigenvalues to eps
  for (int i = 0; i < H_orig.rows(); ++i)
      D(i, i) = std::max(eps, D(i, i));
  H_spd = V * D * V.inverse();
}

double SymmetricDirichletElement::max_feasible_step(const VecV& _x, const VecV& _v, const VecC&)
{
  // get quadratic coefficients (ax^2 + b^x + c)
  auto U11 = _x[0];
  auto U12 = _x[1];
  auto U21 = _x[2];
  auto U22 = _x[3];
  auto U31 = _x[4];
  auto U32 = _x[5];

  auto V11 = _v[0];
  auto V12 = _v[1];
  auto V21 = _v[2];
  auto V22 = _v[3];
  auto V31 = _v[4];
  auto V32 = _v[5];

  double a = V11*V22 - V12*V21 - V11*V32 + V12*V31 + V21*V32 - V22*V31;
  double b = U11*V22 - U12*V21 - U21*V12 + U22*V11 - U11*V32 + U12*V31 + U31*V12 - U32*V11 + U21*V32 - U22*V31 - U31*V22 + U32*V21;
  double c = U11*U22 - U12*U21 - U11*U32 + U12*U31 + U21*U32 - U22*U31;

  double delta_in = pow(b,2) - 4*a*c;
  if (delta_in < 0) {
    return std::numeric_limits<double>::max();
  }
  double delta = sqrt(delta_in);
  double t1 = (-b + delta)/ (2*a);
  double t2 = (-b - delta)/ (2*a);

  double tmp_n = std::min(t1,t2);
  t1 = std::max(t1,t2); t2 = tmp_n;
  // return the smallest negative root if it exists, otherwise return infinity
  if (t1 > 0)
  {
    if (t2 > 0)
    {
      return 0.999 * t2;
    }
    else
    {
      return 0.999 * t1;
    }
  }
  else
  {
    return std::numeric_limits<double>::max();
  }
}

adouble SymmetricDirichletElement::f_adouble(const adouble* _x)
{
  Matrix2x2ad B;
  B(0,0) = _x[2]-_x[0];
  B(0,1) = _x[4]-_x[0];
  B(1,0) = _x[3]-_x[1];
  B(1,1) = _x[5]-_x[1];
  Matrix2x2ad Bin = B.inverse();

  Matrix2x2ad R;
  R(0,0) = _x[6+2]-_x[6+0];
  R(0,1) = _x[6+4]-_x[6+0];
  R(1,0) = _x[6+3]-_x[6+1];
  R(1,1) = _x[6+5]-_x[6+1];
  Matrix2x2ad Rin = R.inverse();

  adouble area = 0.5 * R.determinant();
  if (B.determinant() * area <= 0)
  {
    adouble res = std::numeric_limits<double>::max();
    return res;
  }

  Matrix2x2ad J   =  B * Rin;
  Matrix2x2ad Jin =  R * Bin;

  adouble res = J.squaredNorm() + Jin.squaredNorm();

  return area * (res - 4);
}

void SymmetricDirichletElement::retape()
{
  std::cerr << "re-tape..." << std::endl;
  adouble y_d = 0.0;
  double y = 0.0;


  trace_on(tape_); // Start taping
  adouble* xa = new adouble[12];

  // Fill data vectors
  xa[0] <<= 0.0;
  xa[1] <<= 0.0;
  xa[2] <<= 1.0;
  xa[3] <<= 0.0;
  xa[4] <<= 0.0;
  xa[5] <<= 1.0;

  xa[6+0] <<= 0.0;
  xa[6+1] <<= 0.0;
  xa[6+2] <<= 1.0;
  xa[6+3] <<= 0.0;
  xa[6+4] <<= 0.0;
  xa[6+5] <<= 1.0;

  y_d = f_adouble(xa);

  y_d >>= y;

  trace_off();

#ifdef ADOLC_STATS
  print_stats();
#endif

  delete[] xa;
}

void SymmetricDirichletElement::init_tape()
{
  static bool tape_initialized = false;
  static short int tape;

  if (!tape_initialized)
  {
    tape = static_cast<short int>(TapeIDSingleton::Instance()->requestId());
    tape_ = tape;
    retape();
  }

  tape_initialized = true;
  tape_available_ = true;
  tape_ = tape;
}

/// Default constructor
SymmetricDirichletProblem::SymmetricDirichletProblem(const unsigned int _n_vertices)
  :
    FiniteElementProblem(2*_n_vertices)
{
  FiniteElementProblem::add_set(&element_set);
}

void SymmetricDirichletProblem::add_triangle(const IndexVector& _vertex_indices, const ReferencePositionVector2D& _reference_positions)
{
  SymmetricDirichletElement::VecI indices;
  indices << 2*_vertex_indices[0], 2*_vertex_indices[0]+1,
             2*_vertex_indices[1], 2*_vertex_indices[1]+1,
             2*_vertex_indices[2], 2*_vertex_indices[2]+1;
  SymmetricDirichletElement::VecC constants;
  constants << _reference_positions(0, 0), _reference_positions(0, 1),
               _reference_positions(1, 0), _reference_positions(1, 1),
               _reference_positions(2, 0), _reference_positions(2, 1);
  element_set.instances().add_element(indices, constants);
}

void SymmetricDirichletProblem::add_fix_point_constraint(int _vertex_index, double _fix_u, double _fix_v)
{
  add_fix_coordinate_constraint(_vertex_index, 0, _fix_u);
  add_fix_coordinate_constraint(_vertex_index, 1, _fix_v);
}

void SymmetricDirichletProblem::add_fix_coordinate_constraint(int _vertex_index, int _coordinate, double _fix_coordinate)
{
  fix_points.push_back(std::make_pair(2*_vertex_index + _coordinate, _fix_coordinate));
}

void SymmetricDirichletProblem::get_constraints(SMatrixD& _A, VectorD& _b)
{
  _A.resize(fix_points.size(), n_unknowns());
  _b.resize(fix_points.size());
  std::vector<Eigen::Triplet<double>> triplets;
  for (size_t i = 0; i < fix_points.size(); ++i)
  {
    _b[i] = fix_points[i].second;
    triplets.push_back(Eigen::Triplet<double>(i, fix_points[i].first, 1));
  }
  _A.setFromTriplets(triplets.begin(), triplets.end());
}

SymmetricDirichletProblem::ReferencePositionVector2D SymmetricDirichletProblem::get_equilateral_refernce_positions(double _area)
{
  ReferencePositionVector2D equilateral_reference;
  equilateral_reference << 0.0, 0.0,
                           1.0, 0.0,
                           0.5, 0.5*std::sqrt(3.0);
  equilateral_reference *= _area / 0.5*0.5*std::sqrt(3.0);
  return equilateral_reference;
}



double SymmetricDirichletOneVertexElement::eval_f(const VecV& _x, const VecC& _c)
{
  Matrix2x2d B;
  B(0,0) = _c[0]-_x[0];
  B(0,1) = _c[2]-_x[0];
  B(1,0) = _c[1]-_x[1];
  B(1,1) = _c[3]-_x[1];
  Matrix2x2d Bin = B.inverse();

  Matrix2x2d R;
  R(0,0) = _c[4+2]-_c[4+0];
  R(0,1) = _c[4+4]-_c[4+0];
  R(1,0) = _c[4+3]-_c[4+1];
  R(1,1) = _c[4+5]-_c[4+1];
  Matrix2x2d Rin = R.inverse();

  double area = 0.5 * R.determinant();
  if (B.determinant() * area <= 0)
  {
    double res = std::numeric_limits<double>::max();
    return res;
  }

  Matrix2x2d J   =  B * Rin;
  Matrix2x2d Jin =  R * Bin;

  double res = J.squaredNorm() + Jin.squaredNorm();

  return area * (res - 4);
}

void SymmetricDirichletOneVertexElement::eval_gradient(const VecV& _x, const VecC& _c, VecV& _g)
{
  const double a_x = _x[0];
  const double a_y = _x[1];
  const double b_x = _c[0];
  const double b_y = _c[1];
  const double c_x = _c[2];
  const double c_y = _c[3];
  const double d_x = _c[4];
  const double d_y = _c[5];
  const double e_x = _c[6];
  const double e_y = _c[7];
  const double f_x = _c[8];
  const double f_y = _c[9];

  const double det = d_x*e_y - d_x*f_y - d_y*e_x + d_y*f_x + e_x*f_y - e_y*f_x;

  _g[0] = (2*(-(d_y - f_y)*(b_x - c_x) + (e_y - f_y)*(a_x - c_x))*(e_y - f_y) + 2*(-(a_x - c_x)*(e_x - f_x) + (b_x - c_x)*(d_x - f_x))*(-e_x + f_x))/
          std::pow(d_x*e_y - d_x*f_y - d_y*e_x + d_y*f_x + e_x*f_y - e_y*f_x,2)
      -  2*(std::pow(-(a_y - c_y)*(e_x - f_x) + (b_y - c_y)*(d_x - f_x),2) +
            std::pow(-(b_x - c_x)*(d_x - f_x) + (a_x - c_x)*(e_x - f_x),2) +
            std::pow( (b_y - c_y)*(d_y - f_y) - (a_y - c_y)*(e_y - f_y),2) +
            std::pow(-(d_y - f_y)*(b_x - c_x) + (e_y - f_y)*(a_x - c_x),2)) *
            ( b_y - c_y)/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,3)
      + (2*(-(b_x - c_x)*(d_x - f_x) + (a_x - c_x)*(e_x - f_x))*( e_x - f_x) +
         2*(-(d_y - f_y)*(b_x - c_x) + (e_y - f_y)*(a_x - c_x))*( e_y - f_y))/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,2);

  _g[1] = (2*( (a_y - c_y)*(e_y - f_y) - (b_y - c_y)*(d_y - f_y))*(e_y - f_y) + 2*(-(a_y - c_y)*(e_x - f_x) + (b_y - c_y)*(d_x - f_x))*(-e_x + f_x))/
          std::pow(d_x*e_y - d_x*f_y - d_y*e_x + d_y*f_x + e_x*f_y - e_y*f_x,2)
      - 2*(std::pow(-(a_y - c_y)*(e_x - f_x) + (b_y - c_y)*(d_x - f_x),2) +
           std::pow(-(b_x - c_x)*(d_x - f_x) + (a_x - c_x)*(e_x - f_x),2) +
           std::pow( (b_y - c_y)*(d_y - f_y) - (a_y - c_y)*(e_y - f_y),2) +
           std::pow(-(d_y - f_y)*(b_x - c_x) + (e_y - f_y)*(a_x - c_x),2)) *
           (-b_x + c_x)/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,3)
      + (2*(-(a_y - c_y)*(e_x - f_x) + (b_y - c_y)*(d_x - f_x))*(-e_x + f_x) +
         2*( (b_y - c_y)*(d_y - f_y) - (a_y - c_y)*(e_y - f_y))*(-e_y + f_y))/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,2);

  // area weight
  _g[0] *= 0.5*det;
  _g[1] *= 0.5*det;
}

void SymmetricDirichletOneVertexElement::eval_hessian(const VecV& _x, const VecC& _c, std::vector<Triplet>& _triplets)
{
  const double a_x = _x[0];
  const double a_y = _x[1];
  const double b_x = _c[0];
  const double b_y = _c[1];
  const double c_x = _c[2];
  const double c_y = _c[3];
  const double d_x = _c[4];
  const double d_y = _c[5];
  const double e_x = _c[6];
  const double e_y = _c[7];
  const double f_x = _c[8];
  const double f_y = _c[9];

  const double det = d_x*e_y - d_x*f_y - d_y*e_x + d_y*f_x + e_x*f_y - e_y*f_x;

  Eigen::MatrixXd H(2,2);
  H(0,0) = (2*std::pow(e_y - f_y,2) + 2*std::pow(-e_x + f_x,2))/std::pow(d_x*e_y - d_x*f_y - d_y*e_x + d_y*f_x + e_x*f_y - e_y*f_x,2) + 6*(std::pow(-(a_y - c_y)*(e_x - f_x) + (b_y - c_y)*(d_x - f_x),2) + std::pow(-(b_x - c_x)*(d_x - f_x) + (a_x - c_x)*(e_x - f_x),2) + std::pow((b_y - c_y)*(d_y - f_y) - (a_y - c_y)*(e_y - f_y),2) + std::pow(-(d_y - f_y)*(b_x - c_x) + (e_y - f_y)*(a_x - c_x),2))*std::pow( b_y - c_y,2)/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,4) - 4*(2*(-(b_x - c_x)*(d_x - f_x) + (a_x - c_x)*(e_x - f_x))*( e_x - f_x) + 2*(-(d_y - f_y)*(b_x - c_x) + (e_y - f_y)*(a_x - c_x))*( e_y - f_y))*( b_y - c_y)/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,3) + (2*std::pow( e_x - f_x,2) + 2*std::pow( e_y - f_y,2))/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,2);
  H(1,1) = (2*std::pow(e_y - f_y,2) + 2*std::pow(-e_x + f_x,2))/std::pow(d_x*e_y - d_x*f_y - d_y*e_x + d_y*f_x + e_x*f_y - e_y*f_x,2) + 6*(std::pow(-(a_y - c_y)*(e_x - f_x) + (b_y - c_y)*(d_x - f_x),2) + std::pow(-(b_x - c_x)*(d_x - f_x) + (a_x - c_x)*(e_x - f_x),2) + std::pow((b_y - c_y)*(d_y - f_y) - (a_y - c_y)*(e_y - f_y),2) + std::pow(-(d_y - f_y)*(b_x - c_x) + (e_y - f_y)*(a_x - c_x),2))*std::pow(-b_x + c_x,2)/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,4) - 4*(2*(-(a_y - c_y)*(e_x - f_x) + (b_y - c_y)*(d_x - f_x))*(-e_x + f_x) + 2*( (b_y - c_y)*(d_y - f_y) - (a_y - c_y)*(e_y - f_y))*(-e_y + f_y))*(-b_x + c_x)/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,3) + (2*std::pow(-e_x + f_x,2) + 2*std::pow(-e_y + f_y,2))/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,2);
  H(1,0) = 6*(std::pow(-(a_y - c_y)*(e_x - f_x) + (b_y - c_y)*(d_x - f_x),2) + std::pow(-(b_x - c_x)*(d_x - f_x) + (a_x - c_x)*(e_x - f_x),2) + std::pow((b_y - c_y)*(d_y - f_y) - (a_y - c_y)*(e_y - f_y),2) + std::pow(-(d_y - f_y)*(b_x - c_x) + (e_y - f_y)*(a_x - c_x),2))*(b_y - c_y)*(-b_x + c_x)/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,4) - 2*(2*(-(a_y - c_y)*(e_x - f_x) + (b_y - c_y)*(d_x - f_x))*(-e_x + f_x) + 2*((b_y - c_y)*(d_y - f_y) - (a_y - c_y)*(e_y - f_y))*(-e_y + f_y))*(b_y - c_y)/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,3) - 2*(2*(-(b_x - c_x)*(d_x - f_x) + (a_x - c_x)*(e_x - f_x))*(e_x - f_x) + 2*(-(d_y - f_y)*(b_x - c_x) + (e_y - f_y)*(a_x - c_x))*(e_y - f_y))*(-b_x + c_x)/std::pow(a_x*b_y - a_x*c_y - a_y*b_x + a_y*c_x + b_x*c_y - b_y*c_x,3);
  H(0,1) = H(1,0);
  H *=  0.5 * det;

  Eigen::MatrixXd Hspd(2,2);
  project_hessian(H, Hspd, 1e-6);

  _triplets.reserve(2*2);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      _triplets.push_back(Triplet(i,j,Hspd(i,j)));

}

void SymmetricDirichletOneVertexElement::project_hessian(Eigen::MatrixXd& H_orig, Eigen::MatrixXd& H_spd, double eps)
{
  // Compute eigen-decomposition (of symmetric matrix)
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H_orig);
  Eigen::MatrixXd V = eig.eigenvectors();
  Eigen::MatrixXd D = eig.eigenvalues().asDiagonal();

  // Clamp all eigenvalues to eps
  for (int i = 0; i < H_orig.rows(); ++i)
      D(i, i) = std::max(eps, D(i, i));
  H_spd = V * D * V.inverse();
}

double SymmetricDirichletOneVertexElement::max_feasible_step(const VecV& _x, const VecV& _v, const VecC& _c)
{
  // get quadratic coefficients (ax^2 + b^x + c)
  auto U11 = _x[0];
  auto U12 = _x[1];
  auto U21 = _c[0];
  auto U22 = _c[1];
  auto U31 = _c[2];
  auto U32 = _c[3];

  auto V11 = _v[0];
  auto V12 = _v[1];
  const VecV::Scalar V21 = 0.0;
  const VecV::Scalar V22 = 0.0;
  const VecV::Scalar V31 = 0.0;
  const VecV::Scalar V32 = 0.0;

  double a = V11*V22 - V12*V21 - V11*V32 + V12*V31 + V21*V32 - V22*V31;
  double b = U11*V22 - U12*V21 - U21*V12 + U22*V11 - U11*V32 + U12*V31 + U31*V12 - U32*V11 + U21*V32 - U22*V31 - U31*V22 + U32*V21;
  double c = U11*U22 - U12*U21 - U11*U32 + U12*U31 + U21*U32 - U22*U31;

  double delta_in = pow(b,2) - 4*a*c;
  if (delta_in < 0) {
    return std::numeric_limits<double>::max();
  }
  double delta = sqrt(delta_in);
  double t1 = (-b + delta)/ (2*a);
  double t2 = (-b - delta)/ (2*a);

  double tmp_n = std::min(t1,t2);
  t1 = std::max(t1,t2); t2 = tmp_n;
  // return the smallest negative root if it exists, otherwise return infinity
  if (t1 > 0)
  {
    if (t2 > 0)
    {
      return 0.999 * t2;
    }
    else
    {
      return 0.999 * t1;
    }
  }
  else
  {
    return std::numeric_limits<double>::max();
  }
}

SymmetricDirichletOneRingProblem::SymmetricDirichletOneRingProblem()
  :
    FiniteElementProblem(2)
{
  FiniteElementProblem::add_set(&element_set);
}

void SymmetricDirichletOneRingProblem::add_triangle(const InputPositionVector2D& _current_positions, const SymmetricDirichletOneRingProblem::ReferencePositionVector2D& _reference_positions)
{

  SymmetricDirichletOneVertexElement::VecI indices;
  indices << 0,1;
  SymmetricDirichletOneVertexElement::VecC constants;
  constants << _current_positions  (1, 0),  _current_positions (1, 1),
               _current_positions  (2, 0),  _current_positions (2, 1),
               _reference_positions(0, 0), _reference_positions(0, 1),
               _reference_positions(1, 0), _reference_positions(1, 1),
               _reference_positions(2, 0), _reference_positions(2, 1);
  element_set.instances().add_element(indices, constants);
}

 SymmetricDirichletOneRingProblem::ReferencePositionVector2D SymmetricDirichletOneRingProblem::get_equilateral_refernce_positions(double _area)
{
  ReferencePositionVector2D equilateral_reference;
  equilateral_reference << 0.0, 0.0,
                           1.0, 0.0,
                           0.5, 0.5*std::sqrt(3.0);
  equilateral_reference *= _area/(0.5*0.5*std::sqrt(3.0));
  return equilateral_reference;
}

} // namespace COMISO

#endif //(COMISO_ADOLC_AVAILABLE && COMISO_EIGEN3_AVAILABLE)

