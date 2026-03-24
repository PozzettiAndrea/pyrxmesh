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
#include <iostream>

#include <CoMISo/NSolver/NewtonSolver.hh>
#include <CoMISo/NSolver/NProblemInterface.hh>
#include <vector>

#include <CoMISo/Utils/ExactConstraintSatisfaction.hh>

#include <Base/Debug/DebConfig.hh>

//------------------------------------------------------------------------------------------------------

class SmallNProblem : public COMISO::NProblemInterface
{
public:
  // specify a function which has several local minima
  // f(x,y) = x^4 + y^4

  // number of unknown variables, here x and y = 2
  virtual int    n_unknowns   (                                )
  {
    return 2;
  }

  // initial value where the optimization should start from
  virtual void   initial_x    (       double* _x               )
  {
    _x[0] = 4.0;
    _x[1] = 2.0;
  }

  // function evaluation at location _x
  virtual double eval_f       ( const double* _x               )
  {
    return std::pow(_x[0], 4) + std::pow(_x[1], 4);
  }

  // gradient evaluation at location _x
  virtual void   eval_gradient( const double* _x, double*    _g)
  {
    _g[0] =  4.0*std::pow(_x[0], 3);
    _g[1] =  4.0*std::pow(_x[1], 3);
   }

  // hessian matrix evaluation at location _x
  virtual void   eval_hessian ( const double* _x, SMatrixNP& _H)
  {
    _H.resize(n_unknowns(), n_unknowns());
    _H.setZero();

    _H.coeffRef(0,0) =  12.0*std::pow(_x[0], 2);
    _H.coeffRef(1,0) =  0.0;
    _H.coeffRef(0,1) =  0.0;
    _H.coeffRef(1,1) =  12.0*std::pow(_x[1], 2);
  }

  // print result
  virtual void   store_result ( const double* _x               )
  {
    solution.resize(n_unknowns());
    for (int i = 0; i < n_unknowns(); ++i)
      solution[i] = _x[i];
  }

  // advanced properties
  virtual bool   constant_hessian() const { return false; }

  std::vector<double> solution;
};

// Example main
int main(void)
{
  std::cout << "---------- 1) Get an instance of a problem..." << std::endl;
  SmallNProblem problem;

  std::cout << "---------- 2) Set up constraints..." << std::endl;


  int n_constraints = 2; // there will be two constraints
  Eigen::VectorXi b;
  COMISO::ExactConstraintSatisfaction::SP_Matrix_R A(n_constraints, problem.n_unknowns());
  b.setZero(n_constraints);

  // first constraint: first variable equals three times second
  //different number of constraints :
  if(n_constraints == 2)
  {
    A.coeffRef(0,0) =  2;
    A.coeffRef(0,1) =  -1;
    b.coeffRef(0)   =  0;
    A.coeffRef(1,0) =  -2;
    A.coeffRef(1,1) =  8;
    b.coeffRef(1)   =  21;
  }

  std::cout << "Constraints: Ax = b with A = \n" << A << "and b = \n" << b << std::endl;

  std::cout << "---------- 3) Solve with Newton Solver..." << std::endl;
  COMISO::NewtonSolver nsolver;
  Eigen::SparseMatrix<double> Ad = A.cast<double>();
  Eigen::VectorXd bd = b.cast<double>();
  {
    DEB_only(Debug::ScopedOutputLevel output_lvl(0)); // disable output for solve method
    nsolver.solve(&problem, Ad, bd);
  }

  std::cout << "---------- 4) Print solution..." << std::endl;
  std::cout << std::setprecision(100);
  for (unsigned int i = 0; i < problem.n_unknowns(); ++i)
    std::cout << "x[" << i << "] = " << problem.solution[i] << std::endl;


  std::cout << "---------- 5) Check constraint violation..." << std::endl;
  Eigen::VectorXd x;
  x.resize(problem.n_unknowns());
  for (unsigned int i = 0; i < problem.n_unknowns(); ++i)
    x.coeffRef(i) = problem.solution[i];

  std::cout << "Constraint violation: " << (A.cast<double>() *x - b.cast<double>()).squaredNorm() << std::endl;

  std::cout << "---------- 6) Try to exactly fulfill constraints..." << std::endl;

  COMISO::ExactConstraintSatisfaction satisfy;
//  satisfy.print_matrix(A);
  satisfy.evaluation(A, b, x);


  for (unsigned int i = 0; i < problem.n_unknowns(); ++i)
    std::cout << "x[" << i << "] = " << x[i] << std::endl;

  std::cout << "---------- 7) Check constraint violation again..." << std::endl;

  std::cout << "Constraint violation: " << (A.cast<double>() *x - b.cast<double>()).squaredNorm() << std::endl;


  return 0;
}
