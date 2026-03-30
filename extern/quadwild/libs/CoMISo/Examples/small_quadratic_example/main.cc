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



#include <CoMISo/Utils/StopWatch.hh>
#include <gmm/gmm.h>
#include <vector>
#include <CoMISo/Solver/ConstrainedSolver.hh>
#include <CoMISo/Solver/MISolver.hh>
#include <CoMISo/Solver/GMM_Tools.hh>
#include <CoMISo/NSolver/NProblemInterface.hh>
#include <CoMISo/NSolver/GUROBISolver.hh>
#include <CoMISo/NSolver/OSQPSolver.hh>
#include <CoMISo/NSolver/LinearConstraint.hh>

/// function to initialize a simple system of linear equations
template<class MatrixT>
void init_les( MatrixT& _A, std::vector< double >& _b)
{
  _A(0,0) = 25  ; _A(0,1) = 0 ; _A(0,2) = -15; _A(0,3) = 0 ;
  _A(1,0) = 0   ; _A(1,1) = 20; _A(1,2) = -8 ; _A(1,3) = -4;
  _A(2,0) = -15 ; _A(2,1) = -8; _A(2,2) = 17 ; _A(2,3) = 0 ;
  _A(3,0) = 0   ; _A(3,1) = -4; _A(3,2) = 0  ; _A(3,3) = 4 ;

  _b[0] = 0; _b[1] = 4; _b[2] = -2; _b[3] = 0;
}

/// function to print the equations corresponding to the matrices of an equation system
template<class MatrixT>
void print_equations( const MatrixT& _B)
{
  size_t m = gmm::mat_nrows( _B);
  size_t n = gmm::mat_ncols( _B);
  for( size_t i = 0; i < m; ++i)
  {
    for( size_t j = 0; j < n-1; ++j)
    {
      if( _B(i,j) != 0.0)
        std::cout << _B(i,j) << "*x" << j;
      else
        std::cout << "   0 ";
      if( j < n-2 ) std::cout << " + ";
    }
    std::cout << " = " << _B(i, n-1) << std::endl;
  }
}


void quadratic_example_1()
{
  using COMISO_GMM::operator<<;

  std::cout << "---------- 1) Setup small (symmetric) test equation system Ax=b..." << std::endl;
  int n = 4;
  gmm::col_matrix< gmm::wsvector< double > > A(n,n);
  std::vector< double > x(n);
  std::vector< double > b(n);

  init_les( A, b);

  // create an empty constraint matrix (will be used later)
  gmm::row_matrix< gmm::wsvector< double > > constraints(0,n+1); //n+1 because of right hand side
  // create an empty vector of variable indices to be rounded (will be used later)
  std::vector< int > ids_to_round;

  std::cout << A << std::endl << b << std::endl;


  std::cout << "---------- 2) The original solution to this system is..." << std::endl;

  COMISO::ConstrainedSolver cs;
  //void solve( RMatrixT& _constraints, CMatrixT& _A, VectorT&  _x, VectorT&  _rhs, VectorIT& _idx_to_round, double    _reg_factor = 0.0, bool      _show_miso_settings = true, bool      _show_timings = true );
  //_show_miso_settings requires a QT context and hence must be false in this example
  cs.solve( constraints, A, x, b, ids_to_round, 0.0, false);
  // copy this solution for later
  std::vector< double > org_x( x);
  std::cout << x << std::endl;


  std::cout << "---------- 3) Rounding: forcing the second variable to lie on an integer, changes the solution to..." << std::endl;
  // reset system
  init_les( A, b);
  ids_to_round.push_back(1);
  cs.solve( constraints, A, x, b, ids_to_round, 0.0, false);
  std::cout << x << std::endl;


  std::cout << "---------- 4) Constraining: forcing the first variable to equal the second changes the solution to..." << std::endl;
  // reset system
  init_les( A, b);
  ids_to_round.clear();
  ids_to_round.push_back(1);
  // setup constraint x0*1+x1*0+x2*(-1)+x3*0=0
  gmm::resize( constraints, 1, n+1);
  constraints( 0, 0 ) = 1.0;
  constraints( 0, 1 ) = -1.0;
  std::cout << "           the constraint equation looks like this:" << std::endl;
  print_equations( constraints);
  cs.solve( constraints, A, x, b, ids_to_round, 0.0, false);
  std::cout << x << std::endl;
}


// minmize 0.5 x^T H x + q^T x
class SmallQuadraticProblem : public COMISO::NProblemInterface
{
public:

  SmallQuadraticProblem(SMatrixNP _H, Eigen::VectorXd _q)
    :
      H_(std::move(_H)),
      q_(std::move(_q))
  {
  }

  int n_unknowns() override { return q_.size(); }

  void   initial_x(double* _x ) override
  {
    for (int i = 0; i < n_unknowns(); ++i)
      _x[i] = 0.0;
  }

  double eval_f(const double* _x) override
  {
    auto v = vector_from_pointer(_x);
    return 0.5 * v.transpose() * H_ * v + q_.dot(v);
  }

  void eval_gradient(const double *_x, double *_g) override
  {
    auto v = vector_from_pointer(_x);
    Eigen::VectorXd grad = H_*v + q_;
    for (int i = 0; i < n_unknowns(); ++i)
      _g[i] = grad[i];
  }

  void eval_hessian(const double*, SMatrixNP& _H) override
  {
    _H = H_;
  }

  void store_result ( const double* _x ) override
  {
    res_.resize(n_unknowns());
    for (int i = 0; i < n_unknowns(); ++i)
      res_[i] = _x[i];
  }

  // advanced properties
  bool   constant_gradient ()                                    const override { return false; }
  bool   constant_hessian  ()                                    const override { return true; }

  const Eigen::VectorXd& get_result() { return res_; }

private:

  Eigen::VectorXd vector_from_pointer(const double* _x)
  {
    Eigen::VectorXd res;
    res.resize(n_unknowns());
    for (int i = 0; i < n_unknowns(); ++i)
      res[i] = _x[i];
    return res;
  }

  SMatrixNP H_;
  Eigen::VectorXd q_;
  Eigen::VectorXd res_;
};

// minimize quadratic energy with linear equality and inequality constraints constraints
void quadratic_example_2()
{
  COMISO::NProblemInterface::SMatrixNP H;
  Eigen::VectorXd q;

  // setup problem: 3x²+2xy+2y² + 6x + 4y
  // which is equal to
  //
  // 0.5 (x y) (6 2)  (x)  + (6 4) (x)
  //           (2 4)  (y)          (y)
  std::vector<Eigen::Triplet<double>> trips;
  trips.emplace_back(0,0,6);
  trips.emplace_back(0,1,2);
  trips.emplace_back(1,0,2);
  trips.emplace_back(1,1,4);
  H.resize(20,20);
  H.setFromTriplets(trips.begin(), trips.end());

  q.resize(20);
  q.Zero(20);
  q[0] = 6;
  q[1] = 4;

  SmallQuadraticProblem prob(H, q);

  int n_tests = 100;
  double t_gurobi = 0.0;
  double t_nasoq = 0.0;
  double t_osqp = 0.0;

  for (int i = 0; i < n_tests; ++i)
  {

    for (auto use_constraints : {false/*, true*/})
    {

      std::vector<double> min = use_constraints ? std::vector<double>{3, -2.5} : std::vector<double>{-0.8, -0.6};
      std::cout << "Min shoudl be at x = " << min[0] << " and y = " << min[1] << " with value " << (use_constraints ? 32.5 : -3.6) << ". Actual value is " << prob.eval_f(min.data()) << std::endl;

      std::vector<COMISO::LinearConstraint> linear_constraints;
      if (use_constraints)
      {
        // Setup constraints x > 3 which translates to x - 3 > 0
        COMISO::LinearConstraint::SVectorNC coeffs(prob.n_unknowns());
        coeffs.coeffRef(0) = 1;
        COMISO::LinearConstraint lc(coeffs, -3, COMISO::LinearConstraint::NC_GREATER_EQUAL);
        linear_constraints.push_back(lc);
      }
      std::vector<COMISO::NConstraintInterface*> constraints;
      for (auto& lc : linear_constraints)
        constraints.push_back(&lc);

#if COMISO_GUROBI_AVAILABLE
      {
        COMISO::GUROBISolver solver;
        std::vector<COMISO::PairIndexVtype> var_types;
        var_types.emplace_back(0, COMISO::Real);
        var_types.emplace_back(1, COMISO::Real);
        Base::StopWatch sw;
        sw.start();
        bool success = solver.solve(&prob,constraints,var_types);
        sw.stop();
        if (i > 1 || n_tests == 1)
          t_gurobi += sw.elapsed();
        if (success)
        {
          std::cout << "Gurobi succeeded in " << sw.elapsed() << "ms. Optimum found at x = " << prob.get_result()[0] << " and y = " << prob.get_result()[1] << " with value " << prob.eval_f(prob.get_result().data()) << std::endl;
        }
        else
        {
          std::cout << "Gurobi failed" << std::endl;
        }
      }
#endif

#if COMISO_OSQP_AVAILABLE
      //OSQO
      {
        COMISO::OSQPSolver solver;
        Base::StopWatch sw;
        sw.start();
        bool success = solver.solve(&prob, constraints);
        sw.stop();
        if (i > 1 || n_tests == 1)
          t_osqp += sw.elapsed();
        if (success)
        {
          std::cout << "OSQP  succeeded in " << sw.elapsed() << "ms. Optimum found at x = " << prob.get_result()[0] << " and y = " << prob.get_result()[1] << " with value " << prob.eval_f(prob.get_result().data()) << std::endl;
        }
        else
        {
          std::cout << "OSQP failed" << std::endl;
        }
      }
#endif

    }

  }

  std::cout << "avg time gurobi: " << t_gurobi / (2*std::max(n_tests-1, 1)) << "ms." << std::endl;
  std::cout << "avg time osqp  : " << t_osqp   / (2*std::max(n_tests-1, 1)) << "ms." << std::endl;

}

// Example main
int main(void)
{

//  quadratic_example_1();

  quadratic_example_2();



  return -1;
}

