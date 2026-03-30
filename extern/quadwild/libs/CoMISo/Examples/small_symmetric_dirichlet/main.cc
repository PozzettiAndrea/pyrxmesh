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

#if (COMISO_ADOLC_AVAILABLE && COMISO_EIGEN3_AVAILABLE)

#include <CoMISo/Utils/StopWatch.hh>
#include <CoMISo/NSolver/NewtonSolver.hh>
#include <CoMISo/NSolver/SymmetricDirichletProblem.hh>
#include <vector>

//------------------------------------------------------------------------------------------------------


void symmetric_dirichlet_problem_example()
{
  std::cout << "---------- Example of symmetric dirichlet problem:" << std::endl;
  std::cout << "---------- 1) Get an instance of a SymmetricDirichletProblem..." << std::endl;

  // then create finite element problem and add sets
  unsigned int n_vertices = 4;
  COMISO::SymmetricDirichletProblem sd_problem(n_vertices);
  COMISO::SymmetricDirichletProblem::IndexVector indices(0,1,2);
  COMISO::SymmetricDirichletProblem::ReferencePositionVector2D positions;
  positions << 0, 0, // first point
               1, 0, // second point
               0, 1; // third point
  sd_problem.add_triangle(indices, positions);
  COMISO::SymmetricDirichletProblem::IndexVector indices2(3,2,1);
  sd_problem.add_triangle(indices2, positions); // same reference positions can be used because we want both triangles to be isosceles

  std::vector<double> initial_solution{0.0,0.0,
                                       2.0,0.0,
                                       2.0,2.0,
                                       3.0,4.0};
  sd_problem.x() = initial_solution;

  std::cout << "---------- 2) Set up constraints..." << std::endl;
  // fix first vertex to origin to fix translational degree of freedom
  sd_problem.add_fix_point_constraint(0, 0.0, 0.0);
  // fix v coordinate of second vertex to 0 to fix rotational degree of freedom
  sd_problem.add_fix_coordinate_constraint(1, 1, 0.0);

  std::cout << "---------- 3) Solve with Newton Solver..." << std::endl;
  COMISO::SymmetricDirichletProblem::VectorD b;
  COMISO::SymmetricDirichletProblem::SMatrixD A;
  sd_problem.get_constraints(A, b);
  COMISO::NewtonSolver nsolver;
  nsolver.solve(&sd_problem, A, b);

  // print result
  for (unsigned int i = 0; i < n_vertices; ++i)
    std::cout << "p" << i << " = ( " << sd_problem.x()[2*i+0] << ", " << sd_problem.x()[2*i+1] << ")"  << std::endl;
}


void symmetric_dirichlet_problem_example_one_ring_with_constraints(bool verbose = true)
{
  if (verbose)
  {
    std::cout << "---------- Example of symmetric dirichlet problem of a one ring with fixed boundary:" << std::endl;
    std::cout << "---------- 1) Get an instance of a SymmetricDirichletProblem..." << std::endl;
  }

  // then create finite element problem and add sets
  unsigned int n_vertices = 7;
  COMISO::SymmetricDirichletProblem sd_problem(n_vertices);
  auto reference_positions = sd_problem.get_equilateral_refernce_positions();
  for (int i = 0; i < 6; ++i)
  {
    COMISO::SymmetricDirichletProblem::IndexVector indices(0,i+1,(i+2-1)%6+1);
    sd_problem.add_triangle(indices, reference_positions);
  }

  std::vector<double> initial_solution;
  // center vertex position
  initial_solution.push_back(-0.5);
  initial_solution.push_back( 0.3);

  Eigen::Rotation2D<double> rot(60.0/180.0*M_PI);
  Eigen::Vector2d pos{1.0,0.0};
  for (int i = 0; i < 6; ++i)
  {
    initial_solution.push_back(pos(0));
    initial_solution.push_back(pos(1));
    pos = rot * pos;
  }

  sd_problem.x() = initial_solution;

  if (verbose)
    std::cout << "---------- 2) Set up constraints..." << std::endl;
  // fix all boundary positions
  for (int i = 1; i < 7; ++i)
    sd_problem.add_fix_point_constraint(i, initial_solution[2*i], initial_solution[2*i+1]);

  if (verbose)
    std::cout << "---------- 3) Solve with Newton Solver..." << std::endl;
  COMISO::SymmetricDirichletProblem::VectorD b;
  COMISO::SymmetricDirichletProblem::SMatrixD A;
  sd_problem.get_constraints(A, b);
  COMISO::NewtonSolver nsolver;
  nsolver.solve(&sd_problem, A, b);

  // print result
  if (verbose)
    for (unsigned int i = 0; i < n_vertices; ++i)
      std::cout << "p" << i << " = ( " << sd_problem.x()[2*i+0] << ", " << sd_problem.x()[2*i+1] << ")"  << std::endl;
}


void symmetric_dirichlet_problem_example_one_ring_with_specialized_problem(bool verbose = true)
{
  if (verbose)
  {
    std::cout << "---------- Example of specialized symmetric dirichlet problem for one ring optimization:" << std::endl;
    std::cout << "---------- 1) Get an instance of a SymmetricDirichletProblem..." << std::endl;
  }

  // then create finite element problem and add sets
  unsigned int n_vertices = 7;
  COMISO::SymmetricDirichletOneRingProblem sd_problem;
  auto reference_positions = sd_problem.get_equilateral_refernce_positions();

  std::vector<double> initial_solution;
  // center vertex position
  initial_solution.push_back(-0.5);
  initial_solution.push_back( 0.3);

  Eigen::Rotation2D<double> rot(60.0/180.0*M_PI);
  Eigen::Vector2d pos{1.0,0.0};
  for (int i = 0; i < 6; ++i)
  {
    initial_solution.push_back(pos(0));
    initial_solution.push_back(pos(1));
    pos = rot * pos;
  }

  for (int i = 1; i < 7; ++i)
  {
    COMISO::SymmetricDirichletOneRingProblem::InputPositionVector2D current_positions;
    int id0 = 0;
    int id1 = i;
    int id2 = i%6+1;
    current_positions << initial_solution[2*id0], initial_solution[2*id0+1],
                         initial_solution[2*id1], initial_solution[2*id1+1],
                         initial_solution[2*id2], initial_solution[2*id2+1];
    sd_problem.add_triangle(current_positions, reference_positions);
  }

  // only first positions required because we only optimize center vertex
  initial_solution.resize(2);
  sd_problem.x() = initial_solution;

  if (verbose)
    std::cout << "---------- 3) Solve with Newton Solver..." << std::endl;
  COMISO::NewtonSolver nsolver;
  nsolver.solve(&sd_problem);

  // print result

  if (verbose)
    std::cout << "p" << 0 << " = ( " << sd_problem.x()[2*0+0] << ", " << sd_problem.x()[2*0+1] << ")"  << std::endl;
}

// Example main
int main(void)
{
  symmetric_dirichlet_problem_example();
  symmetric_dirichlet_problem_example_one_ring_with_constraints();
  symmetric_dirichlet_problem_example_one_ring_with_specialized_problem();

  COMISO::StopWatch sw_constraints;
  sw_constraints.start();
  for (int i = 0; i < 1000; ++i)
    symmetric_dirichlet_problem_example_one_ring_with_constraints(false);
  double time_constraints = sw_constraints.stop();
  std::cout << "Solve with constraints took " << time_constraints/1000.0 << " seconds" << std::endl;

  COMISO::StopWatch sw_specialized_problem;
  sw_specialized_problem.start();
  for (int i = 0; i < 1000; ++i)
    symmetric_dirichlet_problem_example_one_ring_with_specialized_problem(false);
  double time_specialized_problem = sw_specialized_problem.stop();
  std::cout << "Solve with specialized problem took " << time_specialized_problem/1000.0 << " seconds" << std::endl;

  return 0;
}

#else // (COMISO_ADOLC_AVAILABLE && COMISO_EIGEN3_AVAILABLE)

int main(void)
{
  std::cerr << "Warning: Example cannot be executed since either EIGEN3 or ADOLC is not available..." << std::endl;
  return 0;
}


#endif // (COMISO_ADOLC_AVAILABLE && COMISO_EIGEN3_AVAILABLE)
