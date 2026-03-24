// Copyright 2021 Autodesk, Inc. All rights reserved.

//=============================================================================
//
//  CLASS SolverBaseT
//
//=============================================================================

#ifndef COMISO_SOLVERBASET_HH
#define COMISO_SOLVERBASET_HH

//== COMPILE-TIME PACKAGE REQUIREMENTS ========================================
#include <CoMISo/Config/config.hh>
#if (COMISO_EIGEN3_AVAILABLE)

//== INCLUDES =================================================================

#include <CoMISo/Config/CoMISoDefines.hh>
#include <CoMISo/Utils/VectorT.hh>

#include <vector>
#include <cstddef>

//== NAMESPACES ===============================================================

namespace COMISO
{

template <size_t DIM> struct SolverBaseT
{
  using Point = VectorT<double, DIM>; // Object to minimize

  using PointVector = std::vector<Point>;

  struct Value // It means that X_name_ = val_.
               // Used to get the results and to set the fixed variables
  {
    Value() {}
    Value(size_t _var_name, Point _point) : var_name(_var_name), point(_point)
    {
    }
    size_t var_name; // Variable name.
    Point point;     // Value of the variable

    bool operator<(const Value& _vl) const
    {
      if (var_name != _vl.var_name)
        return var_name < _vl.var_name;
      return point < _vl.point;
    }

    bool operator==(const Value& _vl) const
    {
      return var_name == _vl.var_name && point == _vl.point;
    }

  }; // struct Value

  using ValueVector = std::vector<Value>;

  struct LinearTerm // It means that coeff_ * X_name_
  {
    size_t var_name; // Variable name.
    double coeff;
    bool operator<(const LinearTerm& _lt) const
    {
      return var_name < _lt.var_name;
    }
  }; // struct LinearTerm

  using LinearTermVector = std::vector<LinearTerm>;

  struct LinearEquation // a linear equation in the form Sum(c_i * x_i) =
                        // constant_term
  {
    LinearTermVector linear_terms;
    // Construction must create an equation 'nothing' = 0, so const_term is
    // initialized with zeros.
    Point const_term{};

    // The constraint is in the form "noting" = "something != 0"
    bool infeasible(double _tol) const
    {
      if (!linear_terms.empty())
        return false;
      double sq_len = 0;
      for (auto term : const_term)
        sq_len += term * term;
      return sq_len > _tol * _tol;
    }
  }; // struct LinearEquation

}; // struct Types

//=============================================================================
} // namespace COMISO
//=============================================================================

//=============================================================================
#endif // COMISO_EIGEN3_AVAILABLE
//=============================================================================
#endif // COMISO_SOLVERBASET_HH defined
//=============================================================================
