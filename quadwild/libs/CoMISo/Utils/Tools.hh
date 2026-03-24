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

#ifndef TOOLS_HH
#define TOOLS_HH

#include <math.h>
#include <stdint.h>
#include <vector>
#include <algorithm>

//! get the closest integer to _x, much faster than round(), can overflow!
inline int int_round(const double _x)
{
  return int(_x < 0 ? _x - 0.5 : _x + 0.5);
}

//! a popular macro to access int_round()
#define ROUND_MI(x) int_round(x)

//! get the closest 64-it integer to _x, much faster than round(), no overflow!
inline int64_t int64_round(const double _x)
{
  return int64_t(_x < 0 ? _x - 0.5 : _x + 0.5);
}

/*!
Same as above, but cast to a double, much faster than round()!
Pass through an int64 cast to avoid the possibility of a 32-bit overflow.

The Release assembly of double_round() with MSVC:

00007FFDA3D7F60C  movaps      xmm0,xmm1
00007FFDA3D7F60F  comisd      xmm8,xmm1
00007FFDA3D7F614  jbe         07FFDA3D7F61Ch
00007FFDA3D7F616  subsd       xmm0,xmm7
00007FFDA3D7F61A  jmp         07FFDA3D7F620h
00007FFDA3D7F61C  addsd       xmm0,xmm7
00007FFDA3D7F620  cvttsd2si   rax,xmm0   // eax if int_round() is used
00007FFDA3D7F625  xorps       xmm6,xmm6
00007FFDA3D7F628  cvtsi2sd    xmm6,rax   // eax if int_round() is used

The only difference between int_round and int64_round() is eax vs rax.

Measurements of double_round() in MISolver::solve_multiple_rounding() for
large CF data on a x64 system:
int64_round time: 78.85s
int_round time:   79.52s
*/
inline double double_round(const double _x) { return double(int64_round(_x)); }

//! get the residual after rounding
inline double round_residue(const double _x)
{
  return fabs(double_round(_x) - _x);
}

//! get if _x is rounded within _tol
inline bool is_rounded(const double _x, const double _tol)
{
  return round_residue(_x) <= _tol;
}

//! compare two double values within _tol
inline bool are_same(const double _x, const double _y, const double _tol)
{
  return fabs(_x - _y) <= _tol;
}

//! get _a * _a
template <typename T> inline T sqr(const T& _a) { return _a * _a; }



// Sort list and remove duplicate elements, using provided predicates.
template <typename T, typename LessPredicateT, typename EqualPredicateT>
void sort_unique(std::vector<T>& _v, const LessPredicateT& _less,
    const EqualPredicateT& _equal)
{
  std::sort(_v.begin(), _v.end(), _less);
  _v.erase(std::unique(_v.begin(), _v.end(), _equal), _v.end());
}

// Sort list and remove duplicate elements. Using provided predicate to sort
// elements and to derive an equality predicate.
template <typename T, typename LessPredicateT>
void sort_unique(std::vector<T>& _v, const LessPredicateT& _less)
{
  using value_type = typename std::vector<T>::value_type;
  const auto equal = [&_less](const value_type& _l, const value_type& _r)
  {
    return !_less(_l, _r) && !_less(_r, _l);
  };
  sort_unique(_v, _less, equal);
}

// Sort list and remove duplicate elements. Sorted via operator<(), equality
// of elements checked via operator==().
template <typename T>
void sort_unique(std::vector<T>& _v)
{
  std::sort(_v.begin(), _v.end());
  _v.erase(std::unique(_v.begin(), _v.end()), _v.end());
}

template <typename T>
std::vector<T> make_sorted_unique(const std::vector<T>& _v)
{
  auto v = _v;
  sort_unique(v);
  return v;
}


//=============================================================================
#endif // TOOLS_HH defined
//=============================================================================
