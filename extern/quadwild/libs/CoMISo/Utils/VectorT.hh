// Copyright 2022 Autodesk, Inc. All rights reserved.

#ifndef VECTORT_HH
#define VECTORT_HH

#include <array>

namespace COMISO
{

// A simple wrapper around std::array that provides componentwise arithmetic
// operations.
template <typename ScalarT, std::size_t DIM>
struct VectorT : public std::array<ScalarT, DIM>
{
  using Base = std::array<ScalarT, DIM>;
  using Self = VectorT<ScalarT, DIM>;

  ScalarT& operator[](std::size_t _idx) { return Base::operator[](_idx); }
  const ScalarT& operator[](std::size_t _idx) const
  {
    return Base::operator[](_idx);
  }

  Self& operator+=(const Self& _other)
  {
    for (std::size_t i = 0; i < DIM; ++i)
      (*this)[i] += _other[i];
    return *this;
  }

  Self& operator-=(const Self& _other)
  {
    for (std::size_t i = 0; i < DIM; ++i)
      (*this)[i] -= _other[i];
    return *this;
  }

  Self& operator*=(ScalarT _other)
  {
    for (std::size_t i = 0; i < DIM; ++i)
      (*this)[i] *= _other;
    return *this;
  }

  Self& operator/=(ScalarT _other)
  {
    for (std::size_t i = 0; i < DIM; ++i)
      (*this)[i] /= _other;
    return *this;
  }
};

template <typename ScalarT, std::size_t DIM>
VectorT<ScalarT, DIM> operator+(
    const VectorT<ScalarT, DIM>& _left, const VectorT<ScalarT, DIM>& _right)
{
  auto res = _left;
  res += _right;
  return res;
}

template <typename ScalarT, std::size_t DIM>
VectorT<ScalarT, DIM> operator-(
    const VectorT<ScalarT, DIM>& _left, const VectorT<ScalarT, DIM>& _right)
{
  auto res = _left;
  res -= _right;
  return res;
}

template <typename ScalarT, std::size_t DIM>
VectorT<ScalarT, DIM> operator*(
    ScalarT _left, const VectorT<ScalarT, DIM>& _right)
{
  auto res = _right;
  res *= _left;
  return res;
}

template <typename ScalarT, std::size_t DIM>
VectorT<ScalarT, DIM> operator*(
    const VectorT<ScalarT, DIM> _left, ScalarT _right)
{
  auto res = _left;
  res *= _right;
  return res;
}

template <typename ScalarT, std::size_t DIM>
VectorT<ScalarT, DIM> operator-(const VectorT<ScalarT, DIM>& _vec)
{
  VectorT<ScalarT, DIM> res;
  for (auto i = 0; i < DIM; ++i)
    res[i] = -_vec[i];
  return res;
}

} // namespace COMISO

//=============================================================================
#endif // VECTORT_HH defined
//=============================================================================
