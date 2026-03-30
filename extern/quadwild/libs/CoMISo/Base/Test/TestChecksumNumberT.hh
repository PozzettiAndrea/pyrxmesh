// (C) Copyright 2022 by Autodesk, Inc.

#ifndef BASE_TESTCHECKSUMNUMBERT_HH_INCLUDED
#define BASE_TESTCHECKSUMNUMBERT_HH_INCLUDED

#include <Base/Test/TestChecksum.hh>

#ifdef TEST_ON

#include <cmath>
#include <sstream>
#include <type_traits>

namespace Test
{
namespace Checksum
{
//! Support functionality for NumberT<> test checksums 
namespace Number
{

//! Standard methods to compute the change between two different checksum numbers
namespace Change
{

//! The change of the checksum number is absolute
struct Absolute
{
  template <typename ValueT>
  ValueT operator()(const ValueT& _old, const ValueT& _new) const
  {
    return _new - _old;
  }
};

//! Change of the number checksum is relative w.r.t. the old value
struct Relative
{
  template <typename ValueT>
  ValueT operator()(const ValueT& _old, const ValueT& _new) const
  {
    const auto chng = _new - _old;
    return _old == 0 ? chng : chng / _old; 
  }
};

} // namespace Change

//! Methods for deciding if a number checksum change is negligible
namespace Negligible
{
//! No change is considered negligible, do not use for floating point numbers.
struct None
{
  template <class ValueT> bool operator()(const ValueT&) { return false; }
};

//! Consider a value change <= 10^EXPONENT as neglegible
template <int EXPONENT> struct Exponent10ToleranceT
{
  template <typename ValueT>
  bool operator()(const ValueT& _diff) const
  {
    static_assert(EXPONENT <= std::numeric_limits<ValueT>::max_exponent10,
        "Exponent too large");
    static_assert(EXPONENT >= std::numeric_limits<ValueT>::min_exponent10,
        "Exponent too small");
    static const auto TOLERANCE = std::pow(10, EXPONENT);
    return std::abs(_diff) <= TOLERANCE;
  }
};

} // namespace Negligible

//! Qualify the change of the checksum value as a Difference::Type value
namespace Qualify
{

//! Default approach, qualify as unknown
struct None
{
  template <typename ValueT>
  Difference::Type operator()(const ValueT&, const ValueT&) const
  {
    return Difference::UNKNOWN;
  }
};

//! Qualify the new value as an improvement if it is smaller than the old one
struct SmallerIsBetter
{
  template <typename ValueT>
  Difference::Type operator()(const ValueT& _old, const ValueT& _new) const
  {
    return _old > _new ? Difference::IMPROVED : Difference::REGRESSED;
  }
};

// Qualify the new value as an improvement if it is larger than the old one
struct LargerIsBetter
{
  template <typename ValueT>
  Difference::Type operator()(const ValueT& _old, const ValueT& _new) const
  {
    return _old < _new ? Difference::IMPROVED : Difference::REGRESSED;
  }
};

// Qualify the new value as an improvement if it is closer to 0
struct Target0
{
  template <typename ValueT>
  Difference::Type operator()(const ValueT& _old, const ValueT& _new) const
  {
    return std::abs(_old) > std::abs(_new) ? Difference::IMPROVED
                                           : Difference::REGRESSED;
  }
};

} // namespace Qualify

/*!
Compare the new vs the old number checksum value. Various aspects of the
comparison can be parameterized through the template arguments.
*/
template <class ValueT,
    class ChangeT = Change::Absolute,     //!< compute the values change
    class NegligibleT = Negligible::None, //!< is the change negligible
    class QualifyT = Qualify::None //!< qualify the change as a Difference::Type
    >
struct CompareT
{
  // If we have a floating point Value type, the Negligible type trait should be
  // set to something different than the Neglegible::None default.
  static_assert(!std::is_floating_point<ValueT>::value ||
                    !std::is_same<NegligibleT, Negligible::None>::value,
      "For floating point checksums, please specify a NeglegibleT trait that "
      "allows small changes to qualify as negligible differences, e.g., use "
      "Qualify::Exponent10ToleranceT");

  ValueT change(const ValueT& _old, const ValueT& _new) const
  {
    return ChangeT()(_old, _new);
  }

  bool neglegible(const ValueT& _diff) const
  {
    return NegligibleT()(_diff);
  }

  Difference::Type qualify(const ValueT& _old, const ValueT& _new) const
  {
    return QualifyT()(_old, _new);
  }

  Difference operator()(const ValueT& _old, const ValueT& _new) const
  {
    if (_old == _new) // bitwise equal?
      return Difference::EQUAL;

    const auto diff = change(_old, _new);
    Base::OStringStream strm;
    strm << diff;
    return Difference(
        neglegible(diff) ? Difference::NEGLIGIBLE : qualify(_old, _new),
        strm.str);
  }
};

} // namespace Number

/*!
Generic checksum class to record and compare a value of a certain type.
*/
template <typename ValueT,
    class ChangeT = Number::Change::Absolute,     //!< compute the values change
    class NegligibleT = Number::Negligible::None, //!< is the change negligible
    class QualifyT =
        Number::Qualify::None //!< qualify the change as a Difference::Type
    >
class NumberT : public Object
{
public:
  NumberT(const char* const _name, const Level _lvl = L_ALL)
      : Object(_name, _lvl)
  {
  }

protected:
  virtual Difference compare_data(const String& _old, const String& _new) const
  {
    std::istringstream strm_old(_old), strm_new(_new);
    ValueT val_old, val_new;
    strm_old >> val_old;
    strm_new >> val_new;
    return cmpr_(val_old, val_new);
  }

private:
  Number::CompareT<ValueT, ChangeT, NegligibleT, QualifyT> cmpr_;
};

} // namespace Checksum
} // namespace Test

#endif // TEST_ON
#endif // BASE_TESTCHECKSUMNUMBERT_HH_INCLUDED
