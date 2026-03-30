// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_OPTIONT_HH_INCLUDED
#define BASE_OPTIONT_HH_INCLUDED

#include <Base/Code/Quality.hh>

namespace Base
{
/*!
An option that can be controlled by a Test::Arg but also persists independently.

Use by defining a global _static_ variable as follows: 

BASE_OPTION(bool, my_flag, true);
BASE_OPTION(int, my_number, 42);
BASE_OPTION(std::string, my_str, "hello world")

Currently instantiated for bool, int, size_t, float, double and std::string.
*/
template <typename T> class OptionT
{
public:
  T value;

  OptionT(const char* const _name, const T& _dflt_val) : value(_dflt_val)
  {
#ifdef TEST_ON
    bind(_name);
#else
    LOW_CODE_QUALITY_VARIABLE_ALLOW(_name);
#endif//TEST_ON
  }

  // conversion operators
  operator T&() { return value; }
  operator const T&() const { return value; }

  OptionT<T>& operator=(const T& _val)
  {
    value = _val;
    return *this;
  }

#ifdef TEST_ON // define a binding mechanism for the option value to a test arg
protected: 
  void bind(const char* const _name); //!< Bind the option to a test argument 
#endif//TEST_ON
};

}// namespace Base

#define BASE_OPTION(TYPE, NAME, DFLT_VAL) \
  Base::OptionT<TYPE> NAME(#NAME, DFLT_VAL)

#endif//BASE_OPTIONT_HH_INCLUDED
