// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_TESTOUTCOME_HH_INCLUDED
#define BASE_TESTOUTCOME_HH_INCLUDED

#ifdef TEST_ON

#include <cstddef>

namespace Test {

/*!
Representation of a result in the test system, e.g., for a checksum, test run,
test list report, comparison.

This might hold more information in the future, such as number of errors and
warnings reported, etc. Hence it is more future-proof than an enum.
*/
class Result
{
public:
  enum Type // keep descriptions updated.
  {
    OK,
    WARNING,
    ERROR,
    FAILURE,
    CRASH,
    HANG,
    TYPE_NUMBER
  };

public:
  Result(Type _type = OK) : type_(_type) {}

  // Read in a short Result description
  template <class StreamT>
  friend inline StreamT& operator>>(StreamT& _is, Result& _rslt)
  {
    char c;
    _is >> c;
    auto descr = short_descriptions();
    for (size_t i = 0; i < Result::Type::TYPE_NUMBER; ++i)
    {
      if (descr[i] == c)
      {
        _rslt = Result(Type(i));
        return _is;
      }
    }
    _rslt = Result(OK);
    return _is;
  }

  template <class StreamT>
  friend inline StreamT& operator<<(StreamT& _os, const Result& _rslt)
  {
    _os << short_descriptions()[_rslt.type()];
    return _os;
  }

  // Return a long Result description
  const char* message() const { return long_descriptions()[type_]; }

  Type type() const { return type_; }

  bool ok() const { return type() == OK; }

  //! Update the type to the more severe type (between this and the input)
  Result& operator+=(const Type _type)
  {
    if (type_ < _type)
      type_ = _type;
    return *this;
  }

  Result& operator+=(const Result& _rslt) { return *this += _rslt.type_; }

  friend inline bool operator<(const Result& _a, const Result& _b)
  {
    return _a.type() < _b.type();
  }

  friend inline bool operator!=(const Result& _a, const Result& _b)
  {
    return _a.type() != _b.type();
  }

private:
  Type type_;

private:
  static const char** long_descriptions()
  {
    static const char* long_descr[] = {
        "OK", "WARNING", "ERROR", "FAILURE", "CRASH", "HANG"
    };
    return long_descr;
  }

  static const char* short_descriptions()
  {
    static const char short_descr[] = { ' ', 'W', 'E', 'F', 'C', 'H' };
    return short_descr;
  }
};

}//namespace Test

#endif//TEST_ON
#endif//BASE_TESTOUTCOME_HH_INCLUDED
