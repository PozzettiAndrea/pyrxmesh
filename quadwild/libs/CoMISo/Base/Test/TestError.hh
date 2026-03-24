// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_TESTERROR_HH_INCLUDED
#define BASE_TESTERROR_HH_INCLUDED

#ifdef TEST_ON

#include <Base/Utils/BaseError.hh>
#include <Base/Utils/OStringStream.hh>

#include <exception>

namespace Test
{

class BASEDLLEXPORT Error : public Base::Error
{
public:
  enum Index
  {
  #define DEFINE_ERROR(CODE, MSG) CODE,
  #include <Base/Test/TestErrorInc.hh>
  #undef DEFINE_ERROR
  };

public:
  //! Constructor.
  Error(const Index _idx) : Base::Error((int)_idx) {}

  //! Return the error message
  virtual const char* message() const;

protected:
  Error(const int _idx) : Base::Error(_idx) {}
};

} // namespace Test

// Throw a Test::Error exception with a specific error index (defined in
// TestErrorInc.hh)
#define TEST_THROW_ERROR(INDEX) { THROW_ERROR_MODULE(Test, INDEX); }
#define TEST_THROW_ERROR_if(COND, INDEX) { if (COND) TEST_THROW_ERROR(INDEX); }

// Throw a 'TODO' Test::Error exception and send a specific message to the debug
// output
#define TEST_THROW_ERROR_TODO(MSG) { THROW_ERROR_TODO_MODULE(Test, MSG); }
#define TEST_THROW_ERROR_TODO_if(COND, MSG) { if (COND) TEST_THROW_ERROR_TODO(MSG); }

// Throw a std::runtime_error exception with a message defined by EXPR
#define TEST_THROW_MSG(EXPR) \
  { \
    Base::OStringStream strm; \
    strm << EXPR; \
    throw std::runtime_error(strm.str); \
  }

#define TEST_THROW_MSG_if(COND, EXPR) { if (COND) TEST_THROW_MSG(EXPR); }

// Throw a std::invalid_argument exception with a message defined by EXPR
#define TEST_ARGS_THROW(EXPR) \
  { \
    Base::OStringStream strm; \
    strm << EXPR; \
    throw std::invalid_argument(strm.str); \
  }

#define TEST_ARGS_THROW_if(COND, EXPR) { if (COND) TEST_ARGS_THROW(EXPR); }

#endif // TEST_ON

#endif // BASE_TESTERROR_HH_INCLUDED
