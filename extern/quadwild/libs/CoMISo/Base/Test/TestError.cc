// (C) Copyright 2021 by Autodesk, Inc.

#ifdef TEST_ON

#include "TestError.hh"

namespace Test
{

static const char* ERROR_MESSAGE[] =
{
  #define DEFINE_ERROR(CODE, MSG) MSG,
  #include "TestErrorInc.hh"
  #undef DEFINE_ERROR
};

const char* Error::message() const { return ERROR_MESSAGE[idx_]; }

} // namespace Test

#endif // TEST_ON
