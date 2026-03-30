// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_CHECKSUMOUTCOME_HH_INCLUDE
#define BASE_CHECKSUMOUTCOME_HH_INCLUDE

#ifdef TEST_ON

#include <Base/Test/TestChecksum.hh>

namespace Test
{
namespace Checksum
{

// Logs the returned result of a (function) call along with any associated error
// message.
class Outcome : public Object
{
public:
  Outcome();

  void record(
      const char* const _call, const bool _ok, const char* _err_msg);
};

// Register the checksum to check call outcomes.
extern Outcome outcome;

} // namespace Checksum
} // namespace Test

#endif // TEST_ON
#endif // BASE_CHECKSUMOUTCOME_HH_INCLUDE
