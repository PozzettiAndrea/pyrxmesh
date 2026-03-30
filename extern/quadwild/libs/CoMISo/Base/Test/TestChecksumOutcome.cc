// (C) Copyright 2021 by Autodesk, Inc.

#ifdef TEST_ON

#include "TestChecksumOutcome.hh"

#include <string>

namespace Test
{
namespace Checksum
{

Outcome::Outcome() : Object("Outcome", L_STABLE) {}

void Outcome::record(
    const char* const _call, const bool _ok, const char* _err_msg)
{
  Base::OStringStream mess;
  mess << _err_msg << " : " << _call;

  add(_ok ? Result::OK : Result::FAILURE, mess.str);
}

// Register the checksum to check call outcomes.
Outcome outcome;

} // namespace Checksum
} // namespace Test

#endif // TEST_ON
