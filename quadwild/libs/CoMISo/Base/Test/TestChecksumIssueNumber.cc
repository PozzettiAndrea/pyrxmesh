// (C) Copyright 2021 by Autodesk, Inc.

#ifdef TEST_ON

#include "TestChecksumIssueNumber.hh"
#include "TestResult.hh"

#include <sstream>

namespace Test
{
namespace Checksum
{

void IssueNumber::record(const size_t _issue_nmbr, const Result _bad_rslt)
{
  Object::record(_issue_nmbr == 0 ? Result::OK : _bad_rslt, _issue_nmbr);
}

Difference IssueNumber::compare_data(
    const String& _old, const String& _new) const
{
  std::istringstream strm_old(_old), strm_new(_new);
  size_t val_old, val_new;
  strm_old >> val_old;
  strm_new >> val_new;

  if (val_new < val_old)
    return Difference::IMPROVED;
  else if (val_new > val_old)
    return Difference::REGRESSED;
  else
    return Difference::EQUAL;
}

} // namespace Checksum
} // namespace Test

#endif // TEST_ON
