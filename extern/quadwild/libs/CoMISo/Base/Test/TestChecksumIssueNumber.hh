// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_CHECKSUM_ISSUE_NUMBER_HH_INCLUDED
#define BASE_CHECKSUM_ISSUE_NUMBER_HH_INCLUDED

#include <Base/Test/TestChecksum.hh>

#if defined(TEST_ON)

namespace Test
{
namespace Checksum
{

//! Checksum that records and compares a number of issues in a test
class IssueNumber : public Object
{
public:
  //! Constructor
  IssueNumber(const char* const _name, const Level _lvl = L_ALL)
      : Object(_name, _lvl)
  {
  }

  //! Record an error (or warning) if the issue number > 0 
  void record(const size_t _issue_nmbr, const Result _bad_rslt = Result::ERROR);

protected:
  //! Implements a smart comparison for this type of test checksum
  Difference compare_data(const String& _old, const String& _new) const final;
};

} // namespace Checksum
} // namespace Test

#endif // defined(TEST_ON)

#endif//BASE_CHECKSUM_ISSUE_NUMBER_HH_INCLUDED
