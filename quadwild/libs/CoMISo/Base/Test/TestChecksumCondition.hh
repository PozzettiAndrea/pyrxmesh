// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_CHECKSUMCONDITION_HH_INCLUDED
#define BASE_CHECKSUMCONDITION_HH_INCLUDED

#include <Base/Debug/DebOut.hh>
#include <Base/Test/TestChecksum.hh>

#if defined(TEST_ON)

// The functions in this file are used to count the number of errors and
// warnings and makes sense only if both the debug and test macro are on.

namespace Base
{
struct CodeLink;
} // namespace Base

namespace Test
{
namespace Checksum
{

class Condition : public Object
{
public:
  Condition() : Object("Condition", L_STABLE), nmbr_(0), fail_nmbr_(0) {}

  virtual void record(
      const char* const _cndt, const Base::CodeLink& _lnk, const bool _rslt);
  virtual void record_number();

protected:
  //! Implement "smarter" comparison
  virtual Difference compare_data(const String& _old, const String& _new) const;

protected:
  int nmbr_;      // number of all checked conditions
  int fail_nmbr_; // number of failed checked conditions
};

extern Condition condition;

} // namespace Checksum

void record_condition(
    const char* const _cndt, const Base::CodeLink& _lnk, const bool _rslt);

} // namespace Test

// Write a Condition checksum to record the outcome of CONDITION.
#define TEST_CONDITION_RECORD(CONDITION) \
  ::Test::record_condition(#CONDITION, BASE_CODELINK, CONDITION)

#else // defined(TEST_ON)

#define TEST_CONDITION_RECORD(CONDITION)

#endif // defined(TEST_ON)

#endif // BASE_CHECKSUMCONDITION_HH_INCLUDED
