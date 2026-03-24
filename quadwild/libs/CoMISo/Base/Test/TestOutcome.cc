// (C) Copyright 2021 by Autodesk, Inc.

#ifdef TEST_ON

#include "TestOutcome.hh"

#include <Base/Code/CodeLink.hh>

#include "TestChecksum.hh"
#include "TestChecksumCondition.hh"
#include "TestChecksumOutcome.hh"
#include "TestError.hh"

#include <functional>
#include <iostream>
#include <string>

namespace Test
{

namespace
{
void record_and_handle_unexpected(const char* const _call, const Outcome& _oc,
    const bool _unexpected, const char* const _expctd_what)
{
  if (_unexpected)
  {
    Base::OStringStream strm;
    strm << "Unexpected outcome: Expected " << _expctd_what
         << " but got " << _oc.error_description().c_str() << " instead!";
    TEST(outcome, record(_call, _oc.ok(), strm.str.c_str()));
    _oc.unexpected_handler();
  }
  else
    TEST(outcome, record(_call, _oc.ok(), _oc.error_description().c_str()));
}
} // namespace

const char SUCCESS[] = "Success";
const Outcome FAILURE(false, "FAILURE", "Failure");

Outcome::Outcome() : ok_(true), error_code_(SUCCESS), error_message_(SUCCESS) {}

Outcome::Outcome(
    bool _ok, const std::string& _error_code, const std::string& _error_message)
    : ok_(_ok), error_code_(_error_code), error_message_(_error_message)
{
}

Outcome::Outcome(
    bool _ok, const int _error_code, const std::string& _error_message)
    : ok_(_ok), error_code_(std::to_string(_error_code)),
      error_message_(_error_message)
{
}

const std::string Outcome::error_description() const
{
  return ok() ? error_message() : "[" + error_code() + "] " + error_message();
}

Outcome::UnexpectedHandler Outcome::unexpected_handler = []
{
  TEST_THROW_ERROR(TEST_OUTCOME_UNEXPECTED);
};

const Outcome& record_outcome(const char* const _call, const Outcome& _oc)
{
  // Never call unexpected-outcome handler
  record_and_handle_unexpected(_call, _oc, false, nullptr);
  return _oc;
}

void expect_success(const char* const _call, const Outcome& _oc)
{
  // Call unexpected-outcome handler if the call *failed*
  record_and_handle_unexpected(_call, _oc, !_oc.ok(), SUCCESS);
}

void expect_outcome(
    const char* const _call, const Outcome& _oc, const char* const _error_code)
{
  // Call unexpected-outcome handler if the error code is not correct
  record_and_handle_unexpected(
      _call, _oc, _oc.error_code() != _error_code, _error_code);
}

void expect_outcome(
    const char* const _call, const Outcome& _oc, const int _error_code)
{
  expect_outcome(_call, _oc, std::to_string(_error_code).c_str());
}

void expect_failure(const char* const _call, const Outcome& _oc)
{
  // If _oc represents a failure, replace it with a generic failure outcome
  auto oc = _oc.ok() ? _oc : FAILURE;

  // Call unexpected-outcome handler if the call *succeeds*
  record_and_handle_unexpected(
      _call, oc, oc.ok(), FAILURE.error_code().c_str());
}

void ignore_outcome(const char* const _call, const char* const _reason)
{
  // Write an "Ignored Outcome" checksum to the report
  const auto message = std::string("[IGNORED] ") + _reason;
  record_outcome(_call, Outcome(true, "", message.c_str()));
}

// Pass-through function for Test::Outcome
TEST_OUTCOME_ADD_TYPE(Test::Outcome) { return _oc; }

} // namespace Test

#endif // TEST_ON
