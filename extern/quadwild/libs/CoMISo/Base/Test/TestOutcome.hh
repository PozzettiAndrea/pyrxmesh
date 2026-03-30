// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_TEST_OUTCOME_HH_INCLUDE
#define BASE_TEST_OUTCOME_HH_INCLUDE

#ifdef TEST_ON

#include <Base/Code/CodeLink.hh>

#include <functional>
#include <string>

// Write an Outcome checksum to record the outcome of CALL. Returns the outcome.
#define TEST_OUTCOME_RECORD(CALL) \
  ::Test::record_outcome(#CALL, ::Test::Outcome::make(CALL))

// If CALL *fails*, exit the current function and return the outcome.
#define TEST_OUTCOME_RETURN(CALL) \
  { \
    const auto oc = ::Test::Outcome::make(CALL); \
    if (!oc.ok()) \
      return oc; \
  }

// Write an Outcome checksum to record the outcome of CALL. If CALL *fails*,
// exit the current function and return the outcome.
#define TEST_OUTCOME_RECORD_AND_RETURN(CALL) \
  { \
    const auto oc = TEST_OUTCOME_RECORD(CALL); \
    if (!oc.ok()) \
      return oc; \
  }

// Write an Outcome checksum to record the outcome of CALL. An
// unexpected-outcome handler is called if CALL *fails*.
#define TEST_OUTCOME_SUCCESS(CALL) \
  ::Test::expect_success(#CALL, ::Test::Outcome::make(CALL))

// Write an Outcome checksum to record the outcome of CALL. An
// unexpected-outcome handler is called if the error code associated with the
// outcome is not equal to ERROR_CODE.
#define TEST_OUTCOME_EXPECT(CALL, ERROR_CODE) \
  ::Test::expect_outcome(#CALL, ::Test::Outcome::make(CALL), ERROR_CODE)

// Write a success/failure Outcome checksum to record the outcome of CALL. An
// unexpected-outcome handler is called if CALL *succeeds*.
#define TEST_OUTCOME_FAILURE(CALL) \
  ::Test::expect_failure(#CALL, ::Test::Outcome::make(CALL))

// Write an Outcome checksum to record that the outcome of CALL was ignored and
// the REASON why.
#define TEST_OUTCOME_IGNORE(CALL, REASON) \
  (CALL); \
  ::Test::ignore_outcome(#CALL, REASON)

// Write an Outcome checksum to record that the outcome of CALL was ignored and
// that it is unstable (i.e. gives different outcomes on different
// platforms/architectures).
#define TEST_OUTCOME_IGNORE_UNSTABLE(CALL) \
  TEST_OUTCOME_IGNORE(CALL, "Unstable outcome")

/* Use the following macro to add a new outcome type. It defines a
 * specialisation for Test::Outcome::make<TYPE>(const TYPE& _oc), and must
 * return a Test::Outcome object.
 *
 * Usage:
 *
 *   TEST_OUTCOME_ADD_TYPE(TYPE)
 *   {
 *     // Some additional computations may go here
 *
 *     return Test::Outcome([is _oc ok?],
 *                          [_oc error code],
 *                          [_oc error message],
 *                          [unexpected-outcome handler]);
 *   }
 *
 * where [unexpected-outcome handler] is a function of type
 * Test::Outcome::UnexpectedHandler to call if the outcome is not what was
 * expected.
 */
#define TEST_OUTCOME_ADD_TYPE(TYPE) \
  template <>::Test::Outcome Test::Outcome::make<TYPE>(const TYPE& _oc)

namespace Test
{

/* A generic outcome class that is intended to be a wrapper for all other
 * outcome classes. It has the following member functions:
 *
 *  - constructor:          Calling the constructor with no arguments creates a
 *                          successful outcome.
 *
 *  - make<T>():            Creates an Outcome object from an object of type T.
 *                          A template specialisation should be added for each T
 *                          that is required by the consuming project (see
 *                          TEST_OUTCOME_ADD_TYPE() macro).
 *
 *  - ok():                 Returns whether the outcome was a success.
 *
 *  - error_code():         Returns the associated error code as a std::string&.
 *
 *  - error_message():      Returns the associated error message.
 *
 *  - error_description():  Returns a concatenation of the error code and error
 *                          message.
 *
 * It also contains the static variable unexpected_handler, a std::function that
 * can be called if the outcome is not what was expected.
 */
class Outcome
{
public:
  typedef std::function<void()> UnexpectedHandler;

  Outcome(); // Construct successful outcome

  Outcome(bool _ok, const std::string& _error_code,
      const std::string& _error_message);

  Outcome(bool _ok, const int _error_code, const std::string& _error_message);

  template <class T> static Outcome make(const T& _oc);

  const bool ok() const { return ok_; }

  const std::string& error_code() const { return error_code_; }

  const std::string& error_message() const { return error_message_; }

  const std::string error_description() const;

  static UnexpectedHandler unexpected_handler;

private:
  bool ok_;
  std::string error_code_;
  std::string error_message_;

};

const Outcome& record_outcome(const char* const _call, const Outcome& _oc);

void expect_success(const char* const _call, const Outcome& _oc);

void expect_outcome(
    const char* const _call, const Outcome& _oc, const char* const _error_code);

void expect_outcome(
    const char* const _call, const Outcome& _oc, const int _error_code);

void expect_failure(const char* const _call, const Outcome& _oc);

void ignore_outcome(const char* const _call, const char* const _reason);

} // namespace Test

#endif // TEST_ON

#endif // BASE_TEST_OUTCOME_HH_INCLUDE
