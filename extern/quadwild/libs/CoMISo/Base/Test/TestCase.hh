// (C) Copyright 2022 by Autodesk, Inc.

#ifndef BASE_TESTCASE_HH_INCLUDED
#define BASE_TESTCASE_HH_INCLUDED

#ifdef TEST_ON

namespace Test
{
namespace Checksum
{
namespace Report
{

// forward declration of compare.check_equal() to avoid exposure in the tests
void check_equal();

}
} // namespace Checksum
} // namespace Test

/*
The BASE_TEST() macro wraps around an existing test declaration macro (e.g.
TEST_CASE() for Catch2), augmenting it with a checksum comparison. For now, the
existing test declaration macro must define a void function.

In order to use BASE_TEST(), the following macro *must* be defined (before this
file is included):

  TEST_DECLARATION: a placeholder for the existing test declaration macro;

As an example, to use this with Catch2, one would write:

  #define TEST_DECLARATION TEST_CASE

It is not required to use BASE_TEST(), and it is straightforward to define a
custom macro with a similar function. The following macros are available for
use:

- INTERNAL_UNIQUE_NAME(STEM): uses the compiler extension __COUNTER__ to provide
  a unique name that starts with STEM.

- INTERNAL_BASE_TEST_FUNCTION_STEM: A stem that can be used when defining your
  own "Test and Compare" macros.

- INTERNAL_UNIQUE_BASE_TEST_FUNCTION_NAME: A unique function name that can be
  utilised in your own "Test and Compare" macros.
*/

#define INTERNAL_UNIQUE_NAME3(STEM, COUNTER) STEM##COUNTER
#define INTERNAL_UNIQUE_NAME2(STEM, COUNTER) \
  INTERNAL_UNIQUE_NAME3(STEM, COUNTER)
#define INTERNAL_UNIQUE_NAME(STEM) INTERNAL_UNIQUE_NAME2(STEM, __COUNTER__)

#define INTERNAL_BASE_TEST_FUNCTION_STEM ____B_A_S_E____T_E_S_T____
#define INTERNAL_UNIQUE_BASE_TEST_FUNCTION_NAME \
  INTERNAL_UNIQUE_NAME(BASE_TEST_FUNCTION_STEM)

// Both TEST_DECLARATION and TEST_COMPARE_CHECKSUMS must be defined in order to
// use the BASE_TEST() macro.
#ifdef TEST_DECLARATION

#define INTERNAL_BASE_TEST_VOID(TEST_FUNCTION, ...) \
  static void TEST_FUNCTION(); \
  TEST_DECLARATION(__VA_ARGS__) \
  { \
    TEST_FUNCTION(); \
    ::Test::Checksum::Report::check_equal(); \
  } \
  void TEST_FUNCTION()

#define BASE_TEST(...) \
  INTERNAL_BASE_TEST_VOID(INTERNAL_UNIQUE_BASE_TEST_FUNCTION_NAME, __VA_ARGS__)

#else // TEST_DECLARATION

#define BASE_TEST(...) \
  static_assert(false, \
      "TEST_DECLARATION must be defined in order to use the BASE_TEST() " \
      "macro. Please ensure it is defined before TestCase.hh is included."); \
  static void INTERNAL_UNIQUE_BASE_TEST_FUNCTION_NAME()

#endif // TEST_DECLARATION

#endif // TEST_ON
#endif // BASE_TESTCASE_HH_INCLUDED
