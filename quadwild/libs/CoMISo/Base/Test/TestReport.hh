// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_TESTREPORT_HH_INCLUDED
#define BASE_TESTREPORT_HH_INCLUDED

#ifdef TEST_ON


#include <functional>

namespace Test
{
enum ExitStatus
{
  ES_OK = 0,
  ES_DIFFERENCES = 1, // Test compare has detected differences
  ES_WRONG_ARGUMENT,
  ES_ROOT_DIR_NOT_FOUND,
  ES_NO_OUT_FILE,
  ES_COMPARE_FAILED,
  ES_SYNC_FAILED
};

/*!
Determines the action of the make_comparison() function:

* COT_SHORT_DIFF: Produce a comparison/diff report that only includes checksums
  that are different between <_dir_left> and <_dir_right>.

* COT_FULL_DIFF: Produce a comparison/diff report that includes all checksums,
  even those that are identical, in <_dir_left> and <_dir_right>.

* COT_UPDATE: Copy all summary reports from <_dir_right> to <_dir_left>
  (potentially overwriting summary reports in <_dir_left>). Any summary reports
  that exist in <_dir_left> but not in <_dir_right> are *unaffected*.

* COT_MIRROR: Copy all summary reports from <_dir_right> to <_dir_left>
  (potentially overwriting summary reports in <_dir_left>). Any summary reports
  that exist in <_dir_left> but not in <_dir_right> are *deleted*.
*/
enum CompareOutputType
{
  COT_SHORT_DIFF,
  COT_FULL_DIFF,
  COT_UPDATE,
  COT_MIRROR
};

/*!
Compares test reports in two test directory trees. Its behaviour is modified by
_cot (see descriptions above). Setting _show_progress = false reduces the amount
of progress-related output from the function.
*/
ExitStatus make_comparison(const char* const _dir_left,
    const char* const _dir_right, const CompareOutputType _cot,
    const bool _show_progress = true);

/*!
This is a wrapper to make_comparison() that parses command-line arguments. The
following command-line format is expected:

  > <exec> dir_left dir_right [--full-diff|--update|--mirror] [--no-progress]

* --full-diff, --update and --mirror affect the value of _cot;
* --no-progress will set _show_progress = false.

*/
int report(const int _argc, const char* const _argv[]);

} // namespace Test

#endif // TEST_ON
#endif // BASE_TESTREPORT_HH_INCLUDED
