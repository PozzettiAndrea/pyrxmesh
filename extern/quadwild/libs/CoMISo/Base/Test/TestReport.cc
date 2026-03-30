// (C) Copyright 2022 by Autodesk, Inc.

#ifdef TEST_ON

#include <Base/Security/Mandatory.hh>
#include <Base/Paths/Filesystem.hh>
#include <Base/Paths/PathLink.hh>
#include <Base/Utils/BaseError.hh>
#include "TestChecksum.hh"
#include "TestChecksumCompletion.hh"
#include "TestChecksumReport.hh"
#include "TestReport.hh"
#include "TestResult.hh"

#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace Test
{

using PathLink = Base::PathLink;

namespace
{
const std::locale C_LOCALE {"C"};
const char* const INDENT = "    ";
const char CR = (char)'\r';

namespace fs = Base::filesystem;

/*!
Extracts the list of executed tests (by their relative path names) from a test
suite directory.
*/
class TestTree
{
public:
  explicit TestTree(const char* _dir);

  const fs::path& root_dir() const { return root_dir_; }

  const std::set<fs::path>& tests() const { return tests_; }

private:
  fs::path root_dir_;

  // Set of test paths
  std::set<fs::path> tests_;

  // Scan for all tests that recorded checksums (i.e. all directories that
  // contain report.txt) and store their relative paths in tests_.
  void collect_tests(const fs::path& _dir);

  // Parse the CTest log: warn if the list of tests in the CTest log is
  // different to the list obtained by collect_tests() (which looks for checksum
  // reports).
  void parse_ctest_log(const fs::path& _ctest_log_path);
};

TestTree::TestTree(const char* _dir)
{
  // Exit with error code if path is not a directory
  if (!fs::is_directory(_dir))
  {
    std::cout << "ERROR: Path " << _dir << " is not a directory!" << std::endl;
    exit(ES_ROOT_DIR_NOT_FOUND);
  }
  root_dir_ = _dir;

  // Generate a list (from the file system) of all tests run.
  collect_tests(root_dir_);

  // Check to see if a CTest log of the run exists and parse it. Typically, a
  // CTest log will exist for a test run that has just completed, but will not
  // exist for stored baselines.
  fs::path ctest_log_path(root_dir_ / "ctest.log");
  if (fs::is_regular_file(ctest_log_path))
    parse_ctest_log(ctest_log_path);
}

void TestTree::parse_ctest_log(const fs::path& _ctest_log_path)
{
#if __APPLE__
  std::ifstream ctest_log(_ctest_log_path.string());
#else  // __APPLE__
  std::ifstream ctest_log(_ctest_log_path);
#endif // __APPLE__
  std::string line;

#ifdef TEST_REPORT_CTEST_NAMES_ARE_OUTPUT_PATHS
  /*
  The variable tests_ contains the set of paths to directories that contain
  'report.txt' (the checksum output file). Since 'CTEST_NAMES_ARE_OUTPUT_PATHS',
  we can think of tests_ as the set of tests for which checksums were recorded.

  The CTest log contains the list of tests that were run.

  Below, we parse the CTest log to extract the set of tests, ctest_tests, and
  compare it to tests_ in order to determine the tests that were run but for
  which no checksums were recorded. Note: we assume that ctests_tests is a
  superset of tests_.
  */

  // Compile a set of the tests that are listed in the CTest log.
  std::set<fs::path> ctest_tests;
  while (std::getline(ctest_log, line))
  {
    /*
    Reads the list of tests by parsing a line like this:

      1/120 Test   #2: generic/LCS/lcs.cc ................   Passed    2.03 sec

    Here, ctest_index = 1, c = '/', ctest_nmbr = 120, test_text = "Test",
    test_index = "#2:" and test_name = "generic/LCS/lcs.cc"
    */
    size_t ctest_index = 0, ctest_nmbr = 0;
    char c;
    std::string test_text, test_index, test_name;

    std::istringstream line_ss(line);
    line_ss >> ctest_index >> c >> ctest_nmbr >> test_text >> test_index >>
        test_name;

    if (line_ss && c == '/')
      ctest_tests.emplace(test_name);

    /*
    Break when we reach a line like

      Total Test time (real) =   4.86 sec

    as this will signify the end of the test list in the log.
    */
    if (line.rfind("Total Test time", 0) == 0)
      break;
  }

  // Find the set of tests both listed in the CTest log and for which no
  // checksums were recorded (i.e. ctest_tests - tests_).
  std::set<fs::path> ctests_no_checksum;
  std::set_difference(ctest_tests.begin(), ctest_tests.end(), tests_.begin(),
      tests_.end(),
      std::inserter(ctests_no_checksum, ctests_no_checksum.end()));

  if (ctests_no_checksum.size() > 0)
  {
    std::cout << std::endl
              << "WARNING: The following tests were referenced in the CTest "
                 "log but no checksums were recorded for them:"
              << std::endl;

    for (auto& test : ctests_no_checksum)
      std::cout << " * " << test.generic_string() << std::endl;

    std::cout << std::endl;
  }
#else  // TEST_REPORT_CTEST_NAMES_ARE_OUTPUT_PATHS
  /*
  The variable tests_ contains the set of paths to directories that contain
  'report.txt' (the checksum output file). The CTest log contains the list of
  tests that were run.

  Below, we parse the CTest log to extract the number of tests that were run,
  ctest_tests_nmbr, and compare it to tests_.size() in order to determine the
  number of tests that were run but for which no checksums were recorded.
  */

  // Parse the CTest log and find the number of tests run.
  size_t ctest_tests_nmbr = 0;
  while (std::getline(ctest_log, line))
  {
    /*
    Reads the number of tests by parsing a line like this:

      1/120 Test   #2: generic/LCS/lcs.cc ................   Passed    2.03 sec

    Here, ctest_index = 1, c = '/' and ctest_nmbr = 120.
    */
    size_t ctest_index = 0, ctest_nmbr = 0;
    char c;
    std::stringstream line_ss(line);
    if ((line_ss >> ctest_index >> c >> ctest_nmbr) && ctest_index == 1 &&
        c == '/')
    {
      ctest_tests_nmbr = ctest_nmbr;
      break;
    }
  }

  // Display a warning if the number of tests listed in the CTest log is
  // different to the number of tests for which checksums were recorded (i.e.
  // tests found by scanning the directory using collect_tests()).
  if (tests_.size() != ctest_tests_nmbr)
  {
    std::cout << std::endl
              << "WARNING: There are " << ctest_tests_nmbr
              << " tests listed in the CTest log, but checksums were recorded "
                 "for only "
              << tests_.size() << " test(s)." << std::endl
              << std::endl;
  }
#endif // TEST_REPORT_CTEST_NAMES_ARE_OUTPUT_PATHS
}

void TestTree::collect_tests(const fs::path& _dir)
{
  bool is_test = false;
  for (auto& entry : fs::directory_iterator(_dir))
  {
#if __APPLE__
    if (fs::is_directory(entry.path()))
#else // __APPLE__
    if (entry.is_directory())
#endif // __APPLE__
      collect_tests(entry.path());
    else if (entry.path().filename() == Test::REPORT_FILENAME)
      is_test = true;
  }

  // If the path corresponds to a test output directory, add it (relative to
  // root_dir_) to the list.
  if (is_test)
    tests_.emplace(fs::relative(_dir, root_dir_));
}

// Inner class that implements the stream operations.
typedef std::stringstream TestStream;

// Store compare information about any test that has been executed in both
// suites.
struct TestDiffSummary
{
  TestDiffSummary(const fs::path& _test_path,
      const Checksum::Report::DifferenceDistribution& _diff,
      const TestStream& _descr)
      : path_(_test_path), diffs_(_diff), descr_(_descr.str())
  {
  }

  bool operator>(const TestDiffSummary& _oth) const
  {
    if (diffs_.empty() && _oth.diffs_.empty())
      return false;

    if (diffs_.empty())
      return false;

    if (_oth.diffs_.empty())
      return true;

    auto& last = *(diffs_.rbegin());
    auto& oth_last = *(_oth.diffs_.rbegin());
    if (last.first.type() == oth_last.first.type())
    {
      if (last.second == oth_last.second)
        return path_ > _oth.path_;
      return last.second > oth_last.second;
    }

    return last.first.type() > oth_last.first.type();
  }

  void log_summary(std::ofstream& _log_file) const
  {
    for (const auto& diff : diffs_)
      _log_file << diff.first.type_text() << ": " << diff.second << "; ";
    _log_file << std::endl;
  }

  bool equivalent() const { return diffs_.empty(); }

  fs::path path_;
  Checksum::Report::DifferenceDistribution diffs_;
  std::string descr_; // Test report of test differences description.
};

// Save the comparisons for all tests common between the left and right suites.
// Individual comparison diffs are stored in the right-suite directory for each
// test.
void save_diff_reports(std::ofstream& log_file,
    const std::list<TestDiffSummary>& diff_summary_list,
    const TestTree (&test_trees)[2], const bool _short_frmt = false)
{
  for (const auto& diff_summary : diff_summary_list)
  {
    // Write the test path to the log file
    log_file << diff_summary.path_.generic_string() << ":" << INDENT;

    // Write diff statistics to the log file
    diff_summary.log_summary(log_file);

    // If we're writing the short diff and there are no differences, skip to the
    // next test
    if (diff_summary.equivalent() && _short_frmt)
    {
      log_file << std::endl;
      continue;
    }

    // Only create an individual report (and reference it in log_file) if there
    // are differences
    if (!diff_summary.equivalent())
    {
      // Open a stream to the individual comparison diff (stored in the
      // right-suite test directory).
      const auto diff_path =
          test_trees[1].root_dir() / diff_summary.path_ /
          (_short_frmt ? "baseline_short.diff" : "baseline_full.diff");

#if __APPLE__
      std::ofstream diff_strm(diff_path.string());
#else  // __APPLE__
      std::ofstream diff_strm(diff_path);
#endif // __APPLE__

      // Write path links to a few directories relevant to this test
      diff_strm << "Left-suite test directory: "
                << PathLink(test_trees[0].root_dir() / diff_summary.path_)
                << std::endl;
      diff_strm << "Right-suite test directory: "
                << PathLink(test_trees[1].root_dir() / diff_summary.path_)
                << std::endl;
      diff_strm << std::endl;

      diff_strm << "Left-suite test report: "
                << PathLink(test_trees[0].root_dir() / diff_summary.path_ /
                            Test::REPORT_FILENAME)
                << std::endl;
      diff_strm << "Right-suite test report: "
                << PathLink(test_trees[1].root_dir() / diff_summary.path_ /
                            Test::REPORT_FILENAME)
                << std::endl;
      diff_strm << std::endl;

      // Diff summary description is not empty, so write it to diff_strm
      diff_strm << diff_summary.descr_ << std::endl;

      // Put a path link to diff_file (i.e. diff_path) in the global diff file.
      log_file << INDENT << PathLink(diff_path) << std::endl;
    }

    log_file << std::endl;
  }
}

// Construct an appropriate path for the diff file
fs::path construct_diff_path(const fs::path& _root_left,
    const fs::path& _root_right, const bool _short_frmt)
{
  // Get the path to the left-suite root directory
  auto diff_flnm = _root_left.generic_string();

  // Replace folder slashes with _ and make the string lowercase
  for (auto& c : diff_flnm)
    c = (c == '/') ? '_' : char(std::tolower(c, C_LOCALE));

  // If "test_" appears in the string, erase it and everything to the left of it
  auto test_pos = diff_flnm.find("test_");
  if (test_pos != std::string::npos)
    diff_flnm.erase(diff_flnm.begin(), diff_flnm.begin() + (test_pos + 5));

  // Append either "_short.diff" or "_full.diff" depending on the type of
  // comparison requested
  diff_flnm += (_short_frmt ? "_short.diff" : "_full.diff");

  // Place diff in the right-suite root directory
  return _root_right / diff_flnm;
}


// Add a border of asterisks around multiple strings. Each string is put on a
// new line.
std::string add_border(const std::vector<std::string>& _messages)
{
  using Base::ENDL;
  const auto max_msg_size = std::max_element(_messages.begin(), _messages.end(),
      [](const std::string& _a, const std::string& _b)
      { return _a.size() < _b.size(); })->size();

  const std::string border(max_msg_size + 4, '*');

  Base::OStringStream strm;
  strm << ENDL << border << ENDL;
  for (const auto& message : _messages)
  {
    const std::string padding(max_msg_size - message.size(), ' ');
    strm << "* " << message << padding << " *" << ENDL;
  }
  strm << border << ENDL;
  return strm.str;
}

// Add a border of asterisks around a string
std::string add_border(const std::string& _message)
{
  return add_border(std::vector<std::string>(1, _message));
}

// Namespace for functions that deal with updating and mirroring test
// directories
namespace Sync
{

// Copy the report in _root_from/_rel_path/ to _root_to/_rel_path/
void copy(const fs::path& _root_from, const fs::path& _root_to,
    const fs::path& _rel_path)
{
  const fs::path dir_from = _root_from / _rel_path;
  const fs::path dir_to = _root_to / _rel_path;

  const fs::path report_from = dir_from / Test::REPORT_FILENAME;
  const fs::path report_to = dir_to / Test::REPORT_FILENAME;

  // Create dir_to (and all parent directories if necessary)
  fs::create_directories(dir_to);

  // Copy the contents of dir_from to dir_to
  Base::error_code err_code;
  fs::copy(report_from, report_to, err_code);
  if (err_code)
  {
    std::cout << "WARNING: Failed to copy '" << dir_from.generic_string()
              << "' to '" << dir_to.generic_string()
              << "'. Associated error message: '" << err_code.message() << "'"
              << std::endl;
  }
}

// Copy the directory _root_from/test_path/ to _root_to/test_path/ for every
// test in the set _diff
template <class SetT>
void add(
    const fs::path& _root_from, const fs::path& _root_to, const SetT& _diff)
{
  for (const auto& test_path : _diff)
  {
    std::cout << "ADD " << test_path.generic_string() << std::endl;
    copy(_root_from, _root_to, test_path);
  }
}

// Remove the directory _root/test_path/ for every test in the set _diff
template <class SetT>
void remove(const fs::path& _root, const SetT& _diff)
{
  for (const auto& test_path : _diff)
  {
    std::cout << "REMOVE " << test_path.generic_string() << std::endl;
    fs::remove_all(_root / test_path);
  }
}

// Remove the existing contents of _root_target/_rel_path/ and copy over the
// report from _root_source/_rel_path/
void replace(const fs::path& _root_source, const fs::path& _root_target,
    const fs::path& _rel_path)
{
  std::cout << "REPLACE " << _rel_path.generic_string() << std::endl;
  fs::remove_all(_root_target / _rel_path);
  copy(_root_source, _root_target, _rel_path);
}

} // namespace Sync

} // namespace

// Compares the reports from the executed tests in _dir_right (the 'right
// suite') against those in _dir_left (the 'left suite'). Can produce a
// short/full diff or update/mirror the tests in the left suite based on the
// right.
ExitStatus make_comparison(const char* const _dir_left,
    const char* const _dir_right, const CompareOutputType _cot,
    const bool _show_progress)
{
  // Find the executed tests in the two test suites.
  TestTree test_trees[2]{TestTree(_dir_left), TestTree(_dir_right)};
  std::list<TestDiffSummary> diff_summary_list;

  // Find the common set of tests.
  std::vector<fs::path> common_tests;
  std::set_intersection(test_trees[0].tests().cbegin(),
      test_trees[0].tests().cend(), test_trees[1].tests().cbegin(),
      test_trees[1].tests().cend(),
      std::inserter(common_tests, common_tests.end()));

  // Compare the checksums produced by each test in the common set.
  size_t common_tests_nmbr = common_tests.size();
  size_t diff_test_nmbr = 0, negl_diff_test_nmbr = 0, test_idx = 0;

  if (!_show_progress)
    std::cout << "Comparing tests...";

  for (const auto& test_path : common_tests)
  {
    if (_show_progress)
    {
      std::cout << CR << "Comparing test " << ++test_idx << " of "
                << common_tests_nmbr << "...";
    }

    // Create stream to store the differences for the current test
    Base::OutputStreamAdaptT<TestStream> test_diff_log;

    // Set comparison parameters
    Checksum::Report::compare.set_reports(
        (test_trees[0].root_dir() / test_path / Test::REPORT_FILENAME)
            .generic_string(),
        (test_trees[1].root_dir() / test_path / Test::REPORT_FILENAME)
            .generic_string());

    Checksum::Report::compare.set_short_format(_cot == COT_SHORT_DIFF);

    // Compare the checksums for the test between the two suites
    const auto diff_stats = Checksum::Report::compare.run(test_diff_log);

    if (!diff_stats.empty())
    { // Differences were found

      // Count the number of tests for which differences were found
      ++diff_test_nmbr;

      if (diff_stats.size() == 1 &&
          diff_stats.find(Checksum::Difference::NEGLIGIBLE) != diff_stats.end())
      {
        ++negl_diff_test_nmbr; // all differences are negligible
      }

      // If set to update or mirror the tests, empty the left-suite test output
      // directory and copy over the checksum report from the right-suite test
      // output directory.
      if (_cot == COT_UPDATE || _cot == COT_MIRROR)
      {
        Sync::replace(
            test_trees[1].root_dir(), test_trees[0].root_dir(), test_path);
      }
    }

    // If there is a difference, or if the full diff has been requested, add the
    // difference summary to the log.
    if (!diff_stats.empty() || _cot == COT_FULL_DIFF)
      diff_summary_list.emplace_back(
          test_path, diff_stats, test_diff_log.stream());
  }
  std::cout << "complete." << std::endl;

  const auto test_string = [](size_t _nmbr)
  {
    auto res = std::to_string(_nmbr) + " TEST";
    if (_nmbr != 1)
      res += "S";
    return res;
  };

  // Print the comparison summary.
  std::vector<std::string> messages;
  messages.push_back("COMPARISON HAS DETECTED DIFFERENCES IN " +
                     test_string(diff_test_nmbr) + ".");
  messages.push_back(std::string("THE DIFFERENCES ARE NEGLIGIBLE IN ") +
                     ((diff_test_nmbr == negl_diff_test_nmbr) ? "ALL " : "") +
                     test_string(negl_diff_test_nmbr) + ".");
  std::cout << add_border(messages) << std::endl;

  // Are we creating diffs?
  const bool create_diffs = _cot == COT_SHORT_DIFF || _cot == COT_FULL_DIFF;

  // Create the global and individual diffs if they have been requested.
  std::ofstream glbl_diff_strm;
  fs::path glbl_diff_path;
  if (create_diffs)
  {
    // Construct an appropriate path for the global diff file
    glbl_diff_path = construct_diff_path(test_trees[0].root_dir(),
        test_trees[1].root_dir(), _cot == COT_SHORT_DIFF);

    // Open a stream to the global diff file
#if __APPLE__
    glbl_diff_strm.open(glbl_diff_path.string());
#else  // __APPLE__
    glbl_diff_strm.open(glbl_diff_path);
#endif // __APPLE__
    if (!glbl_diff_strm)
    {
      std::cout << "ERROR: Cannot create the global diff report '"
                << glbl_diff_path.generic_string() << "'." << std::endl;
      return ES_NO_OUT_FILE;
    }

    // For the short diff, order the differences in descending order by severity
    // (i.e. most severe differences first).
    if (_cot == COT_SHORT_DIFF)
      diff_summary_list.sort(std::greater<TestDiffSummary>());

    // Save global and individual diffs
    save_diff_reports(
        glbl_diff_strm, diff_summary_list, test_trees, _cot == COT_SHORT_DIFF);

    glbl_diff_strm << add_border("Differences detected in " +
                                 std::to_string(diff_test_nmbr) + " test" +
                                 (diff_test_nmbr == 1 ? "" : "s") + ".")
                   << std::endl;
  }

  // Finds and logs the tests executed in only in left or right suites.
  const char* suite_name[] = {"left", "right"};
  bool found_diff = diff_test_nmbr > 0;
  bool is_subset[2]; // Is test_trees[i] a subset of test_trees[1 - i]?
  for (int i = 2; i-- > 0;)
  {
    // Get a list of tests that are in test_trees[i] but not test_trees[1 - i]
    std::vector<fs::path> diff_tests;
    std::set_difference(test_trees[i].tests().cbegin(),
        test_trees[i].tests().cend(), test_trees[1 - i].tests().cbegin(),
        test_trees[1 - i].tests().cend(),
        std::inserter(diff_tests, diff_tests.end()));

    // If there are no such tests (i.e. test_trees[i] is a subset of
    // test_trees[1 - i]), skip to the next test list
    is_subset[i] = diff_tests.empty();
    if (is_subset[i])
      continue;

    // UPDATE:
    // * if a test exists in the right suite but not in the left suite (i.e.
    //   i == 1), add it to the left suite.
    //
    // MIRROR:
    // * Same as UPDATE;
    // * if a test exists in the left suite but not in the right suite (i.e.
    //   i == 0), remove it.
    if (_cot == COT_UPDATE || _cot == COT_MIRROR)
    {
      if (i == 1)
      {
        Sync::add(
            test_trees[1].root_dir(), test_trees[0].root_dir(), diff_tests);
      }
      else if (i == 0 && _cot == COT_MIRROR)
        Sync::remove(test_trees[0].root_dir(), diff_tests);
    }

    // List the discovered tests in the diff report (if one has been requested)
    if (create_diffs)
    {
      glbl_diff_strm << "Tests only in "
                     << test_trees[i].root_dir().generic_string() << ": "
                     << std::endl
                     << std::endl;

      for (const auto& test_path : diff_tests)
        glbl_diff_strm << INDENT << test_path.generic_string() << std::endl;

      std::cout << diff_tests.size() << " tests executed are only in the "
                << suite_name[i] << " suite." << std::endl;
    }
  }

  if (create_diffs && (found_diff || !is_subset[0] || !is_subset[1]))
  {
    std::cout << "\nPlease check the compare log: " << PathLink(glbl_diff_path)
              << std::endl;

    // Return an error if a difference was found or if there is a test in the
    // right suite that does not exist in the left suite.
    if (found_diff || !is_subset[1])
      return ES_DIFFERENCES;
  }
  return ES_OK;
}

#define TRY_SET_COT(OPT, COT) \
  if (strcmp(_argv[i], OPT) == 0) \
  { \
    if (cot != COT_SHORT_DIFF) \
    { \
      std::cout << "Compare Output Type has already been set. " << usage \
                << std::endl; \
      return ES_WRONG_ARGUMENT; \
    } \
    cot = COT; \
    continue; \
  }

#define TEST_REPORT_CATCH_EXCEPTION(TYPE, MESSAGE) \
  catch (TYPE) \
  { \
    std::cerr << "FAILED " << error_msg \
              << " due to unexpected exception with message:" << std::endl \
              << "  " << MESSAGE << std::endl \
              << std::endl; \
    rslt = error_rslt; \
  }

#define TEST_REPORT_CATCH_SPECIFIC_EXCEPTION(TYPE, MESSAGE) \
  TEST_REPORT_CATCH_EXCEPTION(const TYPE& _e, _e.MESSAGE)

#define TEST_REPORT_CATCH_UNKNOWN_EXCEPTION \
  TEST_REPORT_CATCH_EXCEPTION(..., "Unknown exception")

int report(const int _argc, const char* const _argv[])
{
  // Set usage text
  const auto usage =
      "Usage: " + std::string(_argv[0]) +
      " dir_left dir_right [--full-diff|--update|--mirror] [--no-progress]";

  // Check if we have the right number of arguments
  if (_argc < 3 || _argc > 5)
  {
    std::cout << "Invalid number of arguments. " << usage << std::endl;
    return ES_WRONG_ARGUMENT;
  }

  // Set default comparison options
  auto cot = COT_SHORT_DIFF;
  bool show_progress = true;

  // Parse arguments
  const char* const& dir_left = _argv[1];
  const char* const& dir_right = _argv[2];

  for (int i = 3; i < _argc; i++)
  {
    TRY_SET_COT("--full-diff", COT_FULL_DIFF);
    TRY_SET_COT("--update", COT_UPDATE);
    TRY_SET_COT("--mirror", COT_MIRROR);

    if (strcmp(_argv[i], "--no-progress") == 0)
    {
      show_progress = false;
      continue;
    }

    std::cout << "Unrecognised argument. " << usage << std::endl;
    return ES_WRONG_ARGUMENT;
  }

  // Print helpful information
  auto rslt = ES_OK;

  ExitStatus error_rslt;
  std::string error_msg;

  if (cot == COT_UPDATE || cot == COT_MIRROR)
  {
    std::cout << (cot == COT_UPDATE ? "Updating" : "Mirroring") << dir_left
              << " to match " << dir_right << "..." << std::endl;
    error_msg = cot == COT_UPDATE ? "update" : "mirror";
    error_rslt = ES_SYNC_FAILED;
  }
  else
  { // cot == COT_SHORT_DIFF or COT_FULL_DIFF
    const std::string diff_type(cot == COT_SHORT_DIFF ? "short" : "full");
    std::cout << "Producing the " << diff_type << " difference report..."
              << std::endl;
    error_msg = "to generate the " + diff_type + " difference report";
    error_rslt = ES_COMPARE_FAILED;
  }

  // Perform the appropriate comparison and catch any errors
  try
  {
    rslt = make_comparison(dir_left, dir_right, cot, show_progress);
  }
  TEST_REPORT_CATCH_SPECIFIC_EXCEPTION(Base::Error, message())
  TEST_REPORT_CATCH_SPECIFIC_EXCEPTION(std::exception, what())
  TEST_REPORT_CATCH_UNKNOWN_EXCEPTION

  return rslt;
}

#undef TRY_SET_COT

#undef TEST_REPORT_CATCH_EXCEPTION
#undef TEST_REPORT_CATCH_SPECIFIC_EXCEPTION
#undef TEST_REPORT_CATCH_UNKNOWN_EXCEPTION

} // namespace Test

#endif // TEST_ON
