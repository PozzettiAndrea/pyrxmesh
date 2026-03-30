// (C) Copyright 2022 by Autodesk, Inc.

#ifdef TEST_ON

#include <Base/Security/Mandatory.hh>

#include "LongestCommonSubsequenceT.hh"
#include "TestChecksum.hh"
#include "TestChecksumCompletion.hh"
#include "TestChecksumReport.hh"
#include "TestError.hh"

#include <Base/Code/Quality.hh>
#include <Base/Debug/DebCallStack.hh>
#include <Base/Paths/Filesystem.hh>
#include <Base/Paths/PathLink.hh>
#include <Base/Utils/NullOutputStream.hh>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>

namespace Test
{
using PathLink = Base::PathLink;

namespace Checksum
{
Level run_lvl = L_NONE;

namespace
{
const std::locale C_LOCALE {"C"};

namespace fs = Base::filesystem;

/*!
Definition of the checksums registry. It is a map from a string (that is the
checksum name to an Checksum::Object.
*/
typedef std::map<String, Object*> Registry;

// Get a modifiable copy of the registry
Registry& registry()
{
  static Registry chksm_reg;
  return chksm_reg;
}

// Trim (remove whitespace) from the end of a std::string (in place)
inline void rtrim(std::string& s)
{
  s.erase(std::find_if(s.rbegin(), s.rend(),
              [](char ch) { return !std::isspace(ch, C_LOCALE); })
              .base(),
      s.end());
}

// Remove code links from _str. Code links are expected to be in the format for
// Base::CodeLink used in IOutputStream.cc
void erase_code_links(std::string& _str)
{
  const char* const CODE_LINK_BEGIN = "@ [";
  const char* const CODE_LINK_END = "]";

  for (auto bgn_pstn = _str.find(CODE_LINK_BEGIN);
       bgn_pstn != std::string::npos && !_str.empty();
       bgn_pstn = _str.find(CODE_LINK_BEGIN, bgn_pstn))
  {
    const auto end_pstn = _str.find(CODE_LINK_END, bgn_pstn);
    if (end_pstn == std::string::npos)
      break;
    _str.erase(bgn_pstn, end_pstn - bgn_pstn + 1);
  }
}
} // namespace

///////////////////////////////////////////////////////////////////////////////
//// class Object implementation (for checksums)
///////////////////////////////////////////////////////////////////////////////

Object::Object(const char* const _name, const Level _lvl)
  : lvl_(_lvl), name_(_name)
{
  auto pos = registry().emplace(name_, this);
  if (!pos.second)
  {
    std::cout << "Duplicate checksum definition: " << _name << std::endl;
    throw;
  }
}

Difference Object::compare_data(const String& _old, const String& _new) const
{
  if (_old == _new)
    return Difference::EQUAL;

  String diff;
  diff.resize((std::max(_old.length(), _new.length())));
  for (size_t i = 0; i < diff.size(); ++i)
  {
    diff[i] = i < _old.length() && i < _new.length() && _old[i] == _new[i] ?
      ' ' : '*';
  }
  return Difference(Difference::UNKNOWN, diff);
}

Difference Object::compare(
    const Record& _old_rcrd, const Record& _new_rcrd) const
{
  const auto old_rslt_type = _old_rcrd.rslt.type();
  const auto new_rslt_type = _new_rcrd.rslt.type();
  auto data_diff = compare_data(_old_rcrd.data, _new_rcrd.data);
  if (old_rslt_type == new_rslt_type) // same result, so just data compare
    return data_diff;

  // result types are different, qualify the difference
#define DIFFERENCE(TYPE) data_diff += Difference(Difference::TYPE)
  switch (old_rslt_type)
  {
  case Result::OK:
    switch (new_rslt_type)
    {
    case Result::WARNING: DIFFERENCE(SUSPICIOUS); break;
    case Result::ERROR: DIFFERENCE(REGRESSED); break;
    default: DIFFERENCE(FAILED); break; // FAILURE, CRASH, HANG
    };
    break;
  case Result::WARNING :
    switch (new_rslt_type)
    {
    case Result::OK: DIFFERENCE(IMPROVED); break;
    case Result::ERROR: DIFFERENCE(REGRESSED); break;
    default: DIFFERENCE(FAILED); break; // FAILURE, CRASH, HANG
    };
    break;
  case Result::ERROR :
    switch (new_rslt_type)
    {
    case Result::OK:
    case Result::WARNING: DIFFERENCE(IMPROVED); break;
    default: DIFFERENCE(FAILED); break; // FAILURE, CRASH, HANG
    };
    break;
  case Result::FAILURE:
    switch (new_rslt_type)
    {
    // worked with or w/o issues, now fails
    case Result::OK:
    case Result::WARNING:
    case Result::ERROR: DIFFERENCE(WORKED); break;
    // gracious failure replaced by HANG or CRASH?!
    default: DIFFERENCE(FAILED); break; // CRASH, HANG
    };
    break;
  case Result::CRASH:
    switch (new_rslt_type)
    {
    case Result::OK:
    case Result::WARNING:
    case Result::ERROR: DIFFERENCE(WORKED); break;
    // CRASH replaced by gracious failure!
    case Result::FAILURE: DIFFERENCE(IMPROVED); break;
    default: DIFFERENCE(FAILED); break; // CRASH replaced by HANG?!
    };
    break;
  case Result::HANG:
    switch (new_rslt_type)
    {
    case Result::OK:
    case Result::WARNING:
    case Result::ERROR: DIFFERENCE(WORKED); break;
    // HANG replaced by gracious failure!
    case Result::FAILURE: DIFFERENCE(IMPROVED); break;
    case Result::CRASH: DIFFERENCE(SUSPICIOUS); break; // HANG is now CRASH!
    default: ; // disable warnings
    };
    break;
  case Result::TYPE_NUMBER:
    ; // make compilers happy
  }
  return data_diff;
}

std::string Object::format_checksum(const Result& _rslt, const String& _data)
{
  Base::OStringStream strm;
  strm << _rslt << "   " << name() << ": " << _data << ::Base::ENDL;
  return strm.str;
}

void Object::add(const Result& _rslt, const String& _data, bool _debg_call_stck)
{
  Base::OStringStream strm;

#ifdef DEB_ON
  if (_debg_call_stck)
  {
    static String prev_call_stck;
    String call_stck("/");
    Debug::CallStack::query().append(call_stck);

    if (prev_call_stck != call_stck)
    {
      strm << call_stck << ::Base::ENDL;
      prev_call_stck = call_stck;
    }
  }
#else
  LOW_CODE_QUALITY_VARIABLE_ALLOW(_debg_call_stck);
#endif

  // Write checksum data and remove trailing whitespace
  strm << format_checksum(_rslt, _data);
  rtrim(strm.str);

  Report::write(strm.str);
}

///////////////////////////////////////////////////////////////////////////////
//// class Checksum::Compare implementation
///////////////////////////////////////////////////////////////////////////////

namespace Report
{
// Read and parse lines of a report to get the checksum Result, name and
// content.
class Read
{
public:
  //! An  entry in the report, some stored information is redundant
  class Entry
  {
  public:
    Entry() {}
    explicit Entry(const std::string& _line) : line_(_line)
    {
      // Ignore lines that represent (debug) callstack groups
      if (group())
        return;

      // Remove whitespace and newline characters from the end of the line
      rtrim(line_);

      // The line should be a checksum. Extract the checksum data.
      std::istringstream line_strm(line_);
      line_strm >> std::noskipws >> rcrd_.rslt;
      line_strm >> std::skipws >> name_;
      if (!name_.empty() && name_.back() == ':')
        name_.pop_back();
      std::getline(line_strm, rcrd_.data);

      // Erase any code links of the form "@ [ ..... ]" from the data
      erase_code_links(rcrd_.data);
    }

    bool group() const { return line_[0] == '/'; }

    /*!
    Comparison operator, returns true if it is the same checksum, not the same
    checksum result!
    */
    bool operator==(const Entry& _othr) const
    {
      if (group())
        return line_ == _othr.line_;
      if (name_ != _othr.name_)
        return false;
      if (name_ == Checksum::completion.name())
        return rcrd_.data == _othr.rcrd_.data;

      return true;
    }

    const std::string& line() const { return line_; }
    const std::string& name() const { return name_; }
    const Checksum::Record& record() const { return rcrd_; }

  private:
    std::string line_;      // Current line
    std::string name_;      // Extracted checksum name
    Checksum::Record rcrd_; // Extracted checksum record
  };

  typedef std::vector<Entry> EntryVector;

  Checksum::Level level() const { return lvl_; }

public:
  Read() : lvl_(Checksum::L_ALL) {}
  explicit Read(const fs::path& _rprt_path) : lvl_(Checksum::L_ALL)
  {
#if __APPLE__
    std::ifstream rprt(_rprt_path.string()); // report stream
#else // __APPLE__
    std::ifstream rprt(_rprt_path); // report stream
#endif // __APPLE__
    for (std::string line; std::getline(rprt, line);)
    {
      // Skip empty lines
      if (line.empty())
        continue;

      // Find the checksum run level for this report by looking for the line
      // "Report Level: <LEVEL>"
      const auto rprt_lvl_tag_idx = line.find(REPORT_LEVEL_TAG);
      if (rprt_lvl_tag_idx != std::string::npos)
      {
        // Found such a line. Parse it to extract <LEVEL>.
        for (int i = (int)Checksum::L_NONE; i <= Checksum::L_ALL; ++i)
        {
          if (line.find(Checksum::LEVEL_TEXT[i], rprt_lvl_tag_idx) !=
              std::string::npos)
          {
            lvl_ = (Checksum::Level)i;
            break;
          }
        }
        continue;
      }

      entrs_.push_back(Entry(line));
    }
  }

  size_t size() const { return entrs_.size(); }

  const Entry& operator[](const size_t _i) const { return entrs_[_i]; }

private:
  Checksum::Level lvl_;
  EntryVector entrs_;
};


class Compare::Impl
{
public:
  Impl() : rgstr_(registry()) {}

  // merge another registry into the current one
  void merge() { rgstr_.merge(registry()); }

  //! Set the report pair to compare
  void set_reports(const fs::path& _left_path, const fs::path& _right_path);

  //! Set the short format, i.e., remove identical checksums, true by default.
  void set_short_format(const bool _short_format = true);

  /*!
  Run the comparison and generate a difference report. Throws a std error if the
  report paths are not set.
  */
  DifferenceDistribution run(Base::IOutputStream& _log_os) const;

  /*!
  Run the comparison and throw a std error if both reports exist but the
  checksums are not equal.
  */
  void check_equal() const;

private:
  /*!
  Compares two checksums through use of the checksum objects stored in the
  registry (rgstr_).
  */
  Difference compare_from_registry(const String& _name, const Record& _old_rcrd,
      const Record& _new_rcrd) const;

  Registry& rgstr_; //<! Checksum registry, potentially merged other registers

  fs::path left_path_;
  fs::path right_path_;
  bool short_format_ = true;
};

void Compare::Impl::set_reports(
    const fs::path& _left_path, const fs::path& _right_path)
{
  left_path_ = _left_path;
  right_path_ = _right_path;
}

void Compare::Impl::set_short_format(const bool _short_format)
{
  short_format_ = _short_format;
}

DifferenceDistribution Compare::Impl::run(Base::IOutputStream& _log_os) const
{
  using Base::ENDL;

  // various tags that trigger highlights in a .diff file
  const char* const GROUP_TAG   = "****";
  const char* const SAME_TAG    = "=   ";
  const char* const OLD_TAG     = "<   ";
  const char* const NEW_TAG     = ">   ";
  const char* const REMOVED_TAG = "<<< ";
  const char* const ADDED_TAG   = ">>> ";
  const char* const NAME_TAG    = "?   ";
  const char* const DIFF_TAG    = "!   ";
  const char* const RESULT_TAG  = "  ";

  // Throw an error if the report paths have not been set
  TEST_THROW_ERROR_if(left_path_.empty() || right_path_.empty(),
      TEST_CHECKSUM_COMPARE_REPORTS_NOT_SET);

  const Read rprts[2] = {Read(left_path_), Read(right_path_)};

  LongestCommonSubsequenceT<Read> rprt_lcs(rprts[0], rprts[1]);
  rprt_lcs.trace();
  LongestCommonSubsequenceT<Read>::IndexPairVector rprt_mtch;
  rprt_lcs.match(rprt_mtch);

  DifferenceDistribution test_diff;

  for (const auto& ij : rprt_mtch) // iterate the mis-/matched index pairs
  {
    if (ij.matched()) // matched entries ?
    {                 // compare the checksums, print if anything else
      const auto& old_entr = rprts[0][ij.i];
      const auto& new_entr = rprts[1][ij.j];

      if (old_entr.group()) // this match entry is a group?
      {
        if (!short_format_)
          _log_os << GROUP_TAG << old_entr.line() << ENDL; // just print it
      }
      else // checksum case
      {
        const auto diff = compare_from_registry(
            old_entr.name(), old_entr.record(), new_entr.record());
        if (diff.equal()) // no difference?
        {
          if (!short_format_)
            _log_os << SAME_TAG << old_entr.line() << ENDL; // just print it
        }
        else
        { // print difference
          _log_os << NAME_TAG << old_entr.name() << ": " << diff.type_text()
                  << ENDL;
          _log_os << OLD_TAG << old_entr.record().rslt << RESULT_TAG
                  << old_entr.record().data << ENDL;
          _log_os << NEW_TAG << new_entr.record().rslt << RESULT_TAG
                  << new_entr.record().data << ENDL;
          const char rslt_mark =
              old_entr.record().rslt.type() == new_entr.record().rslt.type()
                  ? ' '
                  : '!';
          _log_os << DIFF_TAG << rslt_mark << RESULT_TAG << diff.description()
                  << ENDL;
        }
        ++test_diff[diff];
      }
    }
    else if (ij.i_valid() && rprts[0].level() <= rprts[1].level())
    { // old entry removed and the old checksum level is equal or stricter
      const auto& old_entr = rprts[0][ij.i];
      _log_os << REMOVED_TAG << old_entr.line() << ENDL; // just print it
      if (!old_entr.group()) // if it is a group change, just ignore it
      {
        if (Checksum::completion.success(old_entr.line()))
          ++test_diff[Difference(Difference::FAILED, old_entr.line())];
        else
          ++test_diff[Difference(Difference::UNKNOWN, "Removed checksum")];
      }
    }
    else if (ij.j_valid() && rprts[0].level() >= rprts[1].level())
    { // new entry added and the new checksum level is stricter or equal
      const auto& new_entr = rprts[1][ij.j];
      _log_os << ADDED_TAG << new_entr.line() << ENDL; // just print it
      if (!new_entr.group()) // if it is a group change, just ignore it
      {
        if (Checksum::completion.success(new_entr.line()))
          ++test_diff[Difference(Difference::WORKED, new_entr.line())];
        else
          ++test_diff[Difference(Difference::UNKNOWN, "Added checksum")];
      }
    }
  }
  test_diff.erase(Difference()); // Remove the entry that states no differences
  return test_diff;
}

void Compare::Impl::check_equal() const
{
  std::cout << "Info: Comparing checksums...";

  // Throw an error if the report paths have not been set
  TEST_THROW_ERROR_if(left_path_.empty() || right_path_.empty(),
      TEST_CHECKSUM_COMPARE_REPORTS_NOT_SET);

  // Skip the comparison if one of these files doesn't exist
  if (!fs::exists(left_path_))
  {
    std::cout << "no baseline report found (expected at "
              << left_path_.generic_string() << ")." << std::endl;
    return;
  }

  if (!fs::exists(right_path_))
  {
    std::cout << "no checksum report found for the current test (expected at "
              << right_path_.generic_string() << ")." << std::endl;
    return;
  }

  // Compare the test report against the baseline report file
  Base::OStringStream log_os;
  auto diffs = run(log_os);

  std::cout << "complete. ";

  if (diffs.empty())
  {
    // No differences found
    std::cout << "No differences found." << std::endl;
  }
  else
  {
    // Differences found: display short diff and throw error
    std::cout << "Differences found!" << std::endl << std::endl;
    std::cout << log_os.str << std::endl;

    std::cout << "Compare " << PathLink(left_path_) << " with "
              << PathLink(right_path_) << "." << std::endl;

    TEST_THROW_ERROR(TEST_CHECKSUM_COMPARE_DIFFERENCES);
  }
}

Difference Compare::Impl::compare_from_registry(
    const String& _name, const Record& _old_rcrd, const Record& _new_rcrd) const
{
  auto reg_it = rgstr_.find(_name);
  return reg_it == rgstr_.end()
             ? Difference(Difference::UNKNOWN, "Checksum not registered")
             : reg_it->second->compare(_old_rcrd, _new_rcrd);
}

Compare::~Compare() {} // need this to delete impl_

void Compare::make() { impl_ = new Impl(); }

void Compare::set(Compare& _othr) 
{ 
  impl_ = _othr.impl_; // assign implementation which points to another registry
  impl_->merge(); // now merge the local registry into the other one
}

void Compare::set_reports(
    const std::string& _left_path, const std::string& _right_path)
{
  impl_->set_reports(fs::path(_left_path), fs::path(_right_path));
}

void Compare::set_short_format(const bool _short_format)
{
  impl_->set_short_format(_short_format);
}

DifferenceDistribution Compare::run(Base::IOutputStream& _log_os) const
{
  return impl_->run(_log_os);
}

void Compare::check_equal() const { impl_->check_equal(); }

Compare compare;

void check_equal() { compare.check_equal(); }

//////////////////////////////////////////////////////////////////////////
// Write Implementation
//////////////////////////////////////////////////////////////////////////

class Write::Impl
{
public:
  Impl()
  {
  }

  void initialize()
  {
    if (opened_)
      return;

    strm_.open(REPORT_FILENAME);
    TEST_THROW_ERROR_if(!strm_.good(), TEST_CHECKSUM_FAILED_OPEN_REPORT);
    opened_ = true;
    strm_ << REPORT_LEVEL_TAG << LEVEL_TEXT[run_lvl] << ::Base::ENDL;
  }

  void operator()(const std::string& _str)
  {
    if (_str.empty())
      return;

    static std::mutex mtx; // synchronize access to the checksum report stream
    std::lock_guard<std::mutex> lock(mtx);

    initialize(); // ensure stream is opened

    strm_ << _str << ::Base::ENDL;
    strm_.flush();
  }

private: 
  std::ofstream strm_;
  bool opened_ = false;
};

Write::~Write() {}

void Write::make() { impl_ = new Impl; }

void Write::set(Write& _othr) { impl_ = _othr.impl_; }

void Write::initialize() { impl_->initialize(); }

void Write::operator()(const std::string& _str) { (*impl_)(_str); }

Write write;

} // namespace Report
} // namespace Checksum
} // namespace Test

#endif // TEST_ON
