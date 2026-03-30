// (C) Copyright 2022 by Autodesk, Inc.

#ifndef BASE_TESTCHECKSUM_HH_INCLUDED
#define BASE_TESTCHECKSUM_HH_INCLUDED
#ifndef TEST_ON

#define TEST(CHKSM, RCRD)
#define TEST_if(CNDT, CHKSM, RCRD)

#else

#include <Base/Test/TestResult.hh>
#include <Base/Test/TestChecksumLevel.hh>
#include <Base/Utils/OStringStream.hh>

namespace Test
{

const char* const REPORT_FILENAME = "report.txt";
const char* const REPORT_LEVEL_TAG = "Report Level: ";

namespace Checksum
{
extern Level run_lvl; //<! The global checksum run level
const char* const LEVEL_TEXT[4] = {"NONE", "STABLE", "PRIME", "ALL"};

//! typedef String, this is used a lot in this namespace
typedef std::string String;

//! The checksum record
struct Record
{
  Record() {}
  Record(const Result& _rslt, const String _data) : rslt(_rslt), data(_data) {}

  Result rslt; //! record result
  String data; //!< recorded "data" (as string, can be parsed)
};

//! The difference found by the IChecksum::compare() operation
class Difference
{
public:
  enum Type
  {
    EQUAL,      // result is bitwise identical
    NEGLIGIBLE, // result is negligibly different
    UNKNOWN,    // non-negligible difference, but of unknown quality
    IMPROVED,   // result is better
    SUSPICIOUS, // result is different, and the new result might be worse
    REGRESSED,  // result is worse
    WORKED,     // result works now, but used to fail
    FAILED      // result fails now, but used to work
  };

  static const char* type_text(const Type _type)
  {
    static const char dscr[][32] = {"EQUAL", "NEGLIGIBLE", "UNKNOWN",
        "IMPROVED", "SUSPICIOUS", "REGRESSED", "WORKED", "FAILED"};
    return dscr[_type];
  }

  Difference(const Type _type = EQUAL, const String& _dscr = String())
      : type_(_type), dscr_(_dscr)
  {
  }

  Type type() const { return type_; }
  bool equal() const { return type() == EQUAL; }

  bool operator==(const Difference& _othr) const
  {
    return type_ == _othr.type_ && dscr_ == _othr.dscr_;
  }

  Difference& operator+=(const Difference& _othr)
  {
    if (type_ < _othr.type_)
      type_ = _othr.type_;
    if (dscr_.empty())
      dscr_ = _othr.dscr_;
    else if (!_othr.dscr_.empty())
      dscr_ += "; " + _othr.dscr_;
    return *this;
  }

  Difference& operator+=(const Difference::Type& _type)
  {
    if (type_ < _type)
      type_ = _type;
    return *this;
  }

  const String& description() const { return dscr_; }

  const char* type_text() const { return type_text(type_); }

  friend Base::IOutputStream& operator<<(
      Base::IOutputStream& _os, Difference& _diff)
  {
    // TODO: use string description array
    return _os << _diff.type_text() << " " << _diff.dscr_;
  }

  bool operator<(const Difference& _othr) const { return type_ < _othr.type_; }

private:
  Type type_;
  String dscr_;
};

/*!
Base class for test checksums. Whatever check we want to add in the test system,
it must be an instance of a class derived from Checksum. All derived classes
must be instantiated as global variables.
*/
class Object
{
public:
  //! Checksum name.
  const char* name() const { return name_; }

  //! Add a record the checksum (generic version)
  template <typename T> void record(const Result& _rslt, const T& _data)
  {
    Base::OStringStream strm;
    strm << _data;
    add(_rslt, strm.str);
  }

  //! Add a record of the checksum (public version)
  void record(const Result& _rslt, const String& _data) { add(_rslt, _data); }

  /*!
  Compare two existing records (old and new).
  Returns a qualification and a description of the difference.
  The default implementation has some intelligence in comparing the record
  results, but compares the the data simply as strings (no parsing).
  */
  virtual Difference compare(
      const Record& _old_rcrd, //!<[in] "Left" record
      const Record& _new_rcrd  //!<[in] "Right" record
  ) const;

  //! Get if the checksum should be run
  bool allow() const { return lvl_ <= run_lvl; }

  /*!
  Create a formatted string containing the name of the checksum object and the
  checksum (a Result and an associated piece of data).
  */
  std::string format_checksum(const Result& _rslt, const String& _data);

protected:
  /*!
  Performs an automatic registration of the new checksum in a
  global list, and verifies that the name is unique.
  */
  Object(const char* const _name, const Level _lvl = L_ALL);

  /*!
  Add a record of the checksum (protected version).
  \param _debg_call_stck Also output the current call stack location (only in
         Debug mode)
  */
  void add(const Result& _rslt, const String& _data, bool _debg_call_stck = true);

  /*!
  Compare the data, the default implementation does a string comparison.
  This is called by \ref compare to compare the data only without taking into
  account the type of the result. This can be overridden instead of \ref compare
  if the result type is insignificant for the comparison.
  */
  virtual Difference compare_data(const String& _old, const String& _new) const;

protected:
  const Level lvl_; // the checksum level

private:
  const char* const name_;

private:
  Object(const Object&);
  Object* operator=(const Object&);
};

} // namespace Checksum
} // namespace Test

#define TEST(CHKSM_OBJ, RCRD_FN) \
  { \
    if (::Test::Checksum::CHKSM_OBJ.allow()) \
    { \
      ::Test::Checksum::CHKSM_OBJ.RCRD_FN; \
    } \
  }

#define TEST_if(CNDT, CHKSM_OBJ, RCRD_FN) \
  { \
    if (CNDT) \
      TEST(CHKSM_OBJ, RCRD_FN) \
  }

#endif // TEST_ON
#endif // BASE_TESTCHECKSUM_HH_INCLUDED
