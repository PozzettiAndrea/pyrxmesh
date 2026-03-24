// (C) Copyright 2022 by Autodesk, Inc.

#ifndef BASE_TESTCHECKSUMREPORT_HH_INCLUDED
#define BASE_TESTCHECKSUMREPORT_HH_INCLUDED

#ifdef TEST_ON

#include <Base/Test/TestChecksum.hh>
#include <Base/Debug/DebOut.hh>
#include <Base/Utils/NullOutputStream.hh>
#include <Base/Utils/SafeSharedPtrT.hh>

#include <map>
#include <string>

namespace Test
{
namespace Checksum
{
namespace Report
{

typedef std::map<Difference, size_t> DifferenceDistribution;

/*!
Compare two test checksum reports.

This class should be instantiated uniquely in a specific way to ensure it is
shared across different instances of Base in separate binary modules in the same
process.

As the checksums reside in the binary that implements the tested algorithms, the
class instance should be created there with \ref make(). The instance is owned
by that binary and must be deleted by it, as it's stored on its heap.

Once created, the instance should be \ref set() in the test binary (usually an
executable). \ref set() will not take ownership of the pointer. With this
approach all class methods will return the same instance throughout the process.

\note These considerations are irrelevant if the tested and test executable are
linked statically, i.e., they share the same binary.
*/
class Compare
{
public:
  //! Destroy the owned instance
  ~Compare();

  //! Make and store the unique instance
  void make();

  //! Set an instance made in another binary
  void set(Compare&);

  /*!
  Set the report pair to compare. Once all builds are on C++17, std::string can
  be replaced by std::filesystem, and the calling code will be simplified.
  */
  void set_reports(
      const std::string& _left_path, const std::string& _right_path);

  //! Set the short format, i.e., remove identical checksums, true by default.
  void set_short_format(const bool _short_format = true);

  /*!
  Run the comparison and generate a difference report. Throws a std error if the
  report paths are not set.
  */
  DifferenceDistribution run(
      Base::IOutputStream& _log_os = Base::null_os) const;

  /*!
  Run the comparison and throw a std error if both reports exist but the
  checksums are not equal.
  */
  void check_equal() const;

private:
  class Impl;
  Base::SafeSharedPtrT<Impl> impl_; // shared implementation 
};

/*!
Shared instance per binary, make sure to call make() and set() as appropriate.
*/
extern Compare compare;

// Standalone call to compare.check_equal() suitable for forward declaration
void check_equal();

//! A singleton write access to the test report
class Write
{
public:
  //! Destroy the owned instance
  ~Write();

  //! Make and store the unique instance
  void make();

  //! Set an instance made in another binary
  void set(Write&);

  //! Open the report file for writing (in the current working directory)
  void initialize();

  // Write _str to the report file
  void operator()(const std::string& _str); 

private:
  class Impl;
  Base::SafeSharedPtrT<Impl> impl_; // shared implementation 
};

extern Write write;


} // namespace Report
} // namespace Checksum
} // namespace Test

#endif // TEST_ON
#endif // BASE_TESTCHECKSUMREPORT_HH_INCLUDED
