// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_CHECKSUMCOMPLETION_HH_INCLUDE
#define BASE_CHECKSUMCOMPLETION_HH_INCLUDE

#ifdef TEST_ON

#include <Base/Security/Mandatory.hh>
#include <Base/Test/TestChecksum.hh>

namespace Test
{
namespace Checksum
{

// Writes a checksum when the test completes.
class Completion : public Object
{
public:
  Completion();

  //! Record the test "success" completion checksum
  void record_success();

  /*!
  Record a test failure checksum
  \param _msg The reason for the failure, e.g. an exception
  */
  void record_failure(const std::string& _msg);

  //! Get if the line contains the "success" completion checksum record
  static bool success(const std::string& _line);

  //! Get if the line contains the "failure" completion checksum record
  static bool failure(const std::string& _line);
};

// Register the checksum to check test completion.
extern Completion completion;

} // namespace Checksum
} // namespace Test

#endif // TEST_ON
#endif // BASE_CHECKSUMCOMPLETION_HH_INCLUDE
