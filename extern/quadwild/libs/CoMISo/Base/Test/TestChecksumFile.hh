// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_CHECKSUMFILE_HH_INCLUDE
#define BASE_CHECKSUMFILE_HH_INCLUDE

#ifdef TEST_ON

#include <Base/Test/TestChecksum.hh>

namespace Test
{
namespace Checksum
{

/*!
checksum for output files. It has a method record that add a file hash.
*/
class File : public Object
{
public:
  File(const char* const _name) : Object(_name, L_PRIME) {}

  void record(const char* const _flnm);
};

} // namespace Checksum
} // namespace Test

#define TEST_CHECKSUM_FILE(VRBL, NAME) File VRBL(NAME "-file")

#else // TEST_ON

#define TEST_CHECKSUM_FILE(VRBL, NAME)

#endif // TEST_ON

#endif // BASE_CHECKSUMFILE_HH_INCLUDE
