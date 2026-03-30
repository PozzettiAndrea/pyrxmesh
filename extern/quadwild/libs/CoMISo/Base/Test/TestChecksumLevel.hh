// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_TESTCHECKSUMLEVEL_HH_INCLUDED
#define BASE_TESTCHECKSUMLEVEL_HH_INCLUDED

#ifdef TEST_ON
namespace Test
{
namespace Checksum
{

//! Enumerate the checksum level, higher levels include lower ones
enum Level
{
  L_NONE,   //! No checksums are enabled
  L_STABLE, //! Stable checksums, useful for cross-platform testing
  L_PRIME,  //! Checksums that exist in build configurations w/o DEB_ON
  L_ALL     //! Checksums that exist only in DEB_ON build configurations
};

} // namespace Checksum
} // namespace Test
#endif // TEST_ON

#endif // BASE_TESTCHECKSUMLEVEL_HH_INCLUDED
