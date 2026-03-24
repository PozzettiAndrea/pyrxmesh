// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_FILESYSTEM_HH_INCLUDE
#define BASE_FILESYSTEM_HH_INCLUDE

#include <Base/Security/Mandatory.hh>

#ifdef __APPLE__
#include <AvailabilityMacros.h>
#if __MAC_OS_X_VERSION_MIN_REQUIRED < 101500
#define FILESYSTEM_USE_BOOST
#endif // #if __MAC_OS_X_VERSION_MIN_REQUIRED < 101500
#endif // #ifdef __APPLE__
// NOTE: Boost required for old versions of MacOS due to lack of std::filesystem
#ifdef FILESYSTEM_USE_BOOST

#include <boost/filesystem.hpp>

namespace Base
{
namespace filesystem = boost::filesystem;
typedef boost::filesystem::path Path;
typedef boost::system::error_code error_code;
} // namespace Base

#else // #ifdef FILESYSTEM_USE_BOOST

// Include <filesystem> but silence resulting C4995 warnings
INSECURE_INCLUDE_SECTION_BEGIN
#include <filesystem>
INSECURE_INCLUDE_SECTION_END

namespace Base
{
namespace filesystem = std::filesystem;
typedef std::error_code error_code;
} // namespace Base

#endif // #ifdef FILESYSTEM_USE_BOOST

#endif // BASE_FILESYSTEM_HH_INCLUDE
