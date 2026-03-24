// (C) Copyright 2019 by Autodesk, Inc.

#include "Base/Security/Mandatory.hh"
#include "RedirectStream.hh"

#ifdef _WIN32
#include <io.h>
#define dup2 _dup2
#define fileno _fileno
#else // _WIN32
#include <unistd.h>
#endif // _WIN32

namespace System
{

bool RedirectStream::redirect(FILE* _from_stream)
{
  if (_from_stream == nullptr || flnm_ == nullptr || redirected())
    return false;

#ifdef _WIN32
  if (freopen_s(&to_stream_, flnm_, "w", _from_stream) != 0)
    to_stream_ = nullptr;
#else
  to_stream_ = freopen(flnm_, "w", _from_stream);
#endif
  if (to_stream_ == nullptr) // failed?
    return false;

  from_stream_[0] = _from_stream;
  return true;
}

bool RedirectStream::duplicate(FILE* _from_stream)
{
  // check we are in the right state to duplicate
  if (_from_stream == nullptr || !redirected() || duplicated())
    return false;

  if (dup2(fileno(from_stream_[0]), fileno(_from_stream)) != 0)
    return false; // failed to duplicate

  from_stream_[1] = _from_stream;
  return true;
}

} // namespace System
