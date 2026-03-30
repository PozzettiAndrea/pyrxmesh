// (C) Copyright 2020 by Autodesk, Inc.

#ifndef BASE_REDIRECTSTREAM_HH_INCLUDED
#define BASE_REDIRECTSTREAM_HH_INCLUDED

#include <stdio.h>

namespace System
{

/*!
Redirect up to 2 existing opened streams (usually stderr and/or stdout) in a new
file with the provided filename
*/
class RedirectStream
{
public:
  // Default constructor
  RedirectStream() {}

  // Constructor, sets the name of the redirect stream
  RedirectStream(const char* const _flnm) : flnm_(_flnm) {}

  // Destructor, closes the redirect stream
  ~RedirectStream()
  {
    if (to_stream_ != nullptr)
      fclose(to_stream_);
  }

  //! Set the stream filename (if not set already)
  void set_filename(const char* const _flnm)
  {
    if (!redirected()) // redirected already?
      flnm_ = _flnm;
  }

  //! Redirect _from_stream to a newly created stream with the filename
  bool redirect(FILE* _from_stream);

  //! Check if a stream has been already redirected
  bool redirected() const { return from_stream_[0] != nullptr; }

  //! Duplicate another stream to the redirect stream , call that redirect()
  bool duplicate(FILE* _from_stream);

  //! Check if a stream has been already duplicated
  bool duplicated() const { return from_stream_[1] != nullptr; }

protected:
  const char* flnm_ = nullptr;
  FILE* to_stream_ = nullptr;
  FILE* from_stream_[2] = { nullptr, nullptr };
};

} // namespace System

namespace Test
{
const char* const LOG_FILENAME = "out.txt";
} // namespace Test

#endif // BASE_REDIRECTSTREAM_HH_INCLUDED
