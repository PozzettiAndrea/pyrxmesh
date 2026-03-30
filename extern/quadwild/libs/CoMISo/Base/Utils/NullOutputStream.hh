// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_NULLOUTPUTSTREAM_HH_INCLUDED
#define BASE_NULLOUTPUTSTREAM_HH_INCLUDED

#include <Base/Utils/IOutputStream.hh>

namespace Base
{

// An output stream that ignores data streamed to it
class BASEDLLEXPORT NullOutputStream : public IOutputStream
{
public:
  IOutputStream& print(const char) override { return *this; }
  IOutputStream& print(const ptrdiff_t) override { return *this; }
  IOutputStream& print(const size_t) override { return *this; }
  IOutputStream& print(const float) override { return *this; }
  IOutputStream& print(const double) override { return *this; }
  IOutputStream& print(const char* const) override { return *this; }
};

extern NullOutputStream null_os;

} // namespace Base

#endif // BASE_NULLOUTPUTSTREAM_HH_INCLUDED
