// (C) Copyright 2019 by Autodesk, Inc.

#ifndef BASE_DEBUTILS_HH_INCLUDED
#define BASE_DEBUTILS_HH_INCLUDED
#ifdef DEB_ON

#include <Base/Debug/DebFile.hh>

namespace Debug {

class DoubleFormatSession
{
public:
  DoubleFormatSession(const char* const _frmt) 
    : frmt_bck_(File::query().double_format)
  {
    File::modify().double_format = _frmt;
  }

  ~DoubleFormatSession()
  {
    File::modify().double_format = frmt_bck_;
  }

private:
  const char* const frmt_bck_;
};

}// namespace Debug

#define DEB_double_format(FF) Debug::DoubleFormatSession double_format(FF);

#else

#define DEB_double_format(FF)

#endif//DEB_ON
#endif // BASE_DEBUTILS_HH_INCLUDED
