// (C) Copyright 2022 by Autodesk, Inc.

#include "Base/Security/Mandatory.hh"
#include "OStringStream.hh"

#include <stdarg.h>
#include <sstream>
#include <iomanip>

#ifndef _MSC_VER
#include <cstdio>
#include <cstdarg>
#endif

#ifndef DBL_DECIMAL_DIG
#define DBL_DECIMAL_DIG 17
#endif // DBL_DECIMAL_DIG

#ifndef FLT_DECIMAL_DIG
#define FLT_DECIMAL_DIG 9
#endif // FLT_DECIMAL_DIG

namespace Base
{

namespace
{

// Cross-platform approach to avoid buffer overflow when printing. However, 
// there are subtle differences between these functions, see here for details:
// https://stackoverflow.com/questions/46485639/what-is-the-difference-between-vsnprintf-and-vsprintf-s

#ifdef _MSC_VER
  #define VSNPRINTF vsprintf_s
#else
  #define VSNPRINTF vsnprintf
#endif

template <size_t _bffr_size>
int sprintf_s(char (&_bffr)[_bffr_size], const char* _frmt, ...)
{
  va_list arg_ptr;
  va_start(arg_ptr, _frmt);
  int res = VSNPRINTF(_bffr, _bffr_size, _frmt, arg_ptr);
  va_end(arg_ptr);
  return res;
}

} // namespace

int print(char* _bffr, const size_t _bffr_size, const char* _frmt, ...)
{
  va_list arg_ptr;
  va_start(arg_ptr, _frmt);
  int res = VSNPRINTF(_bffr, _bffr_size, _frmt, arg_ptr);
  va_end(arg_ptr);
  return res;
}

struct CLocaleOStringStream : public std::ostringstream
{
  CLocaleOStringStream() { this->imbue(std::locale("C")); }

  void set_precision_double() { *this << std::setprecision(DBL_DECIMAL_DIG); }

  void set_precision_float() { *this << std::setprecision(FLT_DECIMAL_DIG); }
};

IOutputStream& OStringStream::print(const char _c)
{
  str.append(1, _c);
  return *this;
}

IOutputStream& OStringStream::print(const ptrdiff_t _i)
{
  CLocaleOStringStream strm;
  strm << _i;
  str.append(strm.str());
  return *this;
}

IOutputStream& OStringStream::print(const size_t _i)
{
  CLocaleOStringStream strm;
  strm << _i;
  str.append(strm.str());
  return *this;
}

IOutputStream& OStringStream::print(const float _f)
{
  CLocaleOStringStream strm;
  strm.set_precision_float();
  strm << _f;
  str.append(strm.str());
  return *this;
}

IOutputStream& OStringStream::print(const double _d)
{
  CLocaleOStringStream strm;
  strm.set_precision_double();
  strm << _d;
  str.append(strm.str());
  return *this;
}

IOutputStream& OStringStream::print(const char* const _str)
{
  str.append(_str);
  return *this;
}

// template <const char _frmt, const int _bffr_size = 128>
// IOutputStream& format(IOutputStream& _os, const double _d)
//{
//  char bffr[_bffr_size];
//  sprintf_s(bffr, _frmt, _d);
//  return std::string(bffr);
//}

} // namespace Base
