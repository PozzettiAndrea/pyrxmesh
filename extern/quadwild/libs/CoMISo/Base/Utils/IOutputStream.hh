// (C) Copyright 2019 by Autodesk, Inc.

#ifndef BASE_IOUTPUTSTREAM_HH_INCLUDE
#define BASE_IOUTPUTSTREAM_HH_INCLUDE

#include <string>
#include <vector>
#include <cstddef>    // for ptrdiff_t
#include <stdint.h>
#include <cinttypes>
#include <limits.h>
// Find out if we need separate streaming operators for uint and size_t
#if ((UINT_MAX) != (SIZE_MAX))
#define UINT_SIZET_DIFFER
// ptrdiff_t differs from int iff size_t differs from unsigned int
#define INT_PTRDIFFT_DIFFER
#endif

#include <Base/Config/BaseDefines.hh>

#ifdef STD_ARRAY_AVAILABLE
#include <array>
#endif//STD_ARRAY_AVAILABLE

namespace Base {

typedef unsigned int uint;

// use Base::ENDL to stream endline to IOutputStream (same as std::endl)
const char ENDL = '\n';

class BASEDLLEXPORT IOutputStream
{
public:
  virtual ~IOutputStream() {}
  virtual IOutputStream& print(const char) = 0;
  virtual IOutputStream& print(const ptrdiff_t) = 0;
  virtual IOutputStream& print(const size_t) = 0;
  virtual IOutputStream& print(const float) = 0;
  virtual IOutputStream& print(const double) = 0;
  virtual IOutputStream& print(const char* const) = 0;

  /*! Print an array of ElementT */
  template <typename IteratorT>
  inline IOutputStream& print(const IteratorT _bgn_it, const IteratorT _end_it)
  {
    print("[ ");
    for (auto it = _bgn_it; it != _end_it; ++it)
    {
      *this << *it;
      print(' ');
    }
    print(']');
    return *this;
  }

  template <typename ElementT>
  inline IOutputStream& print(const size_t _nmbr, const ElementT* const _elems)
  {
    return print(_elems, _elems + _nmbr);
  }
};

BASEDLLEXPORT IOutputStream& operator<<(IOutputStream& _os, const ptrdiff_t _i);

BASEDLLEXPORT IOutputStream& operator<<(IOutputStream& _os, const double _d);

BASEDLLEXPORT IOutputStream& operator<<(IOutputStream& _os, const char* const _s);

BASEDLLEXPORT IOutputStream& operator<<(IOutputStream& _os, const char _c);

BASEDLLEXPORT IOutputStream& operator<<(IOutputStream& _os, const size_t _i);

#ifdef UINT_SIZET_DIFFER
BASEDLLEXPORT IOutputStream& operator<<(IOutputStream& _os, const unsigned int _i);
#endif//UINT_SIZET_DIFFER

#ifdef INT_PTRDIFFT_DIFFER
BASEDLLEXPORT IOutputStream& operator<<(IOutputStream& _os, const int _i);
#endif//INT_PTRDIFFT_DIFFER

BASEDLLEXPORT IOutputStream& operator<<(IOutputStream& _os, const float _f);

BASEDLLEXPORT IOutputStream& operator<<(IOutputStream& _os, const std::string& _s);

//! IStream operator for std::vector<>
#define BASE_STREAM_CONTAINER(CONTAINER) \
  template <typename ElementT> \
  Base::IOutputStream& operator<<( \
      Base::IOutputStream& _os, const CONTAINER<ElementT>& _cntr) \
  { \
    return _os.print(_cntr.begin(), _cntr.end()); \
  }

BASE_STREAM_CONTAINER(std::vector)

#ifdef STD_ARRAY_AVAILABLE
//! IStream operator for std::array<>
template <typename ElementT, size_t _nmbr>
IOutputStream& operator<<(IOutputStream& _os,
  const std::array<ElementT, _nmbr>& _vec)
{
  return _os.print(_nmbr, &_vec[0]);
}

//! Partial specialization for an empty std::array<>
template <typename ElementT>
IOutputStream& operator<<(IOutputStream& _os,
  const std::array<ElementT, 0>& /*_vec*/)
{
  return _os.print(0, (ElementT*)nullptr);
}

#endif// STD_ARRAY_AVAILABLE

// IStream operator for fixed size arrays
template <typename ElementT, size_t _nmbr>
IOutputStream& operator<<(IOutputStream& _os, const ElementT(&_vec)[_nmbr])
{
  return _os.print(_nmbr, &_vec[0]);
}

// IStream std::pair<>
template <typename T0, typename T1>
IOutputStream& operator<<(IOutputStream& _os, const std::pair<T0, T1>& _pair)
{
  _os << "(" << _pair.first << ", " << _pair.second << ")";
  return _os;
}

template <class StreamT>
class OutputStreamAdaptT : public IOutputStream
{
public:
#if 0
  // This can pass any variable number of arguments, not available in VC2012
  template <typename... ArgsT> OutputStreamAdaptT(const ArgsT&... _args)
    : strm_(_args...) {}

#else
  OutputStreamAdaptT() {}

//  //TODO: get rid of move constructor, cannot submit C++11 code in Base for now
//  OutputStreamAdaptT(OutputStreamAdaptT&& _oth) : strm_(std::move(_oth.strm_))
//  {}

  template <typename ArgT>
  OutputStreamAdaptT(const ArgT& _arg) : strm_(_arg) {}
#endif

  IOutputStream& print(const char _c) override
  {
    strm_ << _c;
    return *this;
  }

  IOutputStream& print(const ptrdiff_t _i) override
  {
    strm_ << _i;
    return *this;
  }

  IOutputStream& print(const size_t _i) override
  {
    strm_ << _i;
    return *this;
  }

  IOutputStream& print(const float _f) override
  {
    strm_ << _f;
    return *this;
  }

  IOutputStream& print(const double _d) override
  {
    strm_ << _d;
    return *this;
  }

  IOutputStream& print(const char* const _str) override
  {
    strm_ << _str;
    return *this;
  }

  StreamT& stream() { return strm_; }
  const StreamT& stream() const { return strm_; }

protected:
  StreamT strm_;

private:
  OutputStreamAdaptT(const OutputStreamAdaptT& _oth);
  OutputStreamAdaptT& operator=(const OutputStreamAdaptT&);
};

/*!
Old Style secure print function, should be used directly only in exceptional
cases. Note: Defined in OStringStream.cc.
*/
extern int print(char* _bffr, const size_t _bffr_size, const char* const _frmt,
  ...);

//! Format a variable for streaming
template <const size_t _bffr_size = 128>
struct FormatT
{
public:
  typedef FormatT<_bffr_size> Self;

  template <typename T>
  FormatT(const char* const _frmt, const T& _vrbl)
  {
    print(bffr_, _bffr_size, _frmt, _vrbl);
  }


#if __cplusplus >= 201103L || _MSC_VER >= 1900 //C++11 support

  // Variadic template constructor
  template <typename... ArgT>
  FormatT(const char* const _frmt, const ArgT&... _args)
  {
    print(bffr_, _bffr_size, _frmt, _args...);
  }

#endif//C++11 support

  friend IOutputStream& operator<<(IOutputStream& _os, const Self& _frmt)
  {
    return _os << _frmt.bffr_;
  }

  const char* buffer() const { return bffr_; }

private:
  char bffr_[_bffr_size];
};

//! Convenient access to format a variable for streaming
template <typename T>
inline FormatT<> format(const char* const _frmt, const T& _vrbl)
{
  return FormatT<>(_frmt, _vrbl);
}

template <typename T> 
inline FormatT<> format_percent(const T& _frct)
{
  return format("%.2f%%", 100. * _frct);
}

namespace FormatHex {

template <int byte_nmbr> struct CastT;
template <> struct CastT<4> { typedef uint32_t Type; };
template <> struct CastT<8> { typedef uint64_t Type; };
const char* const PREFIX_VOID = "";
const char* const PREFIX_0x = "0x";
const char* const PREFIX_0X = "0X";

}//FormatHex

//! Format an unsigned int variable for streaming in hex (e.g. for hash)
template <typename T> //!< 32 or 64 bit type
inline FormatT<> format_hex(
    const T _vrbl, const char* const _prfx = FormatHex::PREFIX_VOID)
{
  return format_hex(typename FormatHex::CastT<sizeof(T)>::Type(_vrbl), _prfx);
}

//! Format a 32-bit unsigned int variable for streaming in hex (e.g. for hash)
template <>
inline FormatT<> format_hex(const uint32_t _vrbl, const char* const _prfx)
{
  return FormatT<>("%s%" PRIx32, _prfx, _vrbl);
}

//! Format a 64-bit unsigned int variable for streaming in hex (e.g. for hash)
template <>
inline FormatT<> format_hex(const uint64_t _vrbl, const char* const _prfx)
{
  return FormatT<>("%s%" PRIx64, _prfx, _vrbl);
}

}//namespace Base
#endif//BASE_IOUTPUTSTREAM_HH_INCLUDE
