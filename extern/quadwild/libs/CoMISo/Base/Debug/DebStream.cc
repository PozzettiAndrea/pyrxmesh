// (C) Copyright 2020 by Autodesk, Inc.

#ifdef DEB_ON

#include "DebOut.hh"
#include "DebFile.hh"
#include "DebUtils.hh"
#include "DebConfig.hh"
#include "Base/Code/CodeLink.hh"
#include "Base/Test/TestChecksumDebugEvent.hh"

namespace Debug
{

void warning(const std::string& _wrng, const Base::CodeLink& _lnk)
{
  TEST(Debug::warning, record(_wrng, _lnk));
  Stream strm(File::modify());
  strm << WARNING << ": " << _wrng << ' ' << _lnk << Base::ENDL;
}

void error(const std::string& _err, const Base::CodeLink& _lnk)
{
  TEST(Debug::error, record(_err, _lnk));
  Stream strm(File::modify());
  strm << ERROR << ": " << _err << ' ' << _lnk << Base::ENDL;
}

//////////////////////////////////////////////////////////////////////////

Enter::Enter(
    const char* const _flnm, const char* const _fnct, int& _nmbr, int& _lvl)
    : flnm_(_flnm), fnct_(_fnct), outs_(0), strm_(File::modify())
{ // TODO: for thread-safety we will need to make the constructor body atomic!
  strm_.file_.enter(this);
  nmbr_ = _nmbr++;

  if (_lvl == INVALID_LEVEL)
    _lvl = Config::query().custom_level(flnm_, fnct_);
  lvl_ = _lvl;

  static int id_cnt = 0;
  id_ = ++id_cnt;
}

Enter::~Enter() { strm_.file_.exit(); }

Stream& Enter::stream()
{
  if (outs_ < 1)
    strm_.file_.start();
  ++outs_;

  return strm_;
}

//////////////////////////////////////////////////////////////////////////

Stream::Stream(const Stream& _othr) : file_(_othr.file_) {}
Stream& Stream::operator=(const Stream&) { return *this; }

Base::IOutputStream& Stream::print(const ptrdiff_t _i)
{
  file_.print(_i);
  return *this;
};

Base::IOutputStream& Stream::print(const size_t _i)
{
  file_.print(_i);
  return *this;
};

Base::IOutputStream& Stream::print(const float _f)
{
  file_.print(_f);
  return *this;
};

Base::IOutputStream& Stream::print(const double _d)
{
  file_.print(_d);
  return *this;
};

Base::IOutputStream& Stream::print(const char* _s)
{
  file_.print(_s);
  return *this;
};

Base::IOutputStream& Stream::print(const char _c)
{
  file_.print(_c);
  return *this;
};

} // namespace Debug

#endif // DEB_ON
