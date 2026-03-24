// (C) Copyright 2019 by Autodesk, Inc.

#ifndef BASE_OSTRINGSTREAM_HH_INCLUDED
#define BASE_OSTRINGSTREAM_HH_INCLUDED

#include <Base/Utils/IOutputStream.hh>

namespace Base {

class BASEDLLEXPORT OStringStream : public IOutputStream
{
public:
  IOutputStream& print(const char) override;
  IOutputStream& print(const ptrdiff_t) override;
  IOutputStream& print(const size_t) override;
  IOutputStream& print(const float) override;
  IOutputStream& print(const double) override;
  IOutputStream& print(const char* const) override;

  void clear() { str.clear(); }

  std::string str;
};

}//namespace Base

#endif//BASE_OSTRINGSTREAM_HH_INCLUDED
