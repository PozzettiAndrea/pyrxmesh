// (C) Copyright 2019 by Autodesk, Inc.

#ifndef BASE_DEBFILE_HH_INCLUDED
#define BASE_DEBFILE_HH_INCLUDED
#ifdef DEB_ON

#include <Base/Utils/IOutputStream.hh>
#include <Base/Debug/DebConfig.hh>

#include <string>

namespace Debug {
typedef unsigned int uint;
using Base::ENDL;
class Enter;

//! Debug file.
class File
{
public:
  static const File& query();
  static File& modify();

public:
  const char* double_format = "%.17g";

  bool no_log() const { return flnm_ == nullptr; }
  bool log() const { return !no_log(); }

  void enter(const Enter* const _entr);
  void start();
  void exit();

  void print(const char _c) { print(_c, true); }
  void print(const char* const _s);
  void print(const std::string& _s);
  void print(const size_t _i);
  void print(const ptrdiff_t _i);
  void print(float _f);
  void print(double _d);

private:
  int flush_nmbr_ = 0;
  bool line_strt_ = true; // are we at the start of the line?
  Config::print_function cnsl_prnt_ = Config::query().console_print;
  const char* const flnm_ = Config::query().log_filename;
  std::string* bffr_ = nullptr;

private:
  ~File();

  bool flushed() const { return flush_nmbr_ > 0; }

  void print_header();
  void print_footer();
  void print_time();

  void print(const char _c, const bool _cnsl);
  void line_break(const bool _cnsl = true);
  void flush();
};

}//namespace Debug

#endif//DEB_ON
#endif//BASE_DEBFILE_HH_INCLUDED
