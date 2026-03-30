// (C) Copyright 2019 by Autodesk, Inc.

#ifdef DEB_ON
#include "DebFile.hh"
#include "DebCallStack.hh"

#include "Base/Utils/Environment.hh"
#include "Base/Utils/OStringStream.hh"

#include <string>
#include <fstream>
#include <cstring>

namespace Debug
{

File::~File()
{
  print_footer();
  flush();
  delete bffr_; // if flush failed to delete it for some reason
}

void File::print(const char _c, const bool _cnsl)
{
  if (_cnsl && cnsl_prnt_ != nullptr)
    (*cnsl_prnt_)(_c); // print on the console
  if (no_log())
    return; // no need to modify the buffer if logging is disabled
  if (bffr_ == nullptr)
    bffr_ = new std::string;
  if (line_strt_)
  {
    line_strt_ = false;
    (*bffr_) += ' ';
  }
  bffr_->append(&_c, 1);
  if (_c == ENDL)
    line_strt_ = true;
}

void File::line_break(const bool _cnsl) { print(ENDL, _cnsl); }

void File::print(const std::string& _s)
{
  for (const auto c : _s)
    print(c);
}

void File::print(const char* const _s)
{
  if (_s == nullptr)
    return;
  auto c = _s[0];
  for (int i = 0; c != '\0'; c = _s[++i])
    print(c);
}

void File::print(const size_t _i)
{
  Base::OStringStream strm;
  strm.print(_i);
  print(strm.str);
}

void File::print(const ptrdiff_t _i)
{
  Base::OStringStream strm;
  strm.print(_i);
  print(strm.str);
}

void File::print(float _f)
{
  Base::OStringStream strm;
  strm.print(_f);
  print(strm.str);
}

void File::print(double _d)
{
  Base::OStringStream strm;
  strm.print(_d);
  print(strm.str);
}

// Append current asctime to given string
void File::print_time()
{
  char bffr[256]; // avoid allocations as this is used on shutdown
  System::Environment::time(sizeof(bffr), bffr);
  print(bffr);
}

void File::print_header()
{
  if (log())
  {
    print(flnm_);
    print(" opened ");
  }
  print_time();
  print("[ Build: " __TIME__ " " __DATE__ "] ");
  line_break();
}

void File::print_footer()
{
  line_break();
  if (log())
  {
    print(flnm_);
  }
  print(" Closed: ");
  print_time();
  line_break();
}


void File::flush()
{
  if (no_log())
    return;

  std::fstream file_stream(flnm_, std::fstream::out |
                  (flushed() ? std::fstream::app : std::fstream::trunc));
  if (!file_stream.is_open()) // failed opening the file?
    return;

  if (!flushed())
    print_header();

  file_stream << *bffr_;
  delete bffr_;
  bffr_ = nullptr;

  ++flush_nmbr_;
}


void File::enter(const Enter* const _entr)
{
  if (log()) // log enabled?
    CallStack::modify().push(_entr);// call stacks are printed only in the log
}

void File::start()
{
  if (no_log()) // log disabled?
    return;

  // First DEB_out in this function so output call-stack, and flush.
  if (!line_strt_)
    line_break(false); // make sure we start on a new line with this

  // lead in for the debug output file only
  print('*', false);
  print('>', false);
  CallStack::query().append(*bffr_);

  line_break(false); // make sure we start on a new line after this
  flush();
}

void File::exit()
{
  if (log())
    CallStack::modify().pop();
}

//////////////////////////////////////////////////////////////////////////

File& File::modify()
{
  // TODO: Thread-local storage, each (per thread) file in a separate  folder
  static File glbl_file;
  return glbl_file;
}

const File& File::query() { return modify(); }

} // namespace Debug

#endif // DEB_ON
