// (C) Copyright 2019 by Autodesk, Inc.

#ifdef DEB_ON

#include "DebCallStack.hh"
#include "DebOut.hh"
#include <cstring>

namespace Debug {
namespace {  

// Replace interior of < > in function name with . for brevity
void append_function(std::string& _str, const char* const _func)
{
  int cnt = 0;
  const char* ptr = _func;
  while (ptr && (*ptr != '\0'))
  {
    char c = *ptr;
    if (c == '>') --cnt;
    if (cnt == 0) _str.append(1, c);
    if (c == '<')
    {
      if (cnt == 0) _str.append(".");
      ++cnt;
    }
    ++ptr;
  }
}

// Add to the string the call stack element string
void append_entry(std::string& _str, const Enter* const _entr, 
  const bool _cmpct, const bool _entr_nmbr, const Enter* const _prev_entr)
{
  if (_prev_entr != nullptr && // there is a previous entry and it is the same?
    strcmp(_prev_entr->function(), _entr->function()) == 0)
  {// ... so do nothing
    return;
  }

  if (_prev_entr != nullptr)
    _str.append("/");

  if (_cmpct)
    append_function(_str, _entr->function());
  else
    _str.append(_entr->function());

  if (_entr_nmbr)
  {
    _str += '.';
    _str += std::to_string(_entr->number());
  }
}

}//namespace


CallStack& CallStack::modify()
{
  static CallStack glbl_call_stck;
  return glbl_call_stck;
}

const CallStack& CallStack::query()
{
  return modify();
}

void CallStack::append(std::string& _str, const bool _entr_nmbr) const
{
  const Enter* prev = nullptr;
  for (size_t i = 0, n = depth(); i < n; prev = calls_[i++])
    append_entry(_str, calls_[i], true, _entr_nmbr, prev);
}

void CallStack::append_indent(std::string& _str, const int _indt) const
{
  if (_indt == 0) 
    return;
  int num = (int)calls_.size();
  int i0 = 1;
  for (int i = i0; i < num; ++i)
    _str.append("  ");
}

}//namespace Debug

#endif//DEB_ON
