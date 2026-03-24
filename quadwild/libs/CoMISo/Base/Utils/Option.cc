// (C) Copyright 2021 by Autodesk, Inc.

#include "Base/Security/Mandatory.hh"
#include "OptionT.hh"

#include <string>
#include <stddef.h>

#ifdef TEST_ON

#include <Base/Test/TestArgs.hh>

#include <memory>
#include <functional>
#include <type_traits>

namespace Base
{

template <>
void OptionT<bool>::bind(const char* const _name)
{
  const auto post_parse = [this](const Test::Args::ToggleGroup& _tggl_grp)
  {
    if (_tggl_grp.parsed())
      value = _tggl_grp.on();
  };

  new Test::Args::BoundToggleGroup(_name, post_parse);
}

template <> 
void OptionT<std::string>::bind(const char* const _name)
{
  const auto post_parse = [this](const Test::Args::StringOption& _opt)
  {
    if (_opt.parsed())
      value = _opt.str;
  };
  new Test::Args::BoundOption(_name, value, post_parse);
}

namespace
{

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type string_to_value(
    const std::string& _str)
{
  return static_cast<T>(std::atoi(_str.c_str()));
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, std::string>::type
value_to_string(const T _val)
{
  return std::to_string(_val);
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
string_to_value(const std::string& _str)
{
  return static_cast<T>(std::atof(_str.c_str()));
}

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, std::string>::type
value_to_string(const T _val)
{
  OStringStream strm;
  strm << _val;
  return strm.str;
}

} // namespace

template <typename T>
void OptionT<T>::bind(const char* const _name)
{
  const auto post_parse = [this](const Test::Args::StringOption& _opt)
  {
    if (_opt.parsed())
      value = string_to_value<T>(_opt.str);
  };
  new Test::Args::BoundOption(_name, value_to_string(value), post_parse);
}

} // namespace Base

#endif // TEST_ON

namespace Base
{
template class OptionT<bool>;
template class OptionT<int>;
template class OptionT<size_t>;
template class OptionT<float>;
template class OptionT<double>;
template class OptionT<std::string>;
}
