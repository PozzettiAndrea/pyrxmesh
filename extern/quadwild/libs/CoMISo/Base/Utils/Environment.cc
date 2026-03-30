// (C) Copyright 2019 by Autodesk, Inc.

#include "Base/Security/Mandatory.hh"
#include "Environment.hh"
#include "Base/Debug/DebOut.hh"

#include <cstdlib>
#include <cstring>
#include <locale.h>
#include <locale>
#include <time.h>

namespace System {
namespace Environment {

bool variable(const char* const _vrbl_name, std::string& _vrbl)
{
#ifdef _MSC_VER
  size_t char_nmbr;
  getenv_s(&char_nmbr, nullptr, 0, _vrbl_name);
  if (char_nmbr == 0)
    return false;
  _vrbl.resize(char_nmbr);
  getenv_s(&char_nmbr, &_vrbl[0], char_nmbr, _vrbl_name);
  _vrbl.resize(char_nmbr - 1); // remove the trailing \0 char
#else
  const char* const vrbl_env = std::getenv(_vrbl_name);
  if (vrbl_env == nullptr)
    return false;
  _vrbl = vrbl_env;
#endif
  return true;
}

//default session locale is "C" (i.e., "." is the decimal point)
LocaleSession::LocaleSession(const char* const _ssn_lcle)
{
  c_lcl_bckp_ = ::setlocale(LC_ALL, nullptr);
  // std::locale::global set the C++ and C locale at the same time
  // only if the locale has a name (as in the following call).
  // See http://www.cplusplus.com/reference/locale/locale/global/
  lcl_bckp_ = std::locale::global(std::locale(_ssn_lcle));
  DEB_only(char* lcle = )::setlocale(LC_ALL, nullptr);
  DEB_error_if(lcle == nullptr,
    "std::locale::global() failed to set " << _ssn_lcle);
  DEB_error_if(lcle != nullptr && strcmp(lcle,  _ssn_lcle) != 0,
    "set_locale() was asked to set " << _ssn_lcle << 
    ", but actually set " << lcle);
}

LocaleSession::~LocaleSession()
{
  std::locale::global(lcl_bckp_);
  // We can not be sure that lcl_bckp_ had a name, so we must reset the
  // C locale explicitly.
  ::setlocale(LC_ALL, c_lcl_bckp_.c_str());
}

void time(const size_t bffr_size, char* const bffr)
{
  time_t rawtime;
  ::time(&rawtime);
  bffr[0] = '\0';

#ifdef _MSC_VER
  struct tm timeinfo;
  if (localtime_s(&timeinfo, &rawtime) != 0)
    return;
  if (asctime_s(bffr, bffr_size, &timeinfo) != 0)
    return;
#else  // _MSC_VER
  // Format: <Abbreviated weekday name> <abbreviated month name> <day of month>
  // <HH>:<MM>:<SS> <Year>
  std::strftime(
      bffr, bffr_size, "%a %b %d %H:%M:%S %Y\n", std::localtime(&rawtime));
#endif // _MSC_VER
}

std::string time() // get the date and time in a string
{                  // not thread-safe, but simplest with standard functions
  char bffr[256];
  time(sizeof(bffr), bffr);
  return std::string(bffr);
}

}//namespace Environment 
}//namespace System
