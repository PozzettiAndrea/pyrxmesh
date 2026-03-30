// (C) Copyright 2020 by Autodesk, Inc.

#ifndef BASE_DEBCONFIG_HH_INCLUDED
#define BASE_DEBCONFIG_HH_INCLUDED
#ifdef DEB_ON

#include <Base/Config/BaseDefines.hh>
#include <string>

namespace Debug
{

void print_char_to_cerr(const char _c); //!< print a char to cerr

/*!
Access the global, per-process, configuration options of the Debug system.
\todo Make this a per-thread configuration.
*/
class BASEDLLEXPORT Config
{
public:
  //! Define the function type to print a character on the console
  typedef void (*print_function)(const char);

public:
  //! Modify the current configuration.
  static Config& modify();

  //! Query the current configuration.
  static const Config& query();

  //! Query the default configuration.
  static const Config& defaults();

public:
  //! The output level for all code in the absence of a config file.
  int output_level = 5;

  //! The deb out log filename, nullptr disables the debug output log file.
  const char* log_filename = nullptr;

  //! Get if the log file output is enabled
  bool logfile() const { return log_filename != nullptr; }

  //! Function to deb out on the console, nullptr if output disabled.
  print_function console_print = print_char_to_cerr;

  //! Get if the console
  bool console() const { return console_print != nullptr; }

public:
  //! The output level for the given filename and function.
  int custom_level(const char* const _flnm, const char* const _fnct) const;

  /*!
  Load the configuration file specified either by the environment variable
  or the filename if the the environment variable is not set.
  \todo Document the config format.
  \return true if the configuration file was loaded properly, false otherwise.
  */
  bool load(const char* const _cnfg_envr, const char* const _cnfg_flnm);

private:
  class LevelFilterMap;

private:
  LevelFilterMap* lvl_fltrs_ = nullptr;

private:
  //! Private constructor
  Config();

  //! Private destructor
  ~Config();

  //! Disable copy
  Config(const Config&);

  //! Disable assignment
  Config& operator=(const Config&);
};

/// Helper struct to locally set the global output level and reset it when leaving the scope
struct ScopedOutputLevel
{
  ScopedOutputLevel(int level) : lvl_bfre_(Config::query().output_level) { Config::modify().output_level = level; }
  ~ScopedOutputLevel() { Config::modify().output_level = lvl_bfre_; }
  int lvl_bfre_;
};

}; // namespace Debug

#endif // DEB_ON
#endif // BASE_DEBCONFIG_HH_INCLUDED
