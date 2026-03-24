// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_TESTARGS_HH_INCLUDED
#define BASE_TESTARGS_HH_INCLUDED

#ifdef TEST_ON

#include "Base/Test/TestChecksumLevel.hh"
#include "Base/Utils/OStringStream.hh"

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Test
{

namespace Args
{
typedef unsigned int uint;

//! Add a test-specific command-line argument
void add(const std::string& _arg);

//! Retrieve the vector of test-specific command-line arguments
const std::vector<std::string>& get();

//! Retrieve the process name
const std::string& process_name();

/*!
Collect the process name and test-specific arguments from argv[] and return the
number of framework-specific args.

The following format is expected for the command-line arguments:

  > <test-bin> <framework-args> [-- <test-args>]

where <test-bin> is the process name, <framework-args> are the arguments, such
as the test name, to be passed to the test framework (e.g. Catch2, gtest), and
<test-args> are arguments to be passed directly to the test being run.
*/
int collect(const int _argc, const char* const* _argv);

/*!
Collect test-specific arguments from a space-delimited string. Note: This is a
simple implementation, and will split by space whether or not the space
character is enclosed in quotes or is escaped in some way.
*/
void collect_from_string(const std::string &_args);

/*!
Collect test-specific arguments from a space-delimited string stored in an
environment variable. Internally, this calls collect_from_string().
*/
void collect_from_variable(const char* const _vrbl_name);

/*!
Base class for test command-line options. Every option that we intend to parse
must have an associated class derived from Option that implements the member
function do_parse(). Options must be constructed with:

  * the option's name (e.g. '--option' or '-O');
  * the minimum number of required arguments (e.g. 2).
  * (optional) a shared pointer to a mutex object, if the option exists as part
    of a mutual exclusion group.

Note: the option's name should not be '--help' or '-h', as these names are
reserved by the parser.
*/
class Option
{
public:
  //! A simple wrapper around a mutex flag
  class Mutex
  {
  public:
    Mutex() = default;

    //! Get if the mutex flag has been set
    bool is_set() { return set_; }

    //! Set the mutex flag
    void set() { set_ = true; }

  private:
    bool set_ = false;
  };

  Option(const std::string& _name, const int _min_arg_nmbr,
      std::shared_ptr<Mutex> _mutex = nullptr);

  virtual ~Option() = default;

  /*!
  Parse the arguments for the current option and return the number of arguments
  processed. Internally, this calls do_parse().

    * _arg_nmbr: number of available arguments
    * _args:     pointer to array of arguments
  */
  uint parse(const size_t _arg_nmbr, const std::string* const _args);

  //! Get the name of the option
  const std::string& name() const { return name_; }

  //! Get if the option has been parsed
  bool parsed() const { return parsed_; }

  /*!
  Check if a mutually-exclusive option has already been parsed. If not, set the
  mutex flag.
  */
  bool check_and_set_mutex();

  /*!
  Get a string describing the usage/syntax of the option. The default string is
  appropriate for options that take 0 arguments.
  */
  const std::string& usage() const { return usage_; };

  //! Set the string describing the usage/syntax of the option.
  void set_usage(const std::string& _usage) { usage_ = _usage; }

  //! Get a string describing the option.
  const std::string& description() const { return description_; }

  //! Set the string describing the option.
  void set_description(const std::string& _description) { description_ = _description; }

protected:
  //! Get the minimum number of arguments required by this option
  uint min_argument_number() const { return min_arg_nmbr_; }

  /*!
  Virtual function to parse the arguments for the current option. The return
  value should be the number of arguments that have been successfully parsed.
  Use TEST_ARGS_THROW[_if]() (defined in TestError.hh) to report errors.

    * const size_t [_arg_nmbr]: Number of available arguments (guaranteed to be
      at least min_argument_number()).

    * const std::string* const [_args]: Pointer to the array of available
      arguments.
  */
  virtual uint do_parse(const size_t, const std::string* const) { return 0; }

  //! Throws an error if the option has not been parsed
  void throw_if_not_parsed() const;

private:
  const std::string name_;
  const uint min_arg_nmbr_;
  std::string usage_;
  std::string description_;
  bool parsed_ = false;
  std::shared_ptr<Mutex> const mutex_;
};

//! Generic option with one argument as string
class StringOption : public Option
{
public:
  std::string str;

  StringOption(const std::string& _name, const std::string& _str);

protected:
  uint do_parse(const size_t, const std::string* const _args) override
  {
    str = _args[0];
    return 1;
  }
};

/*! 
Make an option and bind it to something else (e.g. Base::Option) to allow a
call to a post-parse event.

\note Argument bindings are self-managed! That is: 
  * _Only_ make with new, never on the stack or statically!
  * Do not call delete, all bindings will auto-delete on binary unload.
*/
struct BoundOption : public StringOption
{
  using PostParse = std::function<void(const StringOption&)>;
  const PostParse post_parse;

  BoundOption(const char* const _name, const std::string& _val,
      const PostParse _post_parse);
};

using BoundOptionVector = std::vector<std::unique_ptr<BoundOption>>;

//! Direct access to all bound options
BoundOptionVector& bound_options();

//! Move bound options
void move_bound_options(BoundOptionVector& _bnd_opts);

/*!
A class that wraps mutually-exclusive 'on' and 'off' options for a particular
feature.

The names of the created options will be: "--{_name}-on" and "--{_name}-off".
The respective descriptions of the created options will contain "Enable
{_description}." and "Disable {_description}.".

The options can be added to the parser using the Parser::add_option_group()
function.
*/
class ToggleGroup
{
public:
  ToggleGroup(const std::string& _name, const std::string& _description);

  //! Return references to the individual options
  Option& option_on() { return option_on_; }
  Option& option_off() { return option_off_; }

  //! Get if at least one of the options has been parsed
  bool parsed() const { return option_on_.parsed() || option_off_.parsed(); }

  /*!
  Get the state of the toggle. Throws error if !parsed(). Note: on() == !off().
  */
  bool on() const;
  bool off() const;

protected:
  //! Throws an error if the toggle option group has not been parsed
  void throw_if_not_parsed() const;

private:
  ToggleGroup(const std::string& _name, const std::string& _description,
      std::shared_ptr<Option::Mutex> _mutex);

  std::string name_;
  Option option_on_;
  Option option_off_;
};

/*! 
Make a toggle group and bind it to something else (e.g. Base::Option) to allow a
call to a post-parse event.

\note see \ref BoundOption for important information abound bound options and
groups.
*/
struct BoundToggleGroup : public ToggleGroup
{
  using PostParse = std::function<void(const ToggleGroup&)>;
  const PostParse post_parse;

  BoundToggleGroup(const char* const _name, const PostParse _post_parse);
};

using BoundToggleGroupVector = std::vector<std::unique_ptr<BoundToggleGroup>>;

//! Direct access to all bound toggle groups
BoundToggleGroupVector& bound_toggle_groups();

//! Move bound toggle groups
void move_bound_toggle_groups(BoundToggleGroupVector& _bnd_tggl_grps);

/*!
Class that performs parsing of the arguments in Test::Args::get(). Options may
be added using the add_option() member function.
*/
class Parser
{
public:
  enum Flags //!< Flags to control parser behavior
  {
    FL_NONE = 0,                          //!< no flags are set
    FL_OVERWRITE = 1,                     //!< allow options to be overwritten
    FL_ALLOW_UNKNOWN = FL_OVERWRITE << 1, //!< allow unknown options
    FL_ADD_BOUND = FL_ALLOW_UNKNOWN << 1, //! add bound options and groups
    FL_DEFAULT = FL_ALLOW_UNKNOWN | FL_ADD_BOUND //!< default flag
  };

  /*!
  Constructor for the Parser class.

    * _flags:           controls the behavior of the parser (see enum Flags)
    * _column_width:    controls the width of the first column of the help
                        display
  */
  explicit Parser(
      const uint _flags = FL_DEFAULT, const uint _column_width = 30);

  /*!
  Add an option to the parser. An option will be rejected if an existing option
  has the same name, or if the name is '--help' or '-h', as these are reserved
  by the parser.
  */
  void add_option(Option& _option);

  /*!
  Adds a group of options to the parser. Internally, this calls add_option().
  Note: Currently, the only existing option group is the ToggleGroup class. If
  more option groups are added, they should be refactored to be child classes of
  a base OptionGroup class, and the signature of this function altered to accept
  OptionGroup& instead.
  */
  void add_option_group(ToggleGroup& _option_group);

  /*!
  Parse the arguments in Test::Args::get() using the options that have been
  added.
  */
  void parse();

  /*!
  Display the help text for the parser. This function makes use of the usage()
  and description() member functions for each attached option.
  */
  void display_help();

private:
  //! Check if a flag is set on
  template <enum Flags _flag> bool flag_on() const
  {
    return (flags_ & _flag) == _flag;
  }

  std::map<std::string, Option&> options_;
  const uint flags_;
  const uint column_width_;
};

//! Option to enable/disable the journal
class JournalToggleGroup : public ToggleGroup
{
public:
  JournalToggleGroup() : ToggleGroup("journal", "the journal") {}
};

//! Option to enable/disable progress tracking
class ProgressToggleGroup : public ToggleGroup
{
public:
  ProgressToggleGroup() : ToggleGroup("progress", "progress tracking") {}
};

//! Option to set the debug output level
class DebugLevel : public Option
{
public:
  DebugLevel(int _default_level);

  //! Return debug level. Throws error if !parsed().
  int level() const;

  //! Return default debug level.
  int default_level() const { return default_level_; }

  //! Return if debug output is enabled. Throws error if !parsed().
  bool enabled() const { return level() >= 0; }

protected:
  uint do_parse(const size_t, const std::string* const _args) override;

private:
  int parse_level(const std::string& _arg);

  int level_;
  int default_level_;
};

//! Option to adjust the checksum run level
class ChecksumLevel : public Option
{
public:
  enum Flag
  {
    FL_FORCE, //! Set the checksum run level to the specified value
    FL_MAX    //! *Reduce* the checksum run level to the specified value
  };

  ChecksumLevel();

  //! Return checksum level. Throws error if !parsed().
  Test::Checksum::Level level() const;

  //! Return checksum level flag. Throws error if !parsed().
  ChecksumLevel::Flag flag() const;

protected:
  uint do_parse(
      const size_t _arg_nmbr, const std::string* const _args) override;

private:
  bool parse_flag(const std::string& _arg, ChecksumLevel::Flag& _flag);
  Test::Checksum::Level parse_level(const std::string& _arg);

  Test::Checksum::Level level_;
  Flag flag_;
};

} // namespace Args
} // namespace Test

#endif // TEST_ON

#endif // BASE_TESTARGS_HH_INCLUDED
