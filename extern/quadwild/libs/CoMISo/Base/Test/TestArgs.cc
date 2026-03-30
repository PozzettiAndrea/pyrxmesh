// (C) Copyright 2021 by Autodesk, Inc.

#ifdef TEST_ON

#include "TestArgs.hh"
#include "TestError.hh"

#include "Base/Test/TestChecksum.hh"
#include "Base/Test/TestChecksumLevel.hh"
#include "Base/Utils/Environment.hh"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace Test
{

namespace Args
{

namespace
{
// Stores the process name (<test-bin>)
std::string process_name_;

// Stores test-specific command-line arguments (<test-args>)
std::vector<std::string> args_;

const char* const INDENT = "  ";

// Check if a string corresponds to one of the help options -h/--help
bool help_option(const std::string& _str)
{
  return _str == "-h" || _str == "--help";
}

// Convert a string to uppercase
std::string to_upper(const std::string& _str)
{
  auto upper = _str;
  for (auto& c : upper)
    c = static_cast<char>(::toupper(c));
  return upper;
}

} // namespace

void add(const std::string& _arg) { args_.push_back(_arg); }

const std::vector<std::string>& get() { return args_; }

const std::string& process_name() { return process_name_; }

int collect(const int _argc, const char* const* _argv)
{
  // Collect the test-specific args into 'args'. Find the number of
  // framework-specific args.
  bool test_arg_section = false;
  int framework_argc = _argc;

  // The first argument is the process name
  if (_argc > 0)
    process_name_ = _argv[0];

  for (int i = 1; i < _argc; i++)
  {
    std::string arg(_argv[i]);

    if (test_arg_section)
      add(arg);
    else if (arg == "--")
    {
      test_arg_section = true;
      framework_argc = i;
    }
  }

  return framework_argc;
}

void collect_from_string(const std::string& _args)
{
  std::istringstream args_iss(_args);
  std::string arg;
  while (std::getline(args_iss, arg, ' '))
  {
    if (arg.length() > 0)
      add(arg);
  }
}

void collect_from_variable(const char* const _vrbl_name)
{
  collect_from_string(System::Environment::variable(_vrbl_name));
}

Option::Option(const std::string& _name, const int _min_arg_nmbr,
    std::shared_ptr<Mutex> _mutex)
    : name_(_name), min_arg_nmbr_(_min_arg_nmbr), mutex_(_mutex)
{
  set_usage(name() + (min_argument_number() > 0 ? " <args>" : ""));
}

uint Option::parse(const size_t _arg_nmbr, const std::string* const _args)
{
  // Check that enough arguments have been supplied
  TEST_ARGS_THROW_if(min_argument_number() > _arg_nmbr,
      "Too few arguments supplied for "
          << name() << "." << Base::ENDL << INDENT << "expected: (>=)"
          << min_argument_number() << "; supplied: " << _arg_nmbr << "."
          << Base::ENDL << Base::ENDL << INDENT << "Usage: " << usage());

  uint parsed_args_nmbr = do_parse(_arg_nmbr, _args);
  parsed_ = true;

  return parsed_args_nmbr;
}

bool Option::check_and_set_mutex()
{
  if (mutex_ == nullptr)
    return true;

  if (mutex_->is_set())
    return false;

  mutex_->set();
  return true;
}

void Option::throw_if_not_parsed() const
{
  TEST_ARGS_THROW_if(!parsed(),
      "The " << name()
             << " option is being accessed but it has not been parsed.");
}

StringOption::StringOption(const std::string& _name, const std::string& _str)
  : Option("--" + _name, 1), str(_str)
{
  set_usage(name() + " <arg>");
  set_description("Set " + _name + " to <arg>. Default = " + str);
}

// Bound toggle groups repository 
BoundOptionVector& bound_options() 
{ 
  static BoundOptionVector bnd_opts;
  return bnd_opts; 
}

BoundOption::BoundOption(const char* const _name, const std::string& _val,
    const PostParse _post_parse)
    : StringOption(_name, _val), post_parse(_post_parse)
{
  bound_options().emplace_back(this);
}

void move_bound_options(BoundOptionVector& _bnd_opts)
{
  bound_options().insert(bound_options().end(),
      std::make_move_iterator(_bnd_opts.begin()),
      std::make_move_iterator(_bnd_opts.end()));
  _bnd_opts.clear();
}

ToggleGroup::ToggleGroup(
    const std::string& _name, const std::string& _description)
    : ToggleGroup(_name, _description, std::make_shared<Option::Mutex>())
{
}

bool ToggleGroup::on() const
{
  throw_if_not_parsed();
  return option_on_.parsed();
}

bool ToggleGroup::off() const
{
  throw_if_not_parsed();
  return option_off_.parsed();
}

void ToggleGroup::throw_if_not_parsed() const
{
  TEST_ARGS_THROW_if(!parsed(), "The '"
                                    << name_
                                    << "' toggle option group is being "
                                       "accessed but it has not been parsed.");
}

ToggleGroup::ToggleGroup(const std::string& _name,
    const std::string& _description, std::shared_ptr<Option::Mutex> _mutex)
    : name_{_name},
      option_on_{"--" + _name + "-on", 0, _mutex},
      option_off_{"--" + _name + "-off", 0, _mutex}
{
  option_on_.set_description("Enable " + _description +
                             ". Mutually exclusive with " + option_off_.name() +
                             ".");

  option_off_.set_description("Disable " + _description +
                              ". Mutually exclusive with " + option_on_.name() +
                              ".");
}

// Bound toggle groups repository 
BoundToggleGroupVector& bound_toggle_groups()
{
  static BoundToggleGroupVector bnd_tggl_grps;
  return bnd_tggl_grps;
}

BoundToggleGroup::BoundToggleGroup(
    const char* const _name, const PostParse _post_parse)
    : ToggleGroup(_name, _name), post_parse(_post_parse)
{
  bound_toggle_groups().emplace_back(this);
}


void move_bound_toggle_groups(BoundToggleGroupVector& _bnd_tggl_grps)
{
  bound_toggle_groups().insert(bound_toggle_groups().end(),
      std::make_move_iterator(_bnd_tggl_grps.begin()),
      std::make_move_iterator(_bnd_tggl_grps.end()));
  _bnd_tggl_grps.clear();
}

Parser::Parser(const uint _flags, const uint _column_width)
    : flags_(_flags), column_width_(_column_width)
{
  if (flag_on<FL_ADD_BOUND>())
  { // add bound options and groups
    for (auto& bnd_opt : bound_options())
      add_option(*bnd_opt);
    for (auto& bnd_tggl_grp : bound_toggle_groups())
      add_option_group(*bnd_tggl_grp);
  }
}

void Parser::add_option(Option& _option)
{
  // Throw an error if the option attempts to override the -h or --help options.
  TEST_ARGS_THROW_if(help_option(_option.name()),
      _option.name() << " cannot override -h or --help.");

#ifdef __APPLE__
  TEST_ARGS_THROW_if(!options_.emplace(_option.name(), _option).second,
      _option.name() << " has already been added to the argument parser.");
#else  // __APPLE__
  TEST_ARGS_THROW_if(!options_.try_emplace(_option.name(), _option).second,
      _option.name() << " has already been added to the argument parser.");
#endif // __APPLE__
}

void Parser::add_option_group(ToggleGroup& _option_group)
{
  add_option(_option_group.option_on());
  add_option(_option_group.option_off());
}

void Parser::parse()
{
  const auto& args = get();

  // Run through args once to check for -h or --help
  for (auto& arg : args)
  {
    if (help_option(arg))
    {
      display_help();
      std::exit(0);
    }
  }

  // Run through args again now to parse the options
  for (size_t i = 0; i < args.size(); i++)
  {
    auto entry = options_.find(args[i]);
    if (entry == options_.end())
    {
      // Throw error if the option is not recognised and the flag
      // FL_ALLOW_UNKNOWN is not on
      TEST_ARGS_THROW_if(
          !flag_on<FL_ALLOW_UNKNOWN>(), args[i] << " is not a known option.");
      continue;
    }

    auto& option = entry->second;

    // Throw error if option has already been set and the flag FL_OVERWRITE is
    // not on
    TEST_ARGS_THROW_if(option.parsed() && !flag_on<FL_OVERWRITE>(),
        args[i] << " has already been set.");

    // Throw error if option is mutually exclusive with an option that has
    // already been set
    TEST_ARGS_THROW_if(!option.check_and_set_mutex(),
        "An option that is mutually exclusive with "
            << args[i] << " has already been set.");

    // Parse the option
    i += option.parse(args.size() - (i + 1), args.data() + (i + 1));
  }

  if (flag_on<FL_ADD_BOUND>())
  {// post-parse bound options and groups
    for (auto& bnd_opt : bound_options())
      bnd_opt->post_parse(*bnd_opt);
    for (auto& bnd_tggl_grp : bound_toggle_groups())
      bnd_tggl_grp->post_parse(*bnd_tggl_grp);
  }
}

void Parser::display_help()
{
  Base::OStringStream oss;

  oss << "Usage:" << Base::ENDL << Base::ENDL
      << "  <test-binary> [<framework-args>] [-- <test-args>]" << Base::ENDL
      << Base::ENDL << "<test-args> options:" << Base::ENDL << Base::ENDL;

  for (auto& entry : options_)
  {
    auto& option = entry.second;

    auto usage = option.usage();
    auto description = "= " + option.description();

    oss << INDENT << usage;

    if (usage.length() > column_width_)
      oss << Base::ENDL << INDENT << std::string(column_width_, ' ');
    else
      oss << std::string(column_width_ - usage.length(), ' ');

    oss << description << Base::ENDL;
  }

  std::cout << oss.str << std::endl;
}

DebugLevel::DebugLevel(int _default_level)
    : Option("--debug-level", 1), default_level_(_default_level)
{
  set_usage(name() + " <n>");
  set_description(
      "Set debug level to n (n < 0 disables debug output). Default = " +
      std::to_string(default_level_) + ".");
}

int DebugLevel::level() const
{
  throw_if_not_parsed();
  return level_;
}

uint DebugLevel::do_parse(const size_t, const std::string* const _args)
{
  level_ = parse_level(_args[0]);
  return 1;
}

int DebugLevel::parse_level(const std::string& _arg)
{
  try
  {
    return std::stoi(_arg);
  }
  catch (...)
  {
    TEST_ARGS_THROW(name() << ": '" << _arg << "' is not a valid debug level.");
  }
}

ChecksumLevel::ChecksumLevel() : Option("--checksum-level", 1)
{
  set_usage(name() + " <L> [force|max]");
  set_description(
      "Adjust the checksum run level specified by each test. Use 'force' "
      "(default) to set the level to L, and use 'max' to reduce any level "
      "above L to L, where L = NONE|STABLE|PRIME|ALL.");
}

Test::Checksum::Level ChecksumLevel::level() const
{
  throw_if_not_parsed();
  return level_;
}

ChecksumLevel::Flag ChecksumLevel::flag() const
{
  throw_if_not_parsed();
  return flag_;
}

uint ChecksumLevel::do_parse(
    const size_t _arg_nmbr, const std::string* const _args)
{
  level_ = parse_level(_args[0]);

  if (_arg_nmbr > 1 && parse_flag(_args[1], flag_))
    return 2;

  flag_ = ChecksumLevel::FL_FORCE;
  return 1;
}

bool ChecksumLevel::parse_flag(
    const std::string& _arg, ChecksumLevel::Flag& _flag)
{
  // Make _arg uppercase
  auto arg = to_upper(_arg);

  if (arg == "FORCE")
  {
    _flag = ChecksumLevel::Flag::FL_FORCE;
    return true;
  }

  if (arg == "MAX")
  {
    _flag = ChecksumLevel::Flag::FL_MAX;
    return true;
  }

  return false;
}

Checksum::Level ChecksumLevel::parse_level(const std::string& _arg)
{
  // Make _arg uppercase
  auto arg = to_upper(_arg);

#ifdef __APPLE__
  const size_t num_levels =
      sizeof(Checksum::LEVEL_TEXT) / sizeof(Checksum::LEVEL_TEXT[0]);
#else  // __APPLE__
  const size_t num_levels = std::size(Checksum::LEVEL_TEXT);
#endif // __APPLE__

  for (size_t i = 0; i < num_levels; i++)
  {
    if (arg == Checksum::LEVEL_TEXT[i])
      return static_cast<Checksum::Level>(i);
  }

  TEST_ARGS_THROW(
      name() << ": '" << _arg << "' is not a valid checksum level.");
}

} // namespace Args

} // namespace Test

#endif // TEST_ON
