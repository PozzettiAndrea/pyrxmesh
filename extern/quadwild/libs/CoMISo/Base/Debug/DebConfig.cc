// (C) Copyright 2019 by Autodesk, Inc.

#ifdef DEB_ON

#include "DebConfig.hh"
#include "DebFile.hh"
#include "Base/Utils/Environment.hh"

#include <fstream>
#include <sstream>
#include <iostream>
#include <list>
#include <string>
#include <map>

namespace Debug {
namespace {
// We use this data to decide the debug level of a function in a file.
class FilterLevelSelector
{
public:
  void add_file_string(const std::string& _str) 
  { 
    file_selct_strngs_.push_back(_str); 
  }

  void add_func_string(const std::string& _str) 
  { 
    func_selct_strngs_.push_back(_str); 
  }

  bool select_file(const char* _flnm) const
  {
    // TODO: this should be possible to implement w/o a copy
    std::string flnm(_flnm);
    // TODO: this code below only works in ReForm, should be made to work
    // for IGM, CoMISo, etc
    const std::string root_dir("ReForm");
    size_t pos = flnm.rfind(root_dir);
    if (pos != std::string::npos)
      flnm = flnm.substr(pos + root_dir.size());

    return search(flnm, file_selct_strngs_);
  }

  bool select_function(const char* _func) const
  {
    return search(_func, func_selct_strngs_);
  }

private:
  typedef std::list<std::string> StringList;

  // list of strings to be found inside the file name.
  StringList file_selct_strngs_; 
  // list of strings to be found inside the function name.
  StringList func_selct_strngs_; 

private:
  static bool search(const std::string& _str, const StringList& _slcts)
  {
    for (const auto& slct : _slcts)
    {
      if (_str.find(slct) != std::string::npos)
        return true;
    }
    return false;
  }
};

}//namespace 

void print_char_to_cerr(const char _c) { std::cerr << _c; }

class Config::LevelFilterMap : public std::map<int, FilterLevelSelector> {};

bool Config::load(const char* const _cnfg_envr, const char* const _cnfg_flnm)
{
  const auto flnm = System::Environment::variable(_cnfg_envr, _cnfg_flnm);

  std::ifstream cnfg_strm(flnm.c_str());
  if (!cnfg_strm.is_open())
    return false;

  delete lvl_fltrs_;
  lvl_fltrs_ = new LevelFilterMap;

  std::string line;
  while (std::getline(cnfg_strm, line))
  {
    std::stringstream line_stream(line);
    std::string type;
    line_stream >> type;
    
    void (FilterLevelSelector::*add_string)(const std::string&) = nullptr;
    
    if (type == "all") 
      {}
    else if (type == "file")
      add_string = &FilterLevelSelector::add_file_string;
    else if (type == "func")
      add_string = &FilterLevelSelector::add_func_string;
    else
      continue;

    int lvl;
    line_stream >> lvl;
    if (add_string == nullptr)
    {
      output_level = lvl; // We have read the default level.
      continue;
    }
    char colon;
    line_stream >> colon;
    if (colon != ':')
      continue;
    std::string select_str;
    while (line_stream >> select_str)
      ((*lvl_fltrs_)[lvl].*add_string)(select_str);
  }
  return true;
}

int Config::custom_level(const char* const _flnm, const char* const _fnct) const
{
  if (lvl_fltrs_ == nullptr)
    return output_level;
  int lvl = output_level;
  for (const auto& fltr : *lvl_fltrs_)
  {// continue this iteration until the maximum allowed level if found
    if (fltr.second.select_file(_flnm) || fltr.second.select_function(_fnct))
      lvl = fltr.first;
  }
  return lvl;
}

//////////////////////////////////////////////////////////////////////////
Config& Config::modify()
{
  static Config glbl_cnfg;
  return glbl_cnfg;
}

const Config& Config::query()
{
  return modify();
}

const Config& Config::defaults()
{
  static Config dflt_cnfg;
  return dflt_cnfg;
}

Config::Config() {}

Config::~Config() { delete lvl_fltrs_; }

}//namespace Debug

#endif//DEB_ON
