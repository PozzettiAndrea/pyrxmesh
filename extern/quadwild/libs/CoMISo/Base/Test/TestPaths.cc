// (C) Copyright 2021 by Autodesk, Inc.

#ifdef TEST_ON

#include "TestPaths.hh"
#include "TestChecksumNumberT.hh"
#include "TestError.hh"

#include "Base/Paths/Filesystem.hh"

#include <iostream>
#include <string>

// TODO: convert this to a debug option once we implement it
//#define TEST_WAIT_FOR_DEBUG_ATTACH
#ifdef TEST_WAIT_FOR_DEBUG_ATTACH
#include <thread>
#endif // TEST_WAIT_FOR_DEBUG_ATTACH

namespace Test
{
namespace fs = Base::filesystem;

Base::IOutputStream& operator<<(Base::IOutputStream& _os, const fs::path& _path)
{
  return _os << _path.string();
}

namespace
{

void make_directory(const fs::path& _dir)
{
  fs::path chk_dir;
  for (auto path_it = _dir.begin(); path_it != _dir.end(); ++path_it)
  {
    chk_dir /= *path_it;
    if (!fs::exists(chk_dir) && !fs::create_directories(chk_dir))
    { // Directory creation can fail if the directory has already been created
      // by another process. This is likely because multiple test processes are
      // scheduled all together. In this case we can avoid a failure.
      // TODO: Make this even more secure, e.g., try this check a few more times
      // (5-10?), sleep the process before each attempt by 0.1sec.
      TEST_THROW_MSG_if(!fs::exists(chk_dir),
          "Failed creating directory "
              << chk_dir << " in the requested directory path " << _dir);
    }
  }
}

} // namespace

#ifdef TEST_WAIT_FOR_DEBUG_ATTACH
void Paths::wait_for_debug_attach()
{
  // Wait for a debugger to be attached to the test process
  if (fs::exists("c:/wait_to_debug_test.yes"))
  { // placing a breakpoint on the cycle below allows us to exit the sleep
    std::chrono::seconds duration(10);
    for (int i = 0; i < 100; ++i)
      std::this_thread::sleep_for(duration);
  }
}
#endif // TEST_WAIT_FOR_DEBUG_ATTACH

Paths::Paths(const char* const _in_root, const char* const _in_rel_path,
    const char* const _out_root, const char* const _out_rel_path)
{
#ifdef TEST_WAIT_FOR_DEBUG_ATTACH
  wait_for_debug_attach();
#endif // TEST_WAIT_FOR_DEBUG_ATTACH

  fs::path in_root(_in_root);
  fs::path in_rel_path(_in_rel_path);
  fs::path in_path = in_root / in_rel_path;

  // Verify arguments
  TEST_THROW_MSG_if(
      !fs::is_directory(in_root), in_root << " is not a directory.");

  TEST_THROW_MSG_if(
      !fs::is_regular_file(in_path), in_path << " is not a file.");

  // Determine output directory (see comment in TestPaths.hh)
  auto out_dir = _out_root ? fs::path(_out_root) : fs::current_path();
  if (_out_rel_path) out_dir /= fs::path(_out_rel_path);

  // Store the current working directory. If out_dir != current working
  // directory, create out_dir and switch to it.
  orig_working_dir_ = fs::current_path().string();
  if (!(fs::is_directory(out_dir) &&
          fs::equivalent(out_dir, fs::current_path())))
  {
    make_directory(out_dir);
    fs::current_path(out_dir); // set out_dir as current path
  }

  in_path_ = in_path.string();
  in_dir_ = in_path.parent_path().string();
  filename_ = in_path.filename().string();
}

Paths::~Paths()
{
  // Switch back to original working directory
  fs::current_path(orig_working_dir_);
}

std::string Paths::input_filename_path(const char* const _flnm) const
{
  fs::path inpt_flnm_path(input_directory());
  inpt_flnm_path /= _flnm;
  return fs::absolute(inpt_flnm_path).generic_string();
}

} // namespace Test

#endif // TEST_ON
