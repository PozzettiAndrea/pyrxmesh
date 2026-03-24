// (C) Copyright 2021 by Autodesk, Inc.

#ifndef BASE_TESTENVIRONMENT_HH_INCLUDED
#define BASE_TESTENVIRONMENT_HH_INCLUDED

#ifdef TEST_ON

#include <string>

namespace Test
{
//! Reusable setup code to process the test executable arguments.
class Paths
{
public:
  /*!
  Set up the test environment and determine the output directory structure. This
  can be done in one of two ways:

  1. By supplying the test root directory (in_root), the relative path to the
     test file (in_rel_path), and, optionally, the output root directory
     (out_root) and the relative path to the test output directory
     (out_rel_path).

     The (absolute) path to the test file is in_path = in_root/in_rel_path.

     The test output directory, out_dir, is determined as follows (./ = current
     working directory):

     - If neither out_root nor out_rel_path exists, set out_dir = ./
     - If only out_root exists, set out_dir = out_root/
     - If only out_rel_path exists, set out_dir = ./out_rel_path/
     - If both out_root and out_rel_path exist, set out_dir =
       out_root/out_rel_path/

     If out_dir does not exist, it is created (including any parent
     directories). The current working directory is then switched to out_dir.

  2. By forwarding the command-line arguments (argc, argv) from the test binary
     and optionally supplying an output root directory (out_root). We expect the
     following format for the command-line arguments:

       > test-bin in_root in_rel_path out_rel_path [test_arguments]

     Having parsed the command-line arguments, we then proceed as in option 1.

     To facilitate debugging a test within an IDE, we also support the format:

       > test-bin in_path [test arguments]

     In this case the Paths object is set up identically, but no test-specific
     output folder is created, and the test output is stored within the current
     working directory.
  */
  Paths(const char* const _in_root,             //!< input root directory
      const char* const _in_rel_path,           //!< input relative file path
      const char* const _out_root = nullptr,    //!< output root directory
      const char* const _out_rel_path = nullptr //!< output relative file path
  );

  ~Paths();

  //! The input test path (directory + filename)
  const std::string& input_path() const { return in_path_; }

  //! The input test directory (path - filename)
  const std::string& input_directory() const { return in_dir_; }

  //! The input test filename (path - directory)
  const std::string& filename() const { return filename_; }

  /*!
  Construct an input filename path from the input directory and the supplied
  filename.
  */
  std::string input_filename_path(const char* const _flnm) const;

private:
#ifdef TEST_WAIT_FOR_DEBUG_ATTACH
  void wait_for_debug_attach();
#endif // TEST_WAIT_FOR_DEBUG_ATTACH

  std::string in_path_;
  std::string in_dir_;
  std::string filename_;
  std::string orig_working_dir_;
};

} // namespace Test

#endif // TEST_ON

#endif // BASE_TESTENVIRONMENT_HH_INCLUDED
