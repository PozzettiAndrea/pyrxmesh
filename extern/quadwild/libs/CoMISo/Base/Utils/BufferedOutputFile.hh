// (C) Copyright 2022 by Autodesk, Inc.

#ifndef BASE_BUFFEREDOUTPUTFILE_HH_INCLUDED
#define BASE_BUFFEREDOUTPUTFILE_HH_INCLUDED

#include <Base/Config/BaseDefines.hh>
#include <Base/Utils/IOutputStream.hh>
#include <Base/Debug/DebOut.hh>

#include <cstdio>
#include <vector>

namespace Base
{

// Provide buffered text print operations to a FILE, helps with the performance
// when storing large text data.
class BufferedOutputFile
{
public:
  BufferedOutputFile(const char* const _flnm, // filename
      const size_t _flsh_size = 1 << 20, // buffer size that forces flush (1Mb)
      const size_t _ovfl_size = 1 << 10 // "overflow" space in the buffer, also
      // sets the maximum amount of characters that can be printed at once (1Kb)
      )
      : flsh_size_(_flsh_size), ovfl_size_(_ovfl_size),
        bffr_(flsh_size_ + ovfl_size_), bffr_end_(bffr_.data())
  {
    const char* const MODE = "wb"; // open as write binary

#ifdef _MSC_VER
        fopen_s(&file_, _flnm, MODE);
#else
        file_ = std::fopen(_flnm, MODE);
#endif
  }

  ~BufferedOutputFile()
  {
    if (!opened())
      return;
    flush();
    fclose(file_);
  }

  bool opened() const { return file_ != nullptr; }

#define DEB_check_opened \
  DEB_error_if(!opened(), "Write access to a file that has failed open")

  void flush()
  {
    DEB_check_opened;
    const auto bffr_size = buffer_size();
    if (bffr_size == 0)
      return;
    fwrite(bffr_.data(), sizeof(char), bffr_size, file_);
    bffr_end_ = bffr_.data();
  }

  template <typename... ArgT> void print(const ArgT&... _args)
  {
    DEB_check_opened;
    const auto char_nmbr = Base::print(bffr_end_, ovfl_size_, _args...);
    DEB_error_if(char_nmbr < 0 || char_nmbr >= static_cast<int>(ovfl_size_),
        "print() result = "
            << char_nmbr << " is unclear, consider using larger overflow size");
    bffr_end_ += char_nmbr;
    if (buffer_size() + ovfl_size_ >= bffr_.size()) // make sure we always have
      flush(); // at least ovfl_size_ space in the buffer
  }

  template <typename T> void write(const T* const _ptr, const size_t _nmbr)
  {
    DEB_check_opened;
    flush();
    fwrite(_ptr, sizeof(T), _nmbr, file_);
  }

  // Access the file, but make sure we have flushed first
  FILE* file()
  {
    DEB_check_opened;
    flush();
    return file_;
  }

#undef DEB_check_opened

private:
  const size_t flsh_size_; // number of bytes to trigger flushing the buffer
  const size_t ovfl_size_; // number of bytes to print in a single print() call
  std::vector<char> bffr_; // buffer
  char* bffr_end_;         // end of the buffer
  FILE* file_ = nullptr;

  size_t buffer_size() const { return bffr_end_ - bffr_.data(); }
};

} // namespace Base

#endif // BASE_BUFFEREDOUTPUTFILE_HH_INCLUDED
