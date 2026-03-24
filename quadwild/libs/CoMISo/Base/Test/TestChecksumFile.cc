// (C) Copyright 2021 by Autodesk, Inc.

#ifdef TEST_ON

#include "TestChecksumFile.hh"
#include "Base/Utils/OStringStream.hh"

#include <functional>
#include <fstream>

namespace Test
{
namespace Checksum
{

void File::record(const char* const _flnm)
{
  // Reads the file, makes a hash from its data, and finally records the hash
  // and the filename.
  std::ifstream fstr(_flnm);
  std::array<char, 1000> buf;
  size_t file_hash = 0;
  size_t file_size = 0;
  while (fstr)
  {
    fstr.read(buf.data(), buf.size());
    std::string str(buf.data(), fstr.gcount());
    auto buf_hash = std::hash<std::string>{}(str);
    file_size += fstr.gcount();
    file_hash ^= buf_hash << 1; // combine hashes
  }
  Base::OStringStream strm;
  strm << "\"" << _flnm << '\"' << " (Size: " << file_size << ")"
       << "(Hash: " << Base::format_hex(file_hash) << ")";

  add(Result::OK, strm.str);
}

} // namespace Checksum
} // namespace Test

#endif // TEST_ON
