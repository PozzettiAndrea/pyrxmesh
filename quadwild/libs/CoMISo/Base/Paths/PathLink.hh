// (C) Copyright 2020 by Autodesk, Inc.

#ifndef BASE_PATHLINK_HH_INCLUDED
#define BASE_PATHLINK_HH_INCLUDED

#include <Base/Paths/Filesystem.hh>

namespace Base
{
namespace fs = Base::filesystem;

class PathLink
{
public:
  explicit PathLink(const fs::path& _path) : path_(_path) {}

  template <class StreamT>
  friend StreamT& operator<<(StreamT& _os, const PathLink& _link)
  {
    _os << "\""
        << "file://" << fs::weakly_canonical(_link.path_).generic_string()
        << "\"";
    return _os;
  }

private:
  const fs::path& path_;
};

} // namespace Base

#endif // BASE_PATHLINK_HH_INCLUDED
