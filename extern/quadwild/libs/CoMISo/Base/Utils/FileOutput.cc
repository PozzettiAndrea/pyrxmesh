// (C) Copyright 2020 by Autodesk, Inc.

#include "FileOutput.hh"

#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <cctype>

namespace Base
{

std::string make_filename(
    const char* _prfx, const char* _flnm, const char* _ext, const char* _sfx)
{
  static int cnt = 0;
  const char SEP = '_';
  std::stringstream sstr_flnm;
  sstr_flnm << std::setfill('0') << std::setw(4) << cnt++;
  sstr_flnm << SEP << _prfx << SEP << _flnm;
  if (_sfx != nullptr)
    sstr_flnm << SEP << _sfx;

  sstr_flnm << '.' << _ext;
  return sstr_flnm.str();
}

std::string set_filename_extension(const char* _flnm, const char* _ext)
{
  std::string flnm(_flnm);
  const auto dot_pos = flnm.find_last_of('.');

  if (dot_pos == std::string::npos)
    flnm += std::string(".") + _ext; // no dot, add the dot and the extension
  else if (dot_pos + 1 == flnm.size())
    flnm += _ext;                      // last char is dot, add the extension
  else if (isdigit(flnm[dot_pos + 1])) // char after dot is a digit?
    flnm += std::string(".") + _ext;   // add the dot and the extension
  else
    flnm.replace(flnm.begin() + dot_pos + 1, flnm.end(), _ext);

  return flnm;
}

#define SAVE_MESH(EXT, COORD_TYPE, INDEX_TYPE) \
  template void save_mesh_##EXT(const char* const, const size_t, \
      const COORD_TYPE* const, const int, const size_t, \
      const INDEX_TYPE* const)

template <typename CoordT, typename IndexT>
void save_mesh_obj(const char* const _flnm, const size_t _pnt_nmbr,
    const CoordT* const _crds, const int _face_size, const size_t _face_nmbr,
    const IndexT* const _indcs)
{
  std::ofstream strm(set_filename_extension(_flnm, "obj")); // open
  strm.imbue(std::locale("C"));                             // C-locale
  strm << std::setprecision(std::numeric_limits<CoordT>::max_digits10);

  for (size_t i = 0, n = 3 * _pnt_nmbr; i < n; i += 3)
  {
    strm << "v " << _crds[i] << ' ' << _crds[i + 1] << ' ' << _crds[i + 2]
         << std::endl;
  }

  for (size_t i = 0, n = _face_size * _face_nmbr; i < n;)
  { // stream quads
    strm << 'f';
    for (int j = 0; j < _face_size; ++j)
      strm << ' ' << _indcs[i++] + 1; // OBJ indexing starts at 1
    strm << std::endl;
  }
}

SAVE_MESH(obj, double, int);
SAVE_MESH(obj, float, int);
SAVE_MESH(obj, double, unsigned int);
SAVE_MESH(obj, float, unsigned int);

template <typename CoordT, typename IndexT>
void save_mesh_off(const char* const _flnm, const size_t _pnt_nmbr,
    const CoordT* const _crds, const int _face_size, const size_t _face_nmbr,
    const IndexT* const _indcs)
{
  std::ofstream strm(set_filename_extension(_flnm, "off")); // open
  strm.imbue(std::locale("C"));                             // C-locale
  strm << std::setprecision(std::numeric_limits<CoordT>::max_digits10);
  strm << "OFF" << std::endl; // write header
  strm << _pnt_nmbr << ' ' << _face_nmbr << std::endl;

  for (size_t i = 0, n = 3 * _pnt_nmbr; i < n; i += 3)
    strm << _crds[i] << ' ' << _crds[i + 1] << ' ' << _crds[i + 2] << std::endl;

  for (size_t i = 0, n = _face_size * _face_nmbr; i < n;)
  { // stream quads
    strm << _face_size;
    for (int j = 0; j < _face_size; ++j)
      strm << ' ' << _indcs[i++];
    strm << std::endl;
  }
}

SAVE_MESH(off, double, int);
SAVE_MESH(off, float, int);
SAVE_MESH(off, double, unsigned int);
SAVE_MESH(off, float, unsigned int);

} // namespace Base
