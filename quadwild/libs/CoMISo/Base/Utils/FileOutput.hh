// (C) Copyright 2020 by Autodesk, Inc.

#ifndef BASE_FILE_OUTPUT_HH_INCLUDED
#define BASE_FILE_OUTPUT_HH_INCLUDED

#include <Base/Config/BaseDefines.hh>
#include <string>

namespace Base
{
//! Make a file name composing the input arguments:
//               #count#_prefix_filename[_suffix].ext
// count is a number that increase any time the function is called,
// it is expressed with 4 decimal digits filled with zeros.
BASEDLLEXPORT
std::string make_filename(const char* _prfx, const char* _flnm,
    const char* _ext, const char* _sfx = nullptr);

//! Replace or add the supplied extension to the filename
BASEDLLEXPORT
std::string set_filename_extension(const char* _flnm, const char* _ext);

//! Save a mesh in obj format, supports int/uint and double/float
template <typename CoordT, typename IndexT>
void save_mesh_obj(const char* const _flnm, const size_t _pnt_nmbr,
    const CoordT* const _crds, const int _face_size, const size_t _face_nmbr,
    const IndexT* const _indcs);

//! Save a mesh in off format, supports int/uint and double/float
template <typename CoordT, typename IndexT>
void save_mesh_off(const char* const _flnm, const size_t _pnt_nmbr,
    const CoordT* const _crds, const int _face_size, const size_t _face_nmbr,
    const IndexT* const _indcs);

} // namespace Base

#endif // BASE_FILE_OUTPUT_HH_INCLUDED
