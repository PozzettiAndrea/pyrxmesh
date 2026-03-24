// (C) Copyright 2020 by Autodesk, Inc.

#include "Base/Code/Quality.hh"
#include "Base/Security/Mandatory.hh"

#include "DebIndexMeshOut.hh"
#include "Base/Utils/FileOutput.hh"

#ifdef DEB_ON

namespace Debug
{
namespace IndexMeshOut
{

template <typename CoordT, typename IndexT>
bool save(Enter& deb, const char* const _flnm, const size_t _pnt_nmbr,
    const CoordT* const _crds, const int _face_size, const size_t _face_nmbr,
    const IndexT* const _indcs)
{
  const auto mesh_flnm = Base::make_filename("mesh", _flnm, "obj");
  try
  {
    Base::save_mesh_obj(
        mesh_flnm.c_str(), _pnt_nmbr, _crds, _face_size, _face_nmbr, _indcs);
    DEB_line(0, "Saved mesh as " << mesh_flnm);
    return true;
  }
  catch (...)
  {
    DEB_warning(0, "Failed to save mesh as " << mesh_flnm);
    return false;
  }
}

#define SAVE_MESH(COORD_TYPE, INDEX_TYPE) \
  template bool save(Enter&, const char* const, const size_t, \
      const COORD_TYPE* const, const int, const size_t, \
      const INDEX_TYPE* const)

SAVE_MESH(double, int);
SAVE_MESH(float, int);
SAVE_MESH(double, unsigned int);
SAVE_MESH(float, unsigned int);

} // namespace IndexMeshOut
} // namespace Debug

#endif // DEB_ON
