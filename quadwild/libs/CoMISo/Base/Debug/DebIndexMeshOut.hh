// (C) Copyright 2020 by Autodesk, Inc.

#ifndef BASE_DEB_INDEX_MESH_OUT_HH_INCLUDED
#define BASE_DEB_INDEX_MESH_OUT_HH_INCLUDED

#include <Base/Debug/DebOut.hh>
#include <Base/Utils/IOutputStream.hh>

#ifndef DEB_ON

#define DEB_index_mesh_out_if(CC, LL, FF, PNTS, FS, INDCS) { PROGRESS_TICK; }
#define DEB_index_mesh_out(LL, FF, PNTS, FS, INDCS) { PROGRESS_TICK; }

#else // ifndef DEB_ON

namespace Debug
{
namespace IndexMeshOut
{

template <typename CoordT, typename IndexT>
bool save(Enter& deb, const char* const _flnm, const size_t _pnt_nmbr,
    const CoordT* const _crds, const int _face_size, const size_t _face_nmbr,
    const IndexT* const _indcs);

template <class PointArrayT, class IndexArrayT>
bool save(Enter& deb, const char* const _flnm, const PointArrayT& _pnts,
    const int _face_size, const IndexArrayT& _indcs)
{
  return save(deb, _flnm, _pnts.size(), &_pnts[0][0], _face_size,
      _indcs.size() / _face_size, _indcs.data());
}

} // namespace IndexMeshOut
} // namespace Debug

#define DEB_index_mesh_out_if(CC, LL, FF, PNTS, FS, INDCS) \
  DEB_if(CC, LL, Debug::IndexMeshOut::save(deb, FF, PNTS, FS, INDCS))

#define DEB_index_mesh_out(LL, FF, PNTS, FS, INDCS) \
  DEB_index_mesh_out_if(true, LL, FF, PNTS, FS, INDCS)

#endif // ifndef DEB_ON

#endif // BASE_DEB_INDEX_MESH_OUT_HH_INCLUDED
