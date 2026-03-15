// Stub: OpenMesh not available in pyrxmesh bindings.
// The verification code that uses OpenMesh is skipped (verify=false).
#pragma once

struct TriMesh {};

namespace OpenMesh {
namespace IO {
inline bool read_mesh(TriMesh&, const std::string&) { return false; }
}
}
