// GLM compatibility: provide glm::distance2 and glm::length2 which were
// moved to extensions in newer GLM versions bundled with recent CUDA toolkits.
#pragma once

#include <glm/glm.hpp>

#ifndef GLM_COMPAT_PYRXMESH
#define GLM_COMPAT_PYRXMESH
namespace glm {
    template<typename T, qualifier Q>
    __host__ __device__ __forceinline__ T distance2(const vec<3, T, Q>& a, const vec<3, T, Q>& b) {
        vec<3, T, Q> d = a - b;
        return dot(d, d);
    }

    template<typename T, qualifier Q>
    __host__ __device__ __forceinline__ T length2(const vec<3, T, Q>& v) {
        return dot(v, v);
    }
}
#endif
