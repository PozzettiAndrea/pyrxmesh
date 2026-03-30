cmake_minimum_required(VERSION 3.24)
project(pyrxmesh_extern LANGUAGES C CXX)

include(ExternalProject)

set(EXTERN_SRC ${CMAKE_CURRENT_SOURCE_DIR})
set(EXTERN_BUILD ${CMAKE_CURRENT_BINARY_DIR})
set(NPROC 12 CACHE STRING "Number of parallel jobs")

# ── Feature-aligned Penner ────────────────────────────────────────
ExternalProject_Add(penner
    SOURCE_DIR ${EXTERN_SRC}/feature-aligned-penner
    BINARY_DIR ${EXTERN_BUILD}/penner
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DENABLE_VISUALIZATION=OFF
        -DRENDER_TEXTURE=OFF
        -DUSE_PYBIND=OFF
        -DBUILD_CURVATURE_METRIC_TESTS=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${EXTERN_BUILD}/penner --target parameterize_aligned -j${NPROC}
    INSTALL_COMMAND ""
)

# ── QuantizationYoann ─────────────────────────────────────────────
ExternalProject_Add(quantization
    SOURCE_DIR ${EXTERN_SRC}/QuantizationYoann
    BINARY_DIR ${EXTERN_BUILD}/quantization
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
    INSTALL_COMMAND ""
)

# ── libQEx ────────────────────────────────────────────────────────
ExternalProject_Add(libqex
    SOURCE_DIR ${EXTERN_SRC}/libQEx
    BINARY_DIR ${EXTERN_BUILD}/libqex
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
    BUILD_COMMAND ${CMAKE_COMMAND} --build ${EXTERN_BUILD}/libqex --target QExStatic -j${NPROC}
    INSTALL_COMMAND ""
)

# ── QuadWild ──────────────────────────────────────────────────────
ExternalProject_Add(quadwild
    SOURCE_DIR ${EXTERN_SRC}/quadwild
    BINARY_DIR ${EXTERN_BUILD}/quadwild
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
    INSTALL_COMMAND ""
)
