cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# CUDA architectures — default to native GPU
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(pyrxmesh LANGUAGES C CXX CUDA)

# ---------------------------------------------------------------------------
# Configure RXMesh as static library
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Disable unneeded RXMesh components
set(RX_USE_POLYSCOPE OFF CACHE BOOL "" FORCE)
set(RX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(RX_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(RX_USE_SUITESPARSE OFF CACHE BOOL "" FORCE)
set(RX_USE_CUDSS OFF CACHE BOOL "" FORCE)

add_subdirectory(RXMesh EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Python + nanobind
# ---------------------------------------------------------------------------
find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Static CUDA library for RXMesh pipeline (compiled with nvcc)
# This contains all RXMesh calls and __device__ lambdas.
# NO Python/nanobind headers here.
# ---------------------------------------------------------------------------
add_library(rxmesh_pipeline STATIC
  src/pipeline.cu
  src/op_qslim.cu
  src/op_sec.cu
  src/op_delaunay.cu
  src/op_remesh.cu
  src/op_geodesic.cu
  src/op_scp.cu
  src/op_param.cu
)

target_link_libraries(rxmesh_pipeline PUBLIC RXMesh)

set_target_properties(rxmesh_pipeline PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(rxmesh_pipeline PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/src/stubs
  ${CMAKE_CURRENT_SOURCE_DIR}/RXMesh/apps
)

if(NOT MSVC)
  target_compile_options(rxmesh_pipeline PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
  )
endif()

# ---------------------------------------------------------------------------
# Nanobind extension module (host C++ only — no CUDA)
# ---------------------------------------------------------------------------
nanobind_add_module(
  _pyrxmesh
  STABLE_ABI
  src/_pyrxmesh.cpp
)

# Link the pipeline library — only pipeline.h is included, no CUDA leaks
target_link_libraries(_pyrxmesh PRIVATE rxmesh_pipeline)

# Prevent CMake from adding a redundant device link step to _pyrxmesh.
# Device symbols are already resolved in rxmesh_pipeline.
set_target_properties(_pyrxmesh PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
set_property(TARGET _pyrxmesh PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

target_include_directories(_pyrxmesh PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(MSVC)
  target_compile_options(_pyrxmesh PRIVATE /O2)
else()
  target_compile_options(_pyrxmesh PRIVATE -O3)
endif()

# Install into the pyrxmesh Python package directory
install(TARGETS _pyrxmesh LIBRARY DESTINATION pyrxmesh)
