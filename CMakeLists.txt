cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# CUDA architectures — default to native GPU
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(pyrxmesh LANGUAGES C CXX CUDA)

# Use ccache if available (saves ~90% rebuild time)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

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
  src/op_edge_split.cu
  src/op_edge_collapse.cu
  src/op_edge_flip.cu
  src/op_quadwild.cu
  src/op_feature_remesh.cu
  src/gpu_bvh.cu
  src/pyrxmesh_c.cu
)

target_link_libraries(rxmesh_pipeline PUBLIC RXMesh)

# Penner optimization — standalone CUDA + cuSolver, no RXMesh dependency
find_package(CUDAToolkit REQUIRED)
add_library(penner_opt STATIC
  src/penner/op_penner.cu
)
target_link_libraries(penner_opt PUBLIC CUDA::cusolver CUDA::cusparse CUDA::cudart)
set_target_properties(penner_opt PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(penner_opt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(penner_opt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# NOTE: penner_real library (direct linkage to Penner .so files) is disabled.
# It had ABI mismatch issues (bad_alloc). The pipeline uses subprocess calls instead.
# TODO: Fix the ABI issue and re-enable for GPU acceleration.

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
# VCG CPU remeshing library (host C++ only — no CUDA)
# ---------------------------------------------------------------------------
add_library(vcg_remesh STATIC src/vcg_remesh.cpp)
target_include_directories(vcg_remesh PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/vcglib
  ${CMAKE_CURRENT_SOURCE_DIR}/extern/vcglib/eigenlib
)
target_compile_features(vcg_remesh PUBLIC cxx_std_17)
set_target_properties(vcg_remesh PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NOT MSVC)
  target_compile_options(vcg_remesh PRIVATE -O3 -w)
endif()

# ---------------------------------------------------------------------------
# Shared library with C API (for linking from non-CUDA projects like QuadWild)
# All CUDA/RXMesh code + device symbols resolved inside this .so.
# ---------------------------------------------------------------------------
add_library(pyrxmesh_c SHARED src/pyrxmesh_c.cu)
target_link_libraries(pyrxmesh_c PRIVATE rxmesh_pipeline)
set_target_properties(pyrxmesh_c PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME pyrxmesh_c
)
target_include_directories(pyrxmesh_c PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
install(TARGETS pyrxmesh_c LIBRARY DESTINATION lib)
install(FILES src/pyrxmesh_c.h DESTINATION include)

# ---------------------------------------------------------------------------
# Nanobind extension module (host C++ only — no CUDA)
# ---------------------------------------------------------------------------
nanobind_add_module(
  _pyrxmesh
  STABLE_ABI
  src/_pyrxmesh.cpp
)

# Link the pipeline library — only pipeline.h is included, no CUDA leaks
target_link_libraries(_pyrxmesh PRIVATE rxmesh_pipeline vcg_remesh penner_opt)

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

# ---------------------------------------------------------------------------
# QuadWild pipeline is vendored at quadwild/ (git submodule).
# Build separately with: ./build_quadwild.sh
# Produces: quadwild/build/Build/bin/{quadwild,quad_from_patches,cli_trace}
# ---------------------------------------------------------------------------
