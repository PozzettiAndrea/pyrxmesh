#!/usr/bin/env python3
"""Patch RXMesh for CUDA 12.4 – 13+ compatibility.

CUDA 13 removed several deprecated APIs:
  - cudaDeviceProp.clockRate, .memoryClockRate (use cudaDeviceGetAttribute)
  - cub::Sum, cub::Inequality (use cuda::std::plus etc., or inline replacements)

This script creates a compat header and patches include sites. All patches are
guarded by #if CUDART_VERSION >= 13000, so the code still builds on CUDA 12.4+.
"""

from pathlib import Path

RXMESH_ROOT = Path(__file__).parent.parent / "RXMesh"
COMPAT_HEADER = RXMESH_ROOT / "include" / "rxmesh" / "util" / "cuda13_compat.h"
SENTINEL = "PYRXMESH_CUDA13_COMPAT"


def write_compat_header():
    """Create a single compat header with replacements for removed CUB functors."""
    if COMPAT_HEADER.exists():
        print("  cuda13_compat.h already exists")
        return

    COMPAT_HEADER.write_text('''\
#pragma once
// Compatibility shims for CUDA 13+ (removed cub:: functors).
// Guarded so this is a no-op on CUDA 12.x.

#include <cub/cub.cuh>

#if CUDART_VERSION >= 13000

namespace cub {

struct Sum {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct Min {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

struct Max {
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return (b > a) ? b : a;
    }
};

struct Inequality {
    template <typename T>
    __host__ __device__ __forceinline__ bool operator()(const T &a, const T &b) const {
        return a != b;
    }
};

struct Equality {
    template <typename T>
    __host__ __device__ __forceinline__ bool operator()(const T &a, const T &b) const {
        return a == b;
    }
};

}  // namespace cub

#endif  // CUDART_VERSION >= 13000
''')
    print(f"  Created {COMPAT_HEADER.relative_to(RXMESH_ROOT.parent)}")


def patch_file(rel_path: str, after_line: str, include_line: str):
    """Insert an #include for the compat header right after `after_line`."""
    path = RXMESH_ROOT / rel_path
    text = path.read_text()
    if SENTINEL in text or "cuda13_compat.h" in text:
        print(f"  Already patched {rel_path}")
        return
    if after_line not in text:
        print(f"  Marker not found in {rel_path}, skipping")
        return
    text = text.replace(after_line, after_line + "\n" + include_line, 1)
    path.write_text(text)
    print(f"  Patched {rel_path}")


def patch_cuda_query():
    """Fix deprecated cudaDeviceProp fields in cuda_query.h."""
    path = RXMESH_ROOT / "include" / "rxmesh" / "util" / "cuda_query.h"
    text = path.read_text()
    if SENTINEL in text:
        print("  Already patched cuda_query.h")
        return

    old = '''\
    RXMESH_INFO("GPU Max Clock rate: {0:.1f} MHz ({1:.2f} GHz)",
                dev_prop.clockRate * 1e-3f,
                dev_prop.clockRate * 1e-6f);
    RXMESH_INFO("Memory Clock rate: {0:.1f} Mhz",
                dev_prop.memoryClockRate * 1e-3f);
    RXMESH_INFO("Memory Bus Width:  {}-bit", dev_prop.memoryBusWidth);
    const double maxBW = 2.0 * dev_prop.memoryClockRate *
                         (dev_prop.memoryBusWidth / 8.0) / 1.0E6;
    RXMESH_INFO("Peak Memory Bandwidth: {0:f}(GB/s)", maxBW);'''

    new = '''\
    // {SENTINEL}
#if CUDART_VERSION >= 13000
    int clockRateKHz_ = 0, memClockRateKHz_ = 0;
    cudaDeviceGetAttribute(&clockRateKHz_, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&memClockRateKHz_, cudaDevAttrMemoryClockRate, dev);
    RXMESH_INFO("GPU Max Clock rate: {0:.1f} MHz ({1:.2f} GHz)",
                clockRateKHz_ * 1e-3f, clockRateKHz_ * 1e-6f);
    RXMESH_INFO("Memory Clock rate: {0:.1f} Mhz",
                memClockRateKHz_ * 1e-3f);
    RXMESH_INFO("Memory Bus Width:  {}-bit", dev_prop.memoryBusWidth);
    const double maxBW = 2.0 * memClockRateKHz_ *
                         (dev_prop.memoryBusWidth / 8.0) / 1.0E6;
#else
    RXMESH_INFO("GPU Max Clock rate: {0:.1f} MHz ({1:.2f} GHz)",
                dev_prop.clockRate * 1e-3f, dev_prop.clockRate * 1e-6f);
    RXMESH_INFO("Memory Clock rate: {0:.1f} Mhz",
                dev_prop.memoryClockRate * 1e-3f);
    RXMESH_INFO("Memory Bus Width:  {}-bit", dev_prop.memoryBusWidth);
    const double maxBW = 2.0 * dev_prop.memoryClockRate *
                         (dev_prop.memoryBusWidth / 8.0) / 1.0E6;
#endif
    RXMESH_INFO("Peak Memory Bandwidth: {0:f}(GB/s)", maxBW);'''.replace(
        "{SENTINEL}", SENTINEL)

    if old not in text:
        print("  Pattern not found in cuda_query.h")
        return

    text = text.replace(old, new)
    path.write_text(text)
    print("  Patched cuda_query.h")


def patch_report_h():
    """Fix deprecated cudaDeviceProp fields in report.h."""
    path = RXMESH_ROOT / "include" / "rxmesh" / "util" / "report.h"
    text = path.read_text()
    if SENTINEL in text:
        print("  Already patched report.h")
        return

    old = '''\
        // Clocks
        add_member(
            "GPU Max Clock rate (GHz)", devProp.clockRate * 1e-6f, subdoc);
        add_member(
            "Memory Clock rate (GHz)", devProp.memoryClockRate * 1e-6f, subdoc);
        add_member("Memory Bus Width (bit)", devProp.memoryBusWidth, subdoc);
        add_member("Peak Memory Bandwidth (GB/s)",
                   2.0 * devProp.memoryClockRate *
                       (devProp.memoryBusWidth / 8.0) / 1.0E6,
                   subdoc);'''

    new = '''\
        // Clocks  // {SENTINEL}
#if CUDART_VERSION >= 13000
        int clockRateKHz_r = 0, memClockRateKHz_r = 0;
        cudaDeviceGetAttribute(&clockRateKHz_r, cudaDevAttrClockRate, device_id);
        cudaDeviceGetAttribute(&memClockRateKHz_r, cudaDevAttrMemoryClockRate, device_id);
        add_member("GPU Max Clock rate (GHz)", clockRateKHz_r * 1e-6f, subdoc);
        add_member("Memory Clock rate (GHz)", memClockRateKHz_r * 1e-6f, subdoc);
        add_member("Memory Bus Width (bit)", devProp.memoryBusWidth, subdoc);
        add_member("Peak Memory Bandwidth (GB/s)",
                   2.0 * memClockRateKHz_r *
                       (devProp.memoryBusWidth / 8.0) / 1.0E6,
                   subdoc);
#else
        add_member("GPU Max Clock rate (GHz)", devProp.clockRate * 1e-6f, subdoc);
        add_member("Memory Clock rate (GHz)", devProp.memoryClockRate * 1e-6f, subdoc);
        add_member("Memory Bus Width (bit)", devProp.memoryBusWidth, subdoc);
        add_member("Peak Memory Bandwidth (GB/s)",
                   2.0 * devProp.memoryClockRate *
                       (devProp.memoryBusWidth / 8.0) / 1.0E6,
                   subdoc);
#endif'''.replace("{SENTINEL}", SENTINEL)

    if old not in text:
        print("  Pattern not found in report.h")
        return

    text = text.replace(old, new)
    path.write_text(text)
    print("  Patched report.h")


def patch_delaunay_openmesh():
    """Remove OpenMesh dependency from delaunay_rxmesh.cuh.

    OpenMesh is only used for verification (count_non_delaunay_edges).
    We stub it out since we call delaunay_rxmesh with verify=false.
    """
    path = RXMESH_ROOT / "apps" / "Delaunay" / "delaunay_rxmesh.cuh"
    text = path.read_text()
    if SENTINEL in text:
        print("  Already patched delaunay_rxmesh.cuh")
        return

    # Comment out the OpenMesh include and guard dependent code
    text = text.replace(
        '#include "../common/openmesh_trimesh.h"',
        '// #include "../common/openmesh_trimesh.h"  // ' + SENTINEL,
    )

    # Wrap count_non_delaunay_edges in #ifdef PYRXMESH_HAS_OPENMESH
    text = text.replace(
        'inline uint32_t count_non_delaunay_edges(TriMesh& mesh)',
        '#ifdef PYRXMESH_HAS_OPENMESH\ninline uint32_t count_non_delaunay_edges(TriMesh& mesh)',
    )
    # Find the closing brace of count_non_delaunay_edges and add #endif
    text = text.replace(
        '    return num_non_delaunay;\n}',
        '    return num_non_delaunay;\n}\n#endif // PYRXMESH_HAS_OPENMESH',
    )

    # Guard the verification call site too
    text = text.replace(
        '    if (with_verify) {',
        '#ifdef PYRXMESH_HAS_OPENMESH\n    if (with_verify) {',
    )
    # Find where verification block ends (report.add_member after it)
    text = text.replace(
        '        report.add_member("after_num_non_delaunay_edges", num_non_del);\n    }',
        '        report.add_member("after_num_non_delaunay_edges", num_non_del);\n    }\n#endif',
    )

    path.write_text(text)
    print("  Patched delaunay_rxmesh.cuh")


def main():
    print("Patching RXMesh for CUDA 12.4+ / 13+ compatibility...")

    # 1. Create the compat header (provides cub::Sum etc. on CUDA 13+)
    write_compat_header()

    # 2. Insert #include of compat header into files that use removed CUB functors
    compat_include = '#include "rxmesh/util/cuda13_compat.h"  // ' + SENTINEL

    # reduce_handle.cu uses cub::Sum()
    patch_file(
        "include/rxmesh/reduce_handle.cu",
        '#include "rxmesh/reduce_handle.h"',
        compat_include,
    )

    # query_dispatcher.cuh uses cub::Inequality()
    patch_file(
        "include/rxmesh/kernels/query_dispatcher.cuh",
        "#include <cub/block/block_discontinuity.cuh>",
        compat_include,
    )

    # diff/scalar_term.h uses cub::Sum()
    patch_file(
        "include/rxmesh/diff/scalar_term.h",
        '#include "rxmesh/reduce_handle.h"',
        compat_include,
    )

    # 3. Fix deprecated cudaDeviceProp fields
    patch_cuda_query()
    patch_report_h()

    # 4. Remove OpenMesh dependency from Delaunay app (used for verification only)
    patch_delaunay_openmesh()

    print("Done.")


if __name__ == "__main__":
    main()
