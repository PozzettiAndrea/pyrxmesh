#!/bin/bash
# Build all pyrxmesh dependencies + the main Python module.
# Single build directory: build/
# Logs: build/build.log
set -e

JOBS=$(nproc)
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/build/build.log"
mkdir -p "$ROOT/build"

echo "=== Building pyrxmesh and all dependencies ==="
echo "Root: $ROOT"
echo "Jobs: $JOBS"
echo "Log:  $LOG"
echo ""

# Log everything
exec > >(tee "$LOG") 2>&1

# ── Build external deps (separate CMake project) ──────────────────
echo "=== [1/3] External dependencies ==="
mkdir -p "$ROOT/build/extern"
cd "$ROOT/build/extern"
cmake "$ROOT/extern" -DCMAKE_BUILD_TYPE=Release -DNPROC=$JOBS 2>&1
echo ""
echo "--- Building penner ---"
cmake --build . --target penner -j$JOBS 2>&1 || echo "!!! penner FAILED"
echo ""
echo "--- Building quantization ---"
cmake --build . --target quantization -j$JOBS 2>&1 || echo "!!! quantization FAILED"
echo ""
echo "--- Building libqex ---"
cmake --build . --target libqex -j$JOBS 2>&1 || echo "!!! libqex FAILED"
# Build extract_quads tool (links against libQExStatic)
g++ -std=c++17 -O2 \
    -I "$ROOT/extern/libQEx/interfaces/c" -I "$ROOT/extern/libQEx/src" \
    "$ROOT/extern/libQEx/extract_quads.cpp" \
    -L "$ROOT/build/extern/libqex" -lQExStatic -lOpenMeshCore -lOpenMeshTools \
    -o "$ROOT/build/extern/libqex/extract_quads" 2>&1 || echo "!!! extract_quads FAILED"
echo ""
echo "--- Building quadwild ---"
cmake --build . --target quadwild -j$JOBS 2>&1 || echo "!!! quadwild FAILED"
echo ""
echo "  ✓ External deps done"

# ── Build PennerQuad tool ─────────────────────────────────────────
echo "=== [2/3] PennerQuad tool ==="
mkdir -p "$ROOT/build/pennerquad"
cd "$ROOT/build/pennerquad"
cmake "$ROOT/tools/pennerquad" 2>&1
cmake --build . -j$JOBS 2>&1 || echo "!!! pennerquad FAILED"
echo "  ✓ PennerQuad done"

# ── Build pyrxmesh Python module ──────────────────────────────────
echo "=== [3/3] pyrxmesh Python module ==="
cd "$ROOT"
PATH="/usr/local/cuda/bin:$PATH" pip install --no-build-isolation -e . 2>&1 || echo "!!! pyrxmesh FAILED"
echo "  ✓ pyrxmesh done"

echo ""
echo "=== Build summary ==="
echo "Log: $LOG"
# Check what built successfully
for bin in \
    "build/extern/penner/bin/parameterize_aligned" \
    "build/extern/quantization/Quantization" \
    "build/extern/libqex/libQExStatic.a" \
    "build/extern/quadwild/Build/bin/quadwild" \
    "build/pennerquad/pennerquad"; do
    if [ -f "$ROOT/$bin" ]; then
        echo "  ✓ $bin"
    else
        echo "  ✗ $bin (NOT FOUND)"
    fi
done
