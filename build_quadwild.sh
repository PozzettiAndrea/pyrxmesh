#!/bin/bash
# Build the vendored QuadWild pipeline (quadwild/ submodule).
# Produces: quadwild/build/Build/bin/{quadwild,quad_from_patches,cli_trace}
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QW_DIR="$SCRIPT_DIR/quadwild"
BUILD_DIR="$QW_DIR/build"

if [ ! -f "$QW_DIR/CMakeLists.txt" ]; then
    echo "Error: quadwild submodule not found. Run: git submodule update --init --recursive"
    exit 1
fi

echo "Building QuadWild pipeline..."
cmake -S "$QW_DIR" -B "$BUILD_DIR" \
    -DSATSUMA_ENABLE_BLOSSOM5=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" --config Release -j$(nproc)

echo ""
echo "QuadWild binaries:"
ls -la "$BUILD_DIR/Build/bin/"
echo ""
echo "Usage:"
echo "  $BUILD_DIR/Build/bin/quadwild mesh.obj 4 basic_setup.txt"
