"""Standalone benchmark for remeshing the Asian Dragon (3.6M verts).

No fd redirection — all RXMesh spdlog output (construction timing, METIS,
per-iteration split/collapse/flip/smooth) prints directly to terminal.
"""

import os
import sys
import time

import numpy as np
import pyrxmesh

RXMESH_INPUT = os.path.join(os.path.dirname(__file__), "..", "RXMesh", "input")
MESH = os.path.join(RXMESH_INPUT, "xyzrgb_dragon.obj")

if not os.path.exists(MESH):
    print(f"ERROR: {MESH} not found")
    sys.exit(1)

print("=== pyrxmesh remesh benchmark ===")
print(f"Mesh: {MESH}")
print()

pyrxmesh.init()

print("--- Loading OBJ ---")
t0 = time.perf_counter()
v, f = pyrxmesh.load_obj(MESH)
t_load = time.perf_counter() - t0
print(f"Loaded: {len(v):,} verts, {len(f):,} faces ({t_load:.2f}s)")
print()

print("--- Running remesh (relative_len=1.0, iterations=2) ---")
sys.stdout.flush()
sys.stderr.flush()

t0 = time.perf_counter()
vo, fo = pyrxmesh.remesh(v, f, relative_len=1.0, iterations=2, verbose=True)
t_remesh = time.perf_counter() - t0

print()
print(f"--- Done ---")
print(f"Output: {len(vo):,} verts, {len(fo):,} faces")
print(f"Python wall clock: {t_remesh:.2f}s")
