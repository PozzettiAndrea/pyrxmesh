"""Multi-mesh remesh benchmark.

Runs pyrxmesh.remesh() on Armadillo (172k), Happy Buddha (543k),
and Asian Dragon (3.6M) to compare performance across scales.
"""

import os
import sys
import time

import numpy as np
import pyrxmesh

RXMESH_INPUT = os.path.join(os.path.dirname(__file__), "..", "RXMesh", "input")

MESHES = [
    ("Armadillo (172k)", "armadillo.obj"),
    ("Happy Buddha (543k)", "happy_buddha.obj"),
    ("Asian Dragon (3.6M)", "xyzrgb_dragon.obj"),
]

print("=== pyrxmesh remesh benchmark ===")
print()

pyrxmesh.init()

for label, filename in MESHES:
    path = os.path.join(RXMESH_INPUT, filename)
    if not os.path.exists(path):
        print(f"--- {label}: SKIPPED (not found) ---")
        print()
        continue

    print(f"--- {label} ---")

    t0 = time.perf_counter()
    v, f = pyrxmesh.load_obj(path)
    t_load = time.perf_counter() - t0
    print(f"  Load: {len(v):,} verts, {len(f):,} faces ({t_load:.2f}s)")

    sys.stdout.flush()
    sys.stderr.flush()

    t0 = time.perf_counter()
    vo, fo = pyrxmesh.remesh(v, f, relative_len=1.0, iterations=2, verbose=True)
    t_remesh = time.perf_counter() - t0

    print(f"  Output: {len(vo):,} verts, {len(fo):,} faces")
    print(f"  Remesh wall clock: {t_remesh:.2f}s")
    print()
