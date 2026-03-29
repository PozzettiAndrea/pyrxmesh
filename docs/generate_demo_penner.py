"""
Test and benchmark Penner conformal metric optimization on GPU.

Usage:
    python docs/generate_demo_penner.py
    python docs/generate_demo_penner.py --mesh dragon
    python docs/generate_demo_penner.py --iterations 50 --verbose
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RXMESH_INPUT = os.path.join(os.path.dirname(__file__), "..", "RXMesh", "input")

def main():
    parser = argparse.ArgumentParser(description="Test Penner conformal optimization on GPU.")
    parser.add_argument("--mesh", default="dragon", help="Mesh name (without .obj)")
    parser.add_argument("--iterations", type=int, default=100, help="Max Newton iterations")
    parser.add_argument("--min-angle", type=float, default=25.0, help="Min angle for metric interpolation (degrees)")
    parser.add_argument("--error-eps", type=float, default=1e-10, help="Convergence threshold")
    parser.add_argument("--verbose", action="store_true", help="Print per-iteration info")
    args = parser.parse_args()

    import pyrxmesh
    pyrxmesh.init()

    mesh_path = os.path.join(RXMESH_INPUT, args.mesh + ".obj")
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        sys.exit(1)

    print(f"Loading {args.mesh}...")
    v, f = pyrxmesh.load_obj(mesh_path)
    print(f"  {len(v)} vertices, {len(f)} faces")

    # ── Run Penner conformal optimization ──────────────────────────────
    print(f"\n=== Penner Conformal Optimization ===")
    print(f"  max_iterations={args.iterations}, min_angle={args.min_angle}°, eps={args.error_eps}")

    t0 = time.time()
    result = pyrxmesh.penner_conformal(
        v, f,
        error_eps=args.error_eps,
        max_iterations=args.iterations,
        min_angle_deg=args.min_angle,
        verbose=args.verbose,
    )
    t_total = time.time() - t0

    print(f"\n=== Results ===")
    print(f"  Newton iterations: {result['newton_iterations']}")
    print(f"  Final error:       {result['final_error']:.2e}")
    print(f"  Total time:        {result['total_time_ms']:.1f}ms (Python wall: {t_total*1000:.1f}ms)")
    print(f"  Vertices:          {result['num_vertices']}")
    print(f"  Edges:             {result['num_edges']}")

    # ── Analyze result ─────────────────────────────────────────────────
    log_lengths = result['log_lengths']
    lengths = np.exp(log_lengths)
    print(f"\n=== Edge Length Statistics ===")
    print(f"  min:  {lengths.min():.6f}")
    print(f"  max:  {lengths.max():.6f}")
    print(f"  avg:  {lengths.mean():.6f}")
    print(f"  std:  {lengths.std():.6f}")
    print(f"  ratio max/min: {lengths.max()/lengths.min():.2f}")

    # ── Compare: run CPU QuadWild if available ─────────────────────────
    print(f"\n=== Comparison: CPU QuadWild Isotropic Remesh (1 iter) ===")
    try:
        t0 = time.time()
        rv, rf = pyrxmesh.feature_remesh(v, f, iterations=1, verbose=False)
        t_cpu = time.time() - t0
        print(f"  Time:     {t_cpu*1000:.1f}ms")
        print(f"  Vertices: {len(rv)} (from {len(v)})")
        print(f"  Faces:    {len(rf)} (from {len(f)})")
    except Exception as e:
        print(f"  [skipped] {e}")

    print(f"\n=== Summary ===")
    converged = "YES" if result['final_error'] < args.error_eps else "NO"
    print(f"  Penner converged:  {converged}")
    print(f"  Penner time:       {result['total_time_ms']:.1f}ms")
    print(f"  Target: conformal metric with uniform cone angles (2π interior)")
    if result['final_error'] < args.error_eps:
        print(f"  ✓ The optimized metric satisfies the cone angle constraints.")
    else:
        print(f"  ✗ Did not converge — error {result['final_error']:.2e} > {args.error_eps}")


if __name__ == "__main__":
    main()
