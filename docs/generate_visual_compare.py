"""
Visual comparison of QuadWild vs Penner on specific meshes.
Runs both pipelines, renders side-by-side with SJ colormap.

Usage:
    python docs/visual_qwpn.py
    python docs/visual_qwpn.py --sa 3
"""

import os
import sys
import glob
import shutil
import time
import argparse
import subprocess
import warnings
import collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("QUADWILD_DATA", "/home/work/quadwild_data/300")
OUT_DIR = os.path.join(SCRIPT_DIR, "_site", "visual")
WORK_DIR = os.path.join(SCRIPT_DIR, "_site", "visual", "_work")

PENNERQUAD_BIN = os.path.join(SCRIPT_DIR, "..", "tools", "pennerquad", "build", "pennerquad")
QUADWILD_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "quadwild", "Build", "bin", "quadwild")
QUADWILD_LIB = os.path.join(SCRIPT_DIR, "..", "build", "extern", "quadwild", "Build", "lib")
QUADWILD_CFG = os.path.join(SCRIPT_DIR, "..", "extern", "quadwild", "quadwild")

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
EDGE_COLOR = "#30363d"

# Meshes to compare — mix of Penner-worse and Penner-better
FOCUS_MESHES = [
    "5",
    "BendTube",
    "bladefem",
    "bolt",
    "anti_backlash_nut",
    "bearing_plate",
    "6",
    "Hub",
]

os.makedirs(OUT_DIR, exist_ok=True)


def find_mesh(name):
    for cat in ["Mechanical", "Organic"]:
        p = os.path.join(DATA_DIR, cat, name + ".obj")
        if os.path.exists(p): return p
    return None


def count_vf(path):
    nv = nf = 0
    with open(path) as f:
        for line in f:
            if line.startswith("v "): nv += 1
            elif line.startswith("f "): nf += 1
    return nv, nf


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:]])
    return np.array(verts), faces


def pv_mesh(verts, faces):
    import pyvista as pv
    pv_faces = []
    for face in faces:
        pv_faces.append(len(face))
        pv_faces.extend(face)
    return pv.PolyData(verts, np.array(pv_faces, dtype=np.int32))


def compute_sj(mesh):
    import pyvista as pv
    clean = pv.PolyData(mesh.points.copy(), mesh.faces.copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            qual = clean.compute_cell_quality(quality_measure='scaled_jacobian')
        except Exception:
            qual = clean.cell_quality(quality_measure='scaled_jacobian')
    for key in ['CellQuality', 'scaled_jacobian', 'ScaledJacobian']:
        if key in qual.array_names:
            return qual[key]
    if qual.n_arrays > 0:
        return qual[qual.array_names[0]]
    return np.zeros(mesh.n_cells)


def compute_valence(mesh):
    n_per_face = mesh.faces[0]
    fa = mesh.faces.reshape(-1, n_per_face + 1)[:, 1:]
    val = np.zeros(mesh.n_points, dtype=int)
    for face in fa:
        for vi in face: val[vi] += 1
    hist = collections.Counter(val.tolist())
    target = 4 if n_per_face == 4 else 6
    return 100.0 * hist.get(target, 0) / max(mesh.n_points, 1)


def run_penner(mesh_path, mesh_name, work_dir, sa="3"):
    """Run PennerQuad (single binary: Penner → Quantization → libQEx)."""
    os.makedirs(work_dir, exist_ok=True)
    quad_path = os.path.join(work_dir, "penner_quad.obj")
    dump_dir = os.path.join(work_dir, "penner_dump")

    t0 = time.time()
    try:
        r = subprocess.run(
            [PENNERQUAD_BIN, mesh_path, quad_path, "--sa", sa, "--dump", dump_dir],
            capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.exists(quad_path):
            return None, time.time() - t0
    except subprocess.TimeoutExpired:
        return None, time.time() - t0

    return quad_path, time.time() - t0


def run_quadwild(mesh_path, mesh_name, work_dir):
    qw_dir = os.path.join(work_dir, "quadwild")
    os.makedirs(qw_dir, exist_ok=True)
    local_mesh = os.path.join(qw_dir, mesh_name + ".obj")
    shutil.copy(mesh_path, local_mesh)
    for cfg in ["basic_setup.txt", "basic_setup_organic.txt", "basic_setup_mechanical.txt"]:
        src = os.path.join(QUADWILD_CFG, cfg)
        if os.path.exists(src): shutil.copy(src, qw_dir)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = QUADWILD_LIB + ":" + env.get("LD_LIBRARY_PATH", "")

    t0 = time.time()
    try:
        r = subprocess.run(
            [QUADWILD_BIN, local_mesh, "3"],
            capture_output=True, text=True, timeout=600,
            cwd=qw_dir, env=env)
    except subprocess.TimeoutExpired:
        return None, time.time() - t0

    quad_path = os.path.join(qw_dir, mesh_name + "_rem_quadrangulation_smooth.obj")
    if not os.path.exists(quad_path):
        quad_path = os.path.join(qw_dir, mesh_name + "_rem_quadrangulation.obj")
    if not os.path.exists(quad_path):
        return None, time.time() - t0

    return quad_path, time.time() - t0


def process_mesh(args_tuple):
    name, mesh_path, sa = args_tuple
    work_dir = os.path.join(WORK_DIR, name)
    nv, nf = count_vf(mesh_path)

    p_path, p_time = run_penner(mesh_path, name, work_dir, sa=sa)
    q_path, q_time = run_quadwild(mesh_path, name, work_dir)

    return {
        "name": name, "mesh_path": mesh_path,
        "input_V": nv, "input_F": nf,
        "p_path": p_path, "p_time": p_time,
        "q_path": q_path, "q_time": q_time,
    }


def render_comparison(mesh_p, mesh_q, sj_p, sj_q, name, out_path,
                      input_mesh=None, p_time=0, q_time=0, window_size=(1800, 700)):
    import pyvista as pv
    pv.OFF_SCREEN = True

    n_cols = 3 if input_mesh else 2
    pl = pv.Plotter(off_screen=True, shape=(1, n_cols), window_size=window_size)

    col = 0
    if input_mesh:
        pl.subplot(0, col)
        pl.add_mesh(input_mesh, color="#404040", show_edges=True,
                    edge_color=EDGE_COLOR, line_width=0.3, lighting=True,
                    smooth_shading=True)
        pl.add_text(f"Input: {name}", position="upper_left", font_size=10, color=TEXT_COLOR)
        pl.add_text(f"{input_mesh.n_points}V {input_mesh.n_cells}F",
                    position="upper_right", font_size=8, color="#8b949e")
        pl.set_background(BG_COLOR)
        pl.camera_position = "iso"
        col += 1

    for label, mesh, sj, t, color in [
        ("Penner", mesh_p, sj_p, p_time, "#3fb950"),
        ("QuadWild", mesh_q, sj_q, q_time, "#58a6ff"),
    ]:
        pl.subplot(0, col)
        if mesh is not None:
            pl.add_mesh(mesh, scalars=sj, cmap="RdYlGn", clim=[0, 1],
                        show_edges=True, edge_color=EDGE_COLOR, line_width=0.3,
                        lighting=True, smooth_shading=False, show_scalar_bar=False)
            sj_avg = float(sj.mean())
            sj_min = float(sj.min())
            reg = compute_valence(mesh)
            pl.add_text(f"{label}: {mesh.n_cells}Q ({t:.0f}s)",
                        position="upper_left", font_size=10, color=TEXT_COLOR)
            pl.add_text(f"SJ avg={sj_avg:.3f} min={sj_min:.2f} reg={reg:.0f}%",
                        position="upper_right", font_size=8, color="#8b949e")
        else:
            pl.add_text(f"{label}: FAILED", position="upper_left",
                        font_size=12, color="#f85149")
        pl.set_background(BG_COLOR)
        pl.camera_position = "iso"
        col += 1

    pl.screenshot(out_path, transparent_background=False)
    pl.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sa", default="3", help="Auto-scale multiplier for Penner quantization")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True

    # Find input meshes
    tasks = []
    for name in FOCUS_MESHES:
        path = find_mesh(name)
        if path:
            tasks.append((name, path, args.sa))
        else:
            print(f"  Skipping {name}: not found in {DATA_DIR}")

    print(f"Running {len(tasks)} meshes with {args.workers} workers (sa={args.sa})")
    print("=" * 80)

    # Run pipelines in parallel
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_mesh, t): t[0] for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            name = r["name"]
            results[name] = r
            p_ok = "OK" if r["p_path"] else "FAIL"
            q_ok = "OK" if r["q_path"] else "FAIL"
            print(f"  {name:<25} P: {p_ok} ({r['p_time']:.0f}s)  Q: {q_ok} ({r['q_time']:.0f}s)")

    # Render comparisons
    print(f"\nRendering {len(results)} comparisons...")
    images = []

    for name in FOCUS_MESHES:
        if name not in results:
            continue
        r = results[name]

        # Load meshes + compute SJ
        mesh_p = sj_p = p_sj_avg = None
        if r["p_path"]:
            v, f = load_obj(r["p_path"])
            mesh_p = pv_mesh(v, f)
            sj_p = compute_sj(mesh_p)
            p_sj_avg = float(sj_p.mean())

        mesh_q = sj_q = q_sj_avg = None
        if r["q_path"]:
            v, f = load_obj(r["q_path"])
            mesh_q = pv_mesh(v, f)
            sj_q = compute_sj(mesh_q)
            q_sj_avg = float(sj_q.mean())

        # Input mesh
        iv, if_ = load_obj(r["mesh_path"])
        input_mesh = pv_mesh(iv, if_)

        out_path = os.path.join(OUT_DIR, f"{name}.png")
        render_comparison(mesh_p, mesh_q, sj_p, sj_q, name, out_path,
                          input_mesh=input_mesh,
                          p_time=r["p_time"], q_time=r["q_time"])

        diff = (q_sj_avg or 0) - (p_sj_avg or 0) if p_sj_avg and q_sj_avg else 1.0
        images.append({"path": out_path, "name": name, "diff": diff,
                        "p_sj": p_sj_avg or -1, "q_sj": q_sj_avg or -1,
                        "p_time": r["p_time"], "q_time": r["q_time"]})
        p_str = f"{p_sj_avg:.3f}" if p_sj_avg else "FAIL"
        q_str = f"{q_sj_avg:.3f}" if q_sj_avg else "FAIL"
        print(f"  {name:<25} P={p_str} Q={q_str} diff={diff:+.3f}")

    # Sort by diff (Penner-worse first)
    images.sort(key=lambda x: -x["diff"])

    # Generate HTML
    cache_bust = int(time.time())
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>QuadWild vs Penner — Visual Comparison</title>
<style>
body {{ background: {BG_COLOR}; color: {TEXT_COLOR}; font-family: monospace; padding: 20px; max-width: 1900px; margin: 0 auto; }}
h1 {{ color: #58a6ff; }}
h2 {{ color: #d29922; margin-top: 30px; font-size: 16px; }}
.mesh img {{ width: 100%; border: 1px solid #30363d; border-radius: 6px; }}
.winner-q {{ color: #f85149; }}
.winner-p {{ color: #3fb950; }}
.tie {{ color: #d29922; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
th, td {{ padding: 6px 10px; border: 1px solid #30363d; text-align: center; }}
th {{ background: #161b22; color: #58a6ff; }}
</style></head><body>
<h1>QuadWild vs Penner (sa={args.sa}) — Visual Comparison</h1>
<p>Left: Input | Center: Penner (green=good SJ) | Right: QuadWild</p>
<table>
<tr><th>Mesh</th><th>Penner SJ</th><th>QuadWild SJ</th><th>Diff</th><th>P time</th><th>Q time</th><th>Winner</th></tr>
"""
    for img in images:
        p = f"{img['p_sj']:.3f}" if img['p_sj'] >= 0 else "FAIL"
        q = f"{img['q_sj']:.3f}" if img['q_sj'] >= 0 else "FAIL"
        d = img["diff"]
        cls = "winner-q" if d > 0.01 else "winner-p" if d < -0.01 else "tie"
        w = "QuadWild" if d > 0.01 else "Penner" if d < -0.01 else "Tie"
        html += f'<tr><td>{img["name"]}</td><td>{p}</td><td>{q}</td><td class="{cls}">{d:+.3f}</td><td>{img["p_time"]:.0f}s</td><td>{img["q_time"]:.0f}s</td><td class="{cls}">{w}</td></tr>\n'
    html += "</table>\n"

    for img in images:
        rel = os.path.relpath(img["path"], OUT_DIR)
        d = img["diff"]
        cls = "winner-q" if d > 0.01 else "winner-p" if d < -0.01 else "tie"
        w = "QuadWild better" if d > 0.01 else "Penner better" if d < -0.01 else "Tie"
        html += f'<div class="mesh"><h2>{img["name"]} <span class="{cls}">({w} {d:+.3f})</span></h2><img src="{rel}?v={cache_bust}" /></div>\n'

    html += "</body></html>"
    html_path = os.path.join(OUT_DIR, "index.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nHTML: {html_path}")
    print("View: http://localhost:8002/visual_compare/index.html")


if __name__ == "__main__":
    main()
