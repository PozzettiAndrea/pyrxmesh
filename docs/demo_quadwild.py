"""
QuadWild pipeline step-by-step visualization (CPU only).
Calls the QuadWild binary and renders its intermediate outputs.

Usage:
    python docs/generate_quadwild_only.py
    python docs/generate_quadwild_only.py --mesh dragon
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
import warnings
import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RXMESH_INPUT = os.path.join(SCRIPT_DIR, "..", "RXMesh", "input")
QUADWILD_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "quadwild", "Build", "bin", "quadwild")
QUADWILD_LIB = os.path.join(SCRIPT_DIR, "..", "build", "extern", "quadwild", "Build", "lib")
QUADWILD_CFG = os.path.join(SCRIPT_DIR, "..", "extern", "quadwild", "quadwild")
OUT_DIR = os.path.join(SCRIPT_DIR, "_site", "demo_quadwild")

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
EDGE_COLOR = "#30363d"
FEAT_COLOR = "#f85149"

os.makedirs(OUT_DIR, exist_ok=True)


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:]])
    return np.array(verts) if verts else np.zeros((0,3)), faces


def load_sharp(path):
    """Load QuadWild .sharp or .feature file → list of (v0, v1) edge pairs."""
    edges = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    edges.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    pass
    return edges


def pv_mesh(verts, faces):
    import pyvista as pv
    pv_faces = []
    for face in faces:
        pv_faces.append(len(face))
        pv_faces.extend(face)
    return pv.PolyData(verts, np.array(pv_faces, dtype=np.int32))


def render(mesh, filename, title, color="#58a6ff", show_edges=True,
           feat_edges=None, scalars=None, cmap="viridis", clim=None,
           window_size=(1200, 900)):
    import pyvista as pv
    pv.OFF_SCREEN = True
    nV, nF = mesh.n_points, mesh.n_cells
    subtitle = f"{nV:,}V  {nF:,}F"

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    kwargs = dict(show_edges=show_edges, edge_color=EDGE_COLOR,
                  line_width=0.3, lighting=True, smooth_shading=True)
    if scalars is not None:
        pl.add_mesh(mesh, scalars=scalars, cmap=cmap, clim=clim,
                    show_scalar_bar=True, **kwargs)
    else:
        pl.add_mesh(mesh, color=color, **kwargs)

    if feat_edges and len(feat_edges) > 0:
        pts = mesh.points
        line_pts, line_conn = [], []
        for i, (v0, v1) in enumerate(feat_edges):
            if v0 < len(pts) and v1 < len(pts):
                line_pts.extend([pts[v0], pts[v1]])
                line_conn.append([2, 2*i, 2*i+1])
        if line_pts:
            lm = pv.PolyData(np.array(line_pts), lines=np.array(line_conn).ravel())
            pl.add_mesh(lm, color=FEAT_COLOR, line_width=3, lighting=False)
        subtitle += f"  |  {len(feat_edges)} features"

    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.add_text(subtitle, position="upper_right", font_size=9, color="#8b949e")
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"  rendered {os.path.basename(filename)}")


def render_crossfield(obj_path, rosy_path, filename, title, window_size=(1200, 900)):
    """Render cross field from QuadWild's .rosy file."""
    import pyvista as pv
    pv.OFF_SCREEN = True

    v, f = load_obj(obj_path)
    mesh = pv_mesh(v, f)

    # Load rosy field (per-face 3D direction)
    vecs = []
    with open(rosy_path) as rf:
        for line in rf:
            parts = line.strip().split()
            if len(parts) == 3:
                vecs.append([float(x) for x in parts])
    vecs = np.array(vecs)

    if len(vecs) != len(f):
        print(f"  Warning: rosy has {len(vecs)} entries, mesh has {len(f)} faces")
        render(mesh, filename, title + " (no field)", color="#404040")
        return

    # Face centers + normals
    centers = np.zeros((len(f), 3))
    fnormals = np.zeros((len(f), 3))
    for i, face in enumerate(f):
        pts = np.array([v[vi] for vi in face])
        centers[i] = pts.mean(axis=0)
        e0, e1 = pts[1] - pts[0], pts[2] - pts[0]
        n = np.cross(e0, e1)
        nl = np.linalg.norm(n)
        fnormals[i] = n / nl if nl > 1e-15 else [0, 0, 1]

    # Normalize field vectors
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    vecs = vecs / norms

    # Second direction: 90° in tangent plane
    vecs90 = np.cross(fnormals, vecs)
    norms90 = np.linalg.norm(vecs90, axis=1, keepdims=True)
    norms90[norms90 < 1e-10] = 1.0
    vecs90 = vecs90 / norms90

    bbox = mesh.bounds
    diag = np.sqrt((bbox[1]-bbox[0])**2 + (bbox[3]-bbox[2])**2 + (bbox[5]-bbox[4])**2)
    arrow_scale = diag * 0.012

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(mesh, color="#1f2937", show_edges=True, edge_color="#30363d",
                line_width=0.3, lighting=True, smooth_shading=True, opacity=0.4)

    for vec_arr, color in [(vecs, "#58a6ff"), (vecs90, "#3fb950")]:
        pts = pv.PolyData(centers)
        pts["vectors"] = vec_arr * arrow_scale
        arrows = pts.glyph(orient="vectors", scale=False, factor=arrow_scale)
        pl.add_mesh(arrows, color=color, lighting=True)

    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.add_text(f"{len(f)}F", position="upper_right", font_size=9, color="#8b949e")
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"  rendered {os.path.basename(filename)}")


def render_quality(v, f, filename, title, window_size=(1200, 900)):
    import pyvista as pv
    pv.OFF_SCREEN = True
    mesh = pv_mesh(v, f)
    clean = pv.PolyData(mesh.points.copy(), mesh.faces.copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            qual = clean.compute_cell_quality(quality_measure='aspect_ratio')
        except:
            qual = clean.cell_quality(quality_measure='aspect_ratio')
    for key in ['CellQuality', 'aspect_ratio']:
        if key in qual.array_names:
            q = qual[key]; break
    else:
        q = np.ones(mesh.n_cells)

    render(mesh, filename, title, scalars=q, cmap="RdYlGn_r",
           clim=[1, min(float(q.max()), 10)], window_size=window_size)


def main():
    parser = argparse.ArgumentParser(description="QuadWild pipeline step-by-step (CPU)")
    parser.add_argument("--mesh", default="dragon")
    args = parser.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True

    mesh_path = os.path.join(RXMESH_INPUT, args.mesh + ".obj")
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        sys.exit(1)

    v_in, f_in = load_obj(mesh_path)
    print(f"Mesh: {args.mesh} ({len(v_in)}V, {len(f_in)}F)")

    rows = []

    # ── Step 0: Input ─────────────────────────────────────────────
    print("=== Input ===")
    img = os.path.join(OUT_DIR, "step0_input.png")
    render(pv_mesh(v_in, f_in), img, f"Step 0: Input ({args.mesh})")
    rows.append(("Input Mesh", img, "", f"{len(v_in)}V {len(f_in)}F"))

    # ── Run QuadWild step-by-step ─────────────────────────────────
    work_dir = os.path.join(OUT_DIR, "_work")
    os.makedirs(work_dir, exist_ok=True)
    shutil.copy(mesh_path, os.path.join(work_dir, args.mesh + ".obj"))
    for cfg in ["basic_setup.txt", "basic_setup_organic.txt", "basic_setup_mechanical.txt"]:
        src = os.path.join(QUADWILD_CFG, cfg)
        if os.path.exists(src): shutil.copy(src, work_dir)

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = QUADWILD_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
    local_mesh = os.path.join(work_dir, args.mesh + ".obj")

    # Step 1: Remesh + Field (QuadWild step 1)
    print("\n=== QuadWild Step 1: Remesh + Field ===")
    t0 = time.time()
    r = subprocess.run([QUADWILD_BIN, local_mesh, "1"],
                       capture_output=True, text=True, timeout=300,
                       cwd=work_dir, env=env)
    t_step1 = time.time() - t0
    for line in r.stdout.split("\n"):
        if line.strip(): print(f"    {line.strip()}")

    # Render remeshed mesh with features
    rem_path = os.path.join(work_dir, args.mesh + "_rem.obj")
    feat_path = os.path.join(work_dir, args.mesh + "_rem_p0.feature")
    sharp_path = os.path.join(work_dir, args.mesh + "_rem_p0.c_feature")

    if os.path.exists(rem_path):
        v_rem, f_rem = load_obj(rem_path)
        img = os.path.join(OUT_DIR, "step1_remesh.png")
        render_quality(v_rem, f_rem, img,
                       f"Step 1a: Isotropic Remesh (CPU, {t_step1:.0f}s)")
        rows.append(("Remeshed Mesh", img, f"{t_step1:.0f}s",
                     f"{len(v_rem)}V {len(f_rem)}F"))

        # Features
        feat_edges = []
        for fp in [feat_path, sharp_path]:
            if os.path.exists(fp):
                feat_edges = load_sharp(fp)
                break
        if feat_edges:
            img = os.path.join(OUT_DIR, "step1_features.png")
            render(pv_mesh(v_rem, f_rem), img,
                   "Step 1b: Detected Features", color="#404040",
                   feat_edges=feat_edges)
            rows.append(("Features", img, "", f"{len(feat_edges)} feature edges"))

        # Cross field
        rosy_path = os.path.join(work_dir, args.mesh + "_rem.rosy")
        if os.path.exists(rosy_path):
            img = os.path.join(OUT_DIR, "step1_crossfield.png")
            render_crossfield(rem_path, rosy_path, img, "Step 1c: Cross Field")
            rows.append(("Cross Field", img, "", "4-RoSy field"))

    # Step 2: Tracing (QuadWild step 2)
    print("\n=== QuadWild Step 2: Tracing ===")
    t0 = time.time()
    r = subprocess.run([QUADWILD_BIN, local_mesh, "2"],
                       capture_output=True, text=True, timeout=300,
                       cwd=work_dir, env=env)
    t_step2 = time.time() - t0
    for line in r.stdout.split("\n"):
        if any(k in line for k in ["Step", "Trac", "paths"]):
            print(f"    {line.strip()}")

    # Render patch decomposition if available
    patch_path = os.path.join(work_dir, args.mesh + "_rem_p0.obj")
    if os.path.exists(patch_path):
        v_p, f_p = load_obj(patch_path)
        img = os.path.join(OUT_DIR, "step2_patches.png")
        # Color by patch (face index mod N as proxy)
        mesh_p = pv_mesh(v_p, f_p)
        colors = np.array([i % 7 for i in range(len(f_p))], dtype=float)
        render(mesh_p, img, f"Step 2: Trace Patches ({t_step2:.0f}s)",
               scalars=colors, cmap="Set1", clim=[0, 6])
        rows.append(("Trace Patches", img, f"{t_step2:.0f}s", f"{len(f_p)}F"))

    # Step 3: Quadrangulation (QuadWild step 3)
    print("\n=== QuadWild Step 3: Quadrangulation ===")
    t0 = time.time()
    r = subprocess.run([QUADWILD_BIN, local_mesh, "3"],
                       capture_output=True, text=True, timeout=600,
                       cwd=work_dir, env=env)
    t_step3 = time.time() - t0
    for line in r.stdout.split("\n"):
        if any(k in line for k in ["Step", "Quad", "vert"]):
            print(f"    {line.strip()}")

    # Render final quad mesh
    quad_path = os.path.join(work_dir, args.mesh + "_rem_quadrangulation_smooth.obj")
    if not os.path.exists(quad_path):
        quad_path = os.path.join(work_dir, args.mesh + "_rem_quadrangulation.obj")
    if os.path.exists(quad_path):
        v_q, f_q = load_obj(quad_path)
        mesh_q = pv_mesh(v_q, f_q)
        img = os.path.join(OUT_DIR, "step3_quads.png")
        render(mesh_q, img, "Step 3: Final Quad Mesh", color="#3fb950")
        rows.append(("Quad Mesh", img, f"{t_step3:.0f}s",
                     f"{len(v_q)}V {len(f_q)} quads"))

    # ── Generate HTML ─────────────────────────────────────────────
    total = t_step1 + t_step2 + t_step3 if 't_step3' in dir() else t_step1
    cache = int(time.time())
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>QuadWild Pipeline — Step by Step</title>
<style>
body {{ background: {BG_COLOR}; color: {TEXT_COLOR}; font-family: monospace; padding: 20px; max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #58a6ff; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ background: #161b22; color: #58a6ff; padding: 10px; text-align: left; border: 1px solid #30363d; }}
td {{ padding: 8px; border: 1px solid #30363d; vertical-align: top; }}
td.step {{ font-weight: bold; color: #d29922; width: 160px; }}
td img {{ width: 100%; border-radius: 4px; }}
.timing {{ color: #3fb950; }}
.info {{ color: #8b949e; font-size: 13px; }}
</style></head><body>
<h1>QuadWild Pipeline (CPU) — {args.mesh}</h1>
<p>Input → Isotropic Remesh → Feature Detect → Cross Field → Trace → Quadrangulate</p>
<table>
<tr><th>Step</th><th>Result</th><th>Time</th><th>Info</th></tr>
"""
    for step_name, img_path, timing, info in rows:
        rel = os.path.relpath(img_path, OUT_DIR)
        t_html = f'<span class="timing">{timing}</span>' if timing else ""
        html += f"""<tr>
<td class="step">{step_name}</td>
<td><img src="{rel}?v={cache}" /></td>
<td>{t_html}</td>
<td class="info">{info}</td>
</tr>\n"""

    html += f"""</table>
<p>Total: <span class="timing">{total:.0f}s</span></p>
</body></html>"""

    html_path = os.path.join(OUT_DIR, "index.html")
    with open(html_path, "w") as fh:
        fh.write(html)
    print(f"\nHTML: {html_path}")
    print(f"View: http://localhost:8001/quadwild_pipeline/index.html")


if __name__ == "__main__":
    main()
