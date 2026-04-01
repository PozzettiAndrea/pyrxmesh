"""
FSCQ pipeline step-by-step visualization.
Field Smoothness-Controlled Partition for Quadrangulation (Liang et al. 2025).

Usage:
    python docs/demo_fscq.py
    python docs/demo_fscq.py --mesh dragon
"""

import os
import sys
import time
import argparse
import subprocess
import warnings
import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RXMESH_INPUT = os.path.join(SCRIPT_DIR, "..", "RXMesh", "input")
FSCQ_DIR = os.path.join(SCRIPT_DIR, "..", "extern", "fscq")
FSCQ_BIN = os.path.join(FSCQ_DIR, "bin")
FSCQ_DATA = os.path.join(FSCQ_DIR, "data")
OUT_DIR = os.path.join(SCRIPT_DIR, "_site", "demo_fscq")

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
EDGE_COLOR = "#30363d"
FEAT_COLOR = "#f85149"
PATCH_CMAP = "tab20"

os.makedirs(OUT_DIR, exist_ok=True)


def load_obj(path):
    verts, faces, lines = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                face = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                faces.append(face)
            elif parts[0] == "l" and len(parts) >= 3:
                lines.append([int(parts[1]) - 1, int(parts[2]) - 1])
    return np.array(verts) if verts else np.zeros((0, 3)), faces, lines


def load_ply_field(path):
    """Load FSCQ PLY: vertices, faces, per-face field, per-vertex singularity, per-edge type."""
    verts, faces, field_u, field_v, si, edges = [], [], [], [], [], []
    with open(path) as f:
        # Parse header
        nV = nF = nE = 0
        for line in f:
            line = line.strip()
            if line.startswith("element vertex"):
                nV = int(line.split()[-1])
            elif line.startswith("element face"):
                nF = int(line.split()[-1])
            elif line.startswith("element edge"):
                nE = int(line.split()[-1])
            elif line == "end_header":
                break
        # Vertices
        for _ in range(nV):
            parts = f.readline().split()
            verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            si.append(int(parts[3]))
        # Faces
        for _ in range(nF):
            parts = f.readline().split()
            n = int(parts[0])
            face = [int(parts[i + 1]) for i in range(n)]
            faces.append(face)
            field_u.append([float(parts[n + 1]), float(parts[n + 2]), float(parts[n + 3])])
            field_v.append([float(parts[n + 4]), float(parts[n + 5]), float(parts[n + 6])])
        # Edges
        for _ in range(nE):
            parts = f.readline().split()
            edges.append({
                "v0": int(parts[0]), "v1": int(parts[1]),
                "type": int(parts[2]), "matching": int(parts[3])
            })
    return (np.array(verts), faces, np.array(field_u), np.array(field_v),
            np.array(si), edges)


def load_layout_patches(lo_path):
    """Load .lo patch file — per-face patch IDs."""
    patches = []
    with open(lo_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patches.append(int(line))
    return np.array(patches)


def pv_mesh(verts, faces):
    import pyvista as pv
    pv_faces = []
    for face in faces:
        pv_faces.append(len(face))
        pv_faces.extend(face)
    return pv.PolyData(verts, np.array(pv_faces, dtype=np.int32))


def render(mesh, filename, title, color="#58a6ff", show_edges=True,
           edge_lines=None, scalars=None, cmap="viridis", clim=None,
           window_size=(1200, 900)):
    import pyvista as pv
    pv.OFF_SCREEN = True

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    kwargs = dict(show_edges=show_edges, edge_color=EDGE_COLOR,
                  line_width=0.3, lighting=True, smooth_shading=True)
    if scalars is not None:
        pl.add_mesh(mesh, scalars=scalars, cmap=cmap, clim=clim,
                    show_scalar_bar=False, preference="cell", **kwargs)
    else:
        pl.add_mesh(mesh, color=color, **kwargs)

    if edge_lines is not None and len(edge_lines) > 0:
        pts = mesh.points
        line_pts, line_conn = [], []
        for i, (v0, v1) in enumerate(edge_lines):
            if v0 < len(pts) and v1 < len(pts):
                line_pts.extend([pts[v0], pts[v1]])
                line_conn.append([2, 2 * i, 2 * i + 1])
        if line_pts:
            line_mesh = pv.PolyData(np.array(line_pts),
                                     lines=np.array(line_conn).ravel())
            pl.add_mesh(line_mesh, color=FEAT_COLOR, line_width=3, lighting=False)

    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"  rendered {os.path.basename(filename)}")


def render_crossfield(mesh, field_u, sing_pos, filename, title,
                      window_size=(1200, 900)):
    import pyvista as pv
    pv.OFF_SCREEN = True

    # Compute face centers
    faces_arr = mesh.faces.reshape(-1, 4)[:, 1:]  # assumes triangles
    centers = mesh.points[faces_arr].mean(axis=1)

    bbox = mesh.bounds
    diag = np.sqrt((bbox[1]-bbox[0])**2 + (bbox[3]-bbox[2])**2 + (bbox[5]-bbox[4])**2)
    arrow_scale = diag * 0.012

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(mesh, color="#1f2937", show_edges=True, edge_color="#30363d",
                line_width=0.3, lighting=True, smooth_shading=True, opacity=0.4)

    # Draw field_u arrows
    pts = pv.PolyData(centers)
    pts["vectors"] = field_u * arrow_scale
    arrows = pts.glyph(orient="vectors", scale=False, factor=arrow_scale)
    pl.add_mesh(arrows, color="#58a6ff", lighting=True)

    # Draw singularities
    if len(sing_pos) > 0:
        sing_pts = pv.PolyData(sing_pos)
        spheres = sing_pts.glyph(geom=pv.Sphere(radius=diag * 0.005), scale=False)
        pl.add_mesh(spheres, color=FEAT_COLOR)

    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.add_text(f"{len(centers)}F  {len(sing_pos)} singularities",
                position="upper_right", font_size=9, color="#8b949e")
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"  rendered {os.path.basename(filename)}")


def run_cmd(cmd, label, timeout_s=120, env=None):
    print(f"  Running {label}...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                       env=env or os.environ)
    dt = time.time() - t0
    print(f"  {label}: {dt:.1f}s (exit={r.returncode})")
    if r.stdout.strip():
        for line in r.stdout.strip().split("\n")[-5:]:
            print(f"    {line}")
    if r.returncode != 0 and r.stderr.strip():
        for line in r.stderr.strip().split("\n")[-3:]:
            print(f"    [err] {line}")
    return r.stdout, r.stderr, r.returncode, dt


def main():
    parser = argparse.ArgumentParser(description="FSCQ pipeline step-by-step")
    parser.add_argument("--mesh", default="bolt")
    args = parser.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True

    mesh_path = os.path.join(RXMESH_INPUT, args.mesh + ".obj")
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        sys.exit(1)

    # Ensure FSCQ data dirs exist
    for d in ["input", "origin", "field", "layout", "quantization",
              "generate_closed_form_quad", "generate_pattern_based_quad",
              "post_process", "result_analysis"]:
        os.makedirs(os.path.join(FSCQ_DATA, d), exist_ok=True)

    # Copy input mesh
    import shutil
    shutil.copy(mesh_path, os.path.join(FSCQ_DATA, "input", args.mesh + ".obj"))

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = FSCQ_BIN + ":" + env.get("LD_LIBRARY_PATH", "")

    rows = []  # (step_name, image_path, timing, info)
    display_step = 0

    # ── Step 0: Input mesh ───────────────────────────────────────
    print("=== Input Mesh ===")
    v_in, f_in, _ = load_obj(mesh_path)
    mesh_in = pv_mesh(v_in, f_in)
    display_step += 1
    img = os.path.join(OUT_DIR, "step00_input.png")
    render(mesh_in, img, f"Input: {args.mesh}")
    rows.append((f"Step {display_step}: Input", img, "", f"{len(v_in)}V {len(f_in)}F"))

    # ── Step 1: Pre-process (remeshing + feature detection) ──────
    print("\n=== FSCQ Pre-process ===")
    _, _, rc, dt = run_cmd(
        [os.path.join(FSCQ_BIN, "pre_process"), args.mesh, "0", "1.0"],
        "pre_process", env=env)
    origin_obj = os.path.join(FSCQ_DATA, "origin", args.mesh + ".obj")
    if rc == 0 and os.path.exists(origin_obj):
        display_step += 1
        v, f, lines = load_obj(origin_obj)
        m = pv_mesh(v, f)
        img = os.path.join(OUT_DIR, f"step{display_step:02d}_preprocess.png")
        render(m, img, f"Step {display_step}: Pre-process (remesh + features)",
               color="#404040", edge_lines=lines)
        rows.append((f"Step {display_step}: Pre-process", img, f"{dt:.1f}s",
                     f"{len(v)}V {len(f)}F"))

    # ── Step 2: Cross field ──────────────────────────────────────
    print("\n=== FSCQ Cross Field ===")
    _, _, rc, dt_cf = run_cmd(
        [os.path.join(FSCQ_BIN, "generate_cross_field"), args.mesh, "0", "1.0"],
        "generate_cross_field", env=env)
    ply_path = os.path.join(FSCQ_DATA, "field", args.mesh + ".ply")
    if rc == 0 and os.path.exists(ply_path):
        display_step += 1
        verts, faces, field_u, field_v, si, edges = load_ply_field(ply_path)
        m = pv_mesh(verts, faces)

        # Feature edges from PLY
        feat_lines = [(e["v0"], e["v1"]) for e in edges if e["type"] == 1]

        # Singularity positions
        sing_mask = si != 0
        sing_pos = verts[sing_mask] if sing_mask.any() else np.zeros((0, 3))

        img = os.path.join(OUT_DIR, f"step{display_step:02d}_crossfield.png")
        render_crossfield(m, field_u, sing_pos, img,
                          f"Step {display_step}: Cross Field ({sing_mask.sum()} singularities)")
        rows.append((f"Step {display_step}: Cross Field", img, f"{dt_cf:.1f}s",
                     f"{len(verts)}V {len(faces)}F | {sing_mask.sum()} sing | {len(feat_lines)} features"))

    # ── Step 3: Layout (field-smoothness-controlled partitioning) ─
    print("\n=== FSCQ Layout ===")
    _, _, rc, dt_layout = run_cmd(
        [os.path.join(FSCQ_BIN, "generate_layout"), args.mesh, "0", "1.0"],
        "generate_layout", env=env)
    layout_obj = os.path.join(FSCQ_DATA, "layout", args.mesh + ".obj")
    layout_lo = os.path.join(FSCQ_DATA, "layout", args.mesh + ".lo")
    if rc == 0 and os.path.exists(layout_obj):
        display_step += 1
        v_lo, f_lo, lines_lo = load_obj(layout_obj)
        m_lo = pv_mesh(v_lo, f_lo)

        # Try loading patch assignments
        patch_ids = None
        if os.path.exists(layout_lo):
            patch_ids = load_layout_patches(layout_lo)
            if len(patch_ids) == len(f_lo):
                n_patches = len(set(patch_ids))
            else:
                patch_ids = None
                n_patches = "?"
        else:
            n_patches = "?"

        img = os.path.join(OUT_DIR, f"step{display_step:02d}_layout.png")
        if patch_ids is not None:
            render(m_lo, img, f"Step {display_step}: Layout ({n_patches} patches)",
                   scalars=patch_ids, cmap=PATCH_CMAP, edge_lines=lines_lo)
        else:
            render(m_lo, img, f"Step {display_step}: Layout",
                   color="#404040", edge_lines=lines_lo)
        rows.append((f"Step {display_step}: Layout", img, f"{dt_layout:.1f}s",
                     f"{len(v_lo)}V {len(f_lo)}F | {n_patches} patches"))

    # ── Generate HTML ────────────────────────────────────────────
    cache = int(time.time())
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>FSCQ Pipeline — Step by Step</title>
<style>
body {{ background: {BG_COLOR}; color: {TEXT_COLOR}; font-family: monospace; padding: 20px; max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #58a6ff; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ background: #161b22; color: #58a6ff; padding: 10px; text-align: left; border: 1px solid #30363d; }}
td {{ padding: 8px; border: 1px solid #30363d; vertical-align: top; }}
td.step {{ font-weight: bold; color: #d29922; width: 200px; }}
td img {{ width: 100%; border-radius: 4px; }}
.timing {{ color: #3fb950; }}
.info {{ color: #8b949e; font-size: 13px; }}
</style></head><body>
<h1>FSCQ Pipeline — {args.mesh}</h1>
<p>Field Smoothness-Controlled Partition for Quadrangulation (Liang et al. ACM TOG 2025)</p>
<table>
<tr><th>Step</th><th>Result</th><th>Time</th><th>Info</th></tr>
"""
    for step_name, img_path, timing, info in rows:
        t_html = f'<span class="timing">{timing}</span>' if timing else ""
        if img_path:
            rel = os.path.relpath(img_path, OUT_DIR)
            img_html = f'<img src="{rel}?v={cache}" />'
        else:
            img_html = ""
        html += f"""<tr>
<td class="step">{step_name}</td>
<td>{img_html}</td>
<td>{t_html}</td>
<td class="info">{info}</td>
</tr>\n"""

    total = sum(dt for _, _, dt_str, _ in rows if dt_str for dt in [float(dt_str.rstrip('s'))] if dt > 0)
    html += f"""</table>
<p>Total time: <span class="timing">{total:.0f}s</span></p>
</body></html>"""

    html_path = os.path.join(OUT_DIR, "index.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nHTML: {html_path}")


if __name__ == "__main__":
    main()
