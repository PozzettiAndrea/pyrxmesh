"""
Penner pipeline step-by-step visualization.
Shows each intermediate stage from input mesh to final quad mesh.

Usage:
    python docs/generate_penner_only.py
    python docs/generate_penner_only.py --mesh dragon
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
PENNER_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "penner", "bin", "parameterize_aligned")
QUANT_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "quantization", "Quantization")
QEX_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "libqex", "extract_quads")
OUT_DIR = os.path.join(SCRIPT_DIR, "_site", "demo_penner")

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
EDGE_COLOR = "#30363d"
FEAT_COLOR = "#f85149"
SING_COLOR_POS = "#f85149"
SING_COLOR_NEG = "#58a6ff"

os.makedirs(OUT_DIR, exist_ok=True)


def load_obj(path):
    verts, faces, lines = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                face = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                faces.append(face)
            elif parts[0] == "l" and len(parts) >= 3:
                lines.append([int(parts[1]) - 1, int(parts[2]) - 1])
    return np.array(verts) if verts else np.zeros((0, 3)), faces, lines


def load_obj_with_uv(path):
    verts, uvs, faces, face_uvs = [], [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vt" and len(parts) >= 3:
                uvs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "f":
                fv, fuv = [], []
                for p in parts[1:]:
                    sp = p.split("/")
                    fv.append(int(sp[0]) - 1)
                    if len(sp) > 1 and sp[1]: fuv.append(int(sp[1]) - 1)
                faces.append(fv)
                if fuv: face_uvs.append(fuv)
    return np.array(verts), faces, np.array(uvs) if uvs else None, face_uvs if face_uvs else None


def load_crossfield(path):
    centers, dirs = [], []
    with open(path) as f:
        nF = int(f.readline())
        for _ in range(nF):
            vals = list(map(float, f.readline().split()))
            centers.append(vals[:3])
            dirs.append(np.array(vals[3:]).reshape(4, 3))
    return np.array(centers), np.array(dirs)


def load_singularities(path):
    pos, idx = [], []
    with open(path) as f:
        n = int(f.readline())
        for _ in range(n):
            vals = f.readline().split()
            if len(vals) >= 4:
                pos.append([float(vals[0]), float(vals[1]), float(vals[2])])
                idx.append(int(vals[3]))
    return np.array(pos) if pos else np.zeros((0, 3)), np.array(idx) if idx else np.array([])


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
    nV, nF = mesh.n_points, mesh.n_cells
    subtitle = f"{nV:,}V  {nF:,}F"

    pl = pv.Plotter(off_screen=True, window_size=window_size)

    kwargs = dict(show_edges=show_edges, edge_color=EDGE_COLOR,
                  line_width=0.3, lighting=True, smooth_shading=True)
    if scalars is not None:
        pl.add_mesh(mesh, scalars=scalars, cmap=cmap, clim=clim,
                    show_scalar_bar=False, **kwargs)
    else:
        pl.add_mesh(mesh, color=color, **kwargs)

    # Feature/boundary edge lines
    if edge_lines is not None and len(edge_lines) > 0:
        pts = mesh.points
        line_pts = []
        line_conn = []
        for i, (v0, v1) in enumerate(edge_lines):
            if v0 < len(pts) and v1 < len(pts):
                line_pts.extend([pts[v0], pts[v1]])
                line_conn.append([2, 2 * i, 2 * i + 1])
        if line_pts:
            line_mesh = pv.PolyData(np.array(line_pts),
                                     lines=np.array(line_conn).ravel())
            pl.add_mesh(line_mesh, color=FEAT_COLOR, line_width=3, lighting=False)
            subtitle += f"  |  {len(edge_lines)} feature edges"

    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.add_text(subtitle, position="upper_right", font_size=9, color="#8b949e")
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)

    # Also export interactive 3D HTML
    try:
        html_3d = filename.replace(".png", "_3d.html")
        pl.export_html(html_3d)
    except Exception:
        pass

    pl.close()
    print(f"  rendered {os.path.basename(filename)}")


def render_crossfield(mesh, centers, dirs, sing_pos, sing_idx, filename, title,
                      window_size=(1200, 900)):
    import pyvista as pv
    pv.OFF_SCREEN = True

    bbox = mesh.bounds
    diag = np.sqrt((bbox[1]-bbox[0])**2 + (bbox[3]-bbox[2])**2 + (bbox[5]-bbox[4])**2)
    arrow_scale = diag * 0.012

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(mesh, color="#1f2937", show_edges=True, edge_color="#30363d",
                line_width=0.3, lighting=True, smooth_shading=True, opacity=0.4)

    for i, color in [(0, "#58a6ff"), (1, "#3fb950")]:
        pts = pv.PolyData(centers)
        pts["vectors"] = dirs[:, i, :] * arrow_scale
        arrows = pts.glyph(orient="vectors", scale=False, factor=arrow_scale)
        pl.add_mesh(arrows, color=color, lighting=True)

    if len(sing_pos) > 0:
        sing_pts = pv.PolyData(sing_pos)
        spheres = sing_pts.glyph(geom=pv.Sphere(radius=diag * 0.005), scale=False)
        pl.add_mesh(spheres, color=SING_COLOR_POS)

    n_sing = len(sing_pos)
    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.add_text(f"{len(centers)}F  {n_sing} singularities",
                position="upper_right", font_size=9, color="#8b949e")
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"  rendered {os.path.basename(filename)}")


def render_uv_checker(mesh, uvs, face_uvs, filename, title, window_size=(1200, 900)):
    import pyvista as pv
    pv.OFF_SCREEN = True

    checker = np.zeros(len(face_uvs))
    for i, fuv in enumerate(face_uvs):
        u_avg = np.mean([uvs[j, 0] for j in fuv])
        v_avg = np.mean([uvs[j, 1] for j in fuv])
        checker[i] = (int(np.floor(u_avg)) + int(np.floor(v_avg))) % 2

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(mesh, scalars=checker, cmap=["#1a1a2e", "#e0e0e0"],
                show_edges=True, edge_color=EDGE_COLOR, line_width=0.3,
                lighting=True, smooth_shading=False, show_scalar_bar=False,
                preference="cell")
    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.add_text(f"{mesh.n_points:,}V  {mesh.n_cells:,}F  |  {len(uvs)} UV",
                position="upper_right", font_size=9, color="#8b949e")
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"  rendered {os.path.basename(filename)}")


def render_uv_layout(uvs, face_uvs, filename, title, window_size=(1200, 900)):
    import pyvista as pv
    pv.OFF_SCREEN = True

    verts_2d = np.column_stack([uvs[:, 0], uvs[:, 1], np.zeros(len(uvs))])
    pv_faces = []
    for fuv in face_uvs:
        pv_faces.append(len(fuv))
        pv_faces.extend(fuv)
    mesh = pv.PolyData(verts_2d, np.array(pv_faces, dtype=np.int32))

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(mesh, show_edges=True, edge_color="#58a6ff",
                color="#1f2937", line_width=0.3, lighting=False)
    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.add_text(f"{len(uvs)} UV", position="upper_right", font_size=9, color="#8b949e")
    pl.set_background(BG_COLOR)
    pl.view_xy()
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"  rendered {os.path.basename(filename)}")


def run_cmd(cmd, label, timeout_s=300):
    print(f"  Running {label}...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    dt = time.time() - t0
    print(f"  {label}: {dt:.1f}s (exit={r.returncode})")
    return r.stdout, r.stderr, r.returncode, dt


def main():
    parser = argparse.ArgumentParser(description="Penner pipeline step-by-step")
    parser.add_argument("--mesh", default="bolt")
    parser.add_argument("--sa", default="3", help="Quantization auto-scale multiplier")
    parser.add_argument("--no-ed", action="store_true", help="Disable erode/dilate feature cleanup")
    args = parser.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True

    mesh_path = os.path.join(RXMESH_INPUT, args.mesh + ".obj")
    work_dir = os.path.join(OUT_DIR, "_work")
    os.makedirs(work_dir, exist_ok=True)

    rows = []  # (step_name, image_path, timing, info)

    # ── Step 0: Input mesh ────────────────────────────────────────
    print("=== Input Mesh ===")
    v_in, f_in, _ = load_obj(mesh_path)
    mesh_in = pv_mesh(v_in, f_in)
    img = os.path.join(OUT_DIR, "step0_input.png")
    render(mesh_in, img, f"Step 0: Input ({args.mesh})")
    rows.append(("Input Mesh", img, "", f"{len(v_in)}V {len(f_in)}F"))

    # ── Step 1: Run Penner (produces all intermediate dumps) ──────
    print("\n=== Running Penner pipeline ===")
    penner_cmd = [PENNER_BIN, "--name", args.mesh, "-i", RXMESH_INPUT, "-o", work_dir]
    if args.no_ed:
        penner_cmd.append("--no_erode_dilate")
    stdout, stderr, rc, t_penner = run_cmd(penner_cmd,
        "Penner parameterize_aligned", timeout_s=300)
    # Parse step timings from spdlog timestamps
    import re
    from datetime import datetime
    step_times = {}  # step_num → seconds since previous step
    prev_time = None
    all_output = stdout + stderr
    for line in all_output.split("\n"):
        if any(k in line for k in ["Step ", "itr(", "Stopping", "Wrote"]):
            print(f"    {line.strip()}")
        m = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\].*Step (\d+)', line)
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
            step_num = m.group(2)
            if prev_time is not None:
                step_times[step_num] = (t - prev_time).total_seconds()
            prev_time = t

    # ── Render all step dumps ────────────────────────────────────
    print("\n=== Rendering steps ===")
    display_step = 0

    # Steps 1-6: mesh + feature edges (OBJ with 'l' lines)
    step_files = [
        ("01", "Feature Detection ({}°)".format(35), "detect"),
        ("02", "Erode / Dilate", "erode_dilate"),
        ("03", "Prune Small Components", "prune_comp"),
        ("04", "Refine Corner Faces", "refine_corners"),
        ("05", "Refined Mesh + Spanning Tree", "refined"),
    ]
    for file_num, label, suffix in step_files:
        path = os.path.join(work_dir, f"{args.mesh}_step{file_num}_{suffix}.obj")
        if not os.path.exists(path):
            continue
        display_step += 1
        v, f, lines = load_obj(path)
        m = pv_mesh(v, f)
        img = os.path.join(OUT_DIR, f"step{display_step:02d}_{suffix}.png")
        title = f"Step {display_step}: {label}"
        render(m, img, title, color="#404040", edge_lines=lines)
        dt = step_times.get(file_num, 0)
        timing = f"{dt:.2f}s" if dt > 0.01 else ""
        rows.append((title, img, timing, f"{len(v)}V {len(f)}F | {len(lines)} features"))

    # Cut mesh
    cut_path = os.path.join(work_dir, f"{args.mesh}_step06_cut.obj")
    if os.path.exists(cut_path):
        display_step += 1
        v_cut, f_cut, lines_cut = load_obj(cut_path)
        mesh_cut = pv_mesh(v_cut, f_cut)
        img = os.path.join(OUT_DIR, f"step{display_step:02d}_cut.png")
        title = f"Step {display_step}: Cut Mesh (seams opened)"
        render(mesh_cut, img, title, color="#404040", edge_lines=lines_cut)
        dt = step_times.get("7", 0)
        timing = f"{dt:.2f}s" if dt > 0.01 else ""
        rows.append((title, img, timing, f"{len(v_cut)}V {len(f_cut)}F | {len(lines_cut)} boundary edges"))

    # Cross field + singularities
    cf_path = os.path.join(work_dir, args.mesh + "_crossfield.txt")
    sing_path = os.path.join(work_dir, args.mesh + "_singularities.txt")
    overlay_path = os.path.join(work_dir, args.mesh + "_opt.obj")

    if os.path.exists(cf_path) and os.path.exists(overlay_path):
        display_step += 1
        v_ov, f_ov, _, _ = load_obj_with_uv(overlay_path)
        mesh_ov = pv_mesh(v_ov, f_ov)
        centers, dirs = load_crossfield(cf_path)
        sing_pos, sing_idx = (np.zeros((0,3)), np.array([]))
        if os.path.exists(sing_path):
            sing_pos, sing_idx = load_singularities(sing_path)
        img = os.path.join(OUT_DIR, f"step{display_step:02d}_crossfield.png")
        title = f"Step {display_step}: Cross Field + Singularities"
        render_crossfield(mesh_ov, centers, dirs, sing_pos, sing_idx, img, title)
        dt = step_times.get("8", 0) + step_times.get("9", 0)
        timing = f"{dt:.2f}s" if dt > 0.01 else ""
        rows.append((title, img, timing, f"{len(centers)}F | {len(sing_pos)} singularities"))

    # Newton optimization (no image, just timing)
    dt_newton = step_times.get("10", 0)
    if "10a" in step_times:
        dt_newton = step_times.get("10a", 0)
    # Try to get Newton time from step 10 → 11
    newton_lines = [l for l in all_output.split("\n") if "Step 10" in l or "Step 11" in l]
    if len(newton_lines) >= 2:
        m1 = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]', newton_lines[0])
        m2 = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]', newton_lines[-1])
        if m1 and m2:
            t1 = datetime.strptime(m1.group(1), "%Y-%m-%d %H:%M:%S.%f")
            t2 = datetime.strptime(m2.group(1), "%Y-%m-%d %H:%M:%S.%f")
            dt_newton = (t2 - t1).total_seconds()

    display_step += 1
    rows.append((f"Step {display_step}: Newton Optimization", "", f"{dt_newton:.2f}s" if dt_newton > 0 else "",
                 "Conformal metric → target cone angles"))

    # UV (checkerboard + layout) from overlay mesh
    if os.path.exists(overlay_path):
        v_ov, f_ov, uv_ov, fuv_ov = load_obj_with_uv(overlay_path)
        if uv_ov is not None and fuv_ov:
            display_step += 1
            mesh_ov = pv_mesh(v_ov, f_ov)
            img = os.path.join(OUT_DIR, f"step{display_step:02d}_uv_checker.png")
            title = f"Step {display_step}: Seamless UV"
            render_uv_checker(mesh_ov, uv_ov, fuv_ov, img, title)
            rows.append((title, img, f"{t_penner:.0f}s total",
                         f"{len(v_ov)}V {len(f_ov)}F | {len(uv_ov)} UV"))

            display_step += 1
            img = os.path.join(OUT_DIR, f"step{display_step:02d}_uv_layout.png")
            title = f"Step {display_step}: UV Layout (2D)"
            render_uv_layout(uv_ov, fuv_ov, img, title)
            rows.append((title, img, "", f"{len(uv_ov)} UV vertices"))

    # Quantization
    reembed_path = os.path.join(work_dir, "reembed.obj")
    if os.path.exists(overlay_path) and os.path.exists(QUANT_BIN):
        print("\n=== Quantization ===")
        _, _, rc_q, t_quant = run_cmd(
            [QUANT_BIN, "-s", "a", "-sa", args.sa, "-r", "-o", reembed_path, overlay_path],
            "No-T-Mesh Quantization", timeout_s=300)
        if rc_q == 0 and os.path.exists(reembed_path):
            v_re, f_re, uv_re, fuv_re = load_obj_with_uv(reembed_path)
            if uv_re is not None and fuv_re:
                display_step += 1
                mesh_re = pv_mesh(v_re, f_re)
                img = os.path.join(OUT_DIR, f"step{display_step:02d}_quantized.png")
                title = f"Step {display_step}: Quantized UV (integer grid)"
                render_uv_checker(mesh_re, uv_re, fuv_re, img, title)
                rows.append((title, img, f"{t_quant:.0f}s",
                             f"{len(v_re)}V {len(f_re)}F"))

    # Quad extraction
    quad_path = os.path.join(work_dir, "quads.obj")
    if os.path.exists(reembed_path) and os.path.exists(QEX_BIN):
        print("\n=== Quad Extraction ===")
        _, _, rc_qex, t_qex = run_cmd(
            [QEX_BIN, reembed_path, quad_path],
            "libQEx", timeout_s=60)
        if rc_qex == 0 and os.path.exists(quad_path):
            display_step += 1
            v_q, f_q, _ = load_obj(quad_path)
            mesh_q = pv_mesh(v_q, f_q)
            img = os.path.join(OUT_DIR, f"step{display_step:02d}_quads.png")
            title = f"Step {display_step}: Final Quad Mesh"
            render(mesh_q, img, title, color="#3fb950")
            rows.append((title, img, f"{t_qex:.0f}s",
                         f"{len(v_q)}V {len(f_q)} quads"))

    # ── Generate HTML ─────────────────────────────────────────────
    cache = int(time.time())
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Penner Pipeline — Step by Step</title>
<style>
body {{ background: {BG_COLOR}; color: {TEXT_COLOR}; font-family: monospace; padding: 20px; max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #58a6ff; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ background: #161b22; color: #58a6ff; padding: 10px; text-align: left; border: 1px solid #30363d; }}
td {{ padding: 8px; border: 1px solid #30363d; vertical-align: top; }}
td.step {{ font-weight: bold; color: #d29922; width: 160px; }}
td img {{ width: 100%; border-radius: 4px; cursor: pointer; }}
.timing {{ color: #3fb950; }}
.info {{ color: #8b949e; font-size: 13px; }}
.btn3d {{ display: inline-block; padding: 4px 10px; background: #238636; color: white; border-radius: 4px; text-decoration: none; font-size: 12px; margin-top: 4px; }}
.btn3d:hover {{ background: #2ea043; }}
</style></head><body>
<h1>Penner Quad Pipeline — {args.mesh}</h1>
<p>Input → Feature Detection → Cut → Cross Field → UV → Quantization → Quads</p>
<table>
<tr><th>Step</th><th>Result</th><th>Time</th><th>Info</th></tr>
"""
    for step_name, img_path, timing, info in rows:
        t_html = f'<span class="timing">{timing}</span>' if timing else ""
        if img_path:
            rel = os.path.relpath(img_path, OUT_DIR)
            rel_3d = rel.replace(".png", "_3d.html")
            has_3d = os.path.exists(os.path.join(OUT_DIR, rel_3d))
            btn_3d = f'<a class="btn3d" href="{rel_3d}" target="_blank">3D</a>' if has_3d else ""
            img_html = f'<img src="{rel}?v={cache}" />{btn_3d}'
        else:
            img_html = '<span style="color:#8b949e;font-style:italic">(no visual output)</span>'
        html += f"""<tr>
<td class="step">{step_name}</td>
<td>{img_html}</td>
<td>{t_html}</td>
<td class="info">{info}</td>
</tr>\n"""

    total = t_penner
    html += f"""</table>
<p>Total Penner time: <span class="timing">{total:.0f}s</span></p>
</body></html>"""

    html_path = os.path.join(OUT_DIR, "index.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nHTML: {html_path}")
    print(f"View: http://localhost:8001/penner_pipeline/index.html")


if __name__ == "__main__":
    main()
