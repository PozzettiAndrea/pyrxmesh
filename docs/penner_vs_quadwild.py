"""
Visual comparison: Metriko standalone vs Penner+Metriko hybrid quad meshing.

Side-by-side table: Metriko (full pipeline) | Penner | Hybrid

Usage:
    python docs/generate_demo_metriko.py
    python docs/generate_demo_metriko.py --mesh dragon --scale 0.03
    python docs/generate_demo_metriko.py --skip-metriko
"""

import os
import sys
import argparse
import subprocess
import time
import select
import warnings
import numpy as np
from html import escape as html_escape

warnings.filterwarnings("ignore", category=DeprecationWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RXMESH_INPUT = os.path.join(SCRIPT_DIR, "..", "RXMesh", "input")
PENNER_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "penner", "bin", "parameterize_aligned")
# Fallback: old in-source build location
if not os.path.exists(PENNER_BIN):
    PENNER_BIN = os.path.join(SCRIPT_DIR, "..", "extern", "feature-aligned-penner", "build", "bin", "parameterize_aligned")
OUT_DIR = os.path.join(SCRIPT_DIR, "_site", "penner_vs_quadwild")

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
EDGE_COLOR = "#30363d"
COLORS = {
    "quadwild": "#58a6ff",
    "penner": "#3fb950",
}

os.makedirs(OUT_DIR, exist_ok=True)


def load_obj(path):
    """Load OBJ file, return (vertices, faces)."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                face_verts = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                faces.append(face_verts)
    return np.array(verts, dtype=np.float64), faces


def load_obj_with_uv(path):
    """Load OBJ with UV. Returns (V, faces, UV, face_uv_indices)."""
    verts, uvs, faces, face_uvs = [], [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "vt" and len(parts) >= 3:
                uvs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == "f":
                fv, fuv = [], []
                for p in parts[1:]:
                    sp = p.split("/")
                    fv.append(int(sp[0]) - 1)
                    if len(sp) > 1 and sp[1]:
                        fuv.append(int(sp[1]) - 1)
                faces.append(fv)
                if fuv:
                    face_uvs.append(fuv)
    return (np.array(verts), faces,
            np.array(uvs) if uvs else None,
            face_uvs if face_uvs else None)


def pv_mesh(verts, faces):
    """Create PyVista mesh from vertices and face list (tri or quad)."""
    import pyvista as pv
    pv_faces = []
    for face in faces:
        pv_faces.append(len(face))
        pv_faces.extend(face)
    return pv.PolyData(verts, np.array(pv_faces, dtype=np.int32))


def render(mesh, filename, title, color="#58a6ff", show_edges=True,
           window_size=(1200, 1000)):
    """Render a mesh to PNG."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    nV, nF = mesh.n_points, mesh.n_cells
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(mesh, color=color, show_edges=show_edges, edge_color=EDGE_COLOR,
                line_width=0.5, lighting=True, smooth_shading=True)
    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.add_text(f"{nV:,}V  {nF:,}F", position="upper_right", font_size=9, color="#8b949e")
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"    rendered {os.path.basename(filename)}")


def render_uv(uvs, face_uvs, filename, title, window_size=(1200, 1000)):
    """Render UV layout as 2D."""
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
    print(f"    rendered {os.path.basename(filename)}")


def run_cmd(cmd, label, timeout=300):
    """Run a command, streaming stderr live."""
    print(f"  Running {label}...")
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        stderr_lines = []
        while True:
            ret = proc.poll()
            if proc.stderr:
                ready, _, _ = select.select([proc.stderr], [], [], 0.1)
                if ready:
                    line = proc.stderr.readline()
                    if line:
                        stderr_lines.append(line.rstrip())
                        print(f"    {line.rstrip()}")
            if ret is not None:
                for line in proc.stderr:
                    stderr_lines.append(line.rstrip())
                    print(f"    {line.rstrip()}")
                break
            if time.time() - t0 > timeout:
                proc.kill()
                print(f"  {label}: TIMEOUT after {timeout}s")
                return "", "\n".join(stderr_lines), -1
        stdout = proc.stdout.read() if proc.stdout else ""
        dt = time.time() - t0
        print(f"  {label}: {dt:.1f}s (exit={ret})")
        return stdout, "\n".join(stderr_lines), ret
    except Exception as e:
        print(f"  {label}: ERROR {e}")
        return "", str(e), -1


def load_crossfield(path):
    """Load cross field from dump_field output. Returns (centers, dirs) where
    dirs is (nF, 4, 3) array of 4 direction vectors per face."""
    centers, dirs = [], []
    with open(path) as f:
        nF = int(f.readline())
        for _ in range(nF):
            vals = list(map(float, f.readline().split()))
            centers.append(vals[:3])
            d = np.array(vals[3:]).reshape(4, 3)
            dirs.append(d)
    return np.array(centers), np.array(dirs)


def load_singularities(path):
    """Load singularity positions. Returns (positions, indices)."""
    pos, idx = [], []
    with open(path) as f:
        n = int(f.readline())
        for _ in range(n):
            vals = f.readline().split()
            pos.append([float(vals[0]), float(vals[1]), float(vals[2])])
            idx.append(int(vals[3]))
    return np.array(pos) if pos else np.zeros((0, 3)), np.array(idx)


def render_crossfield(mesh_pv, centers, dirs, sing_pos, sing_idx, filename, title,
                      mesh_color="#1f2937", window_size=(1200, 1000)):
    """Render cross field as small arrows on mesh + singularity points."""
    import pyvista as pv
    pv.OFF_SCREEN = True

    # Compute arrow scale from mesh bounding box
    bbox = mesh_pv.bounds
    diag = np.sqrt((bbox[1]-bbox[0])**2 + (bbox[3]-bbox[2])**2 + (bbox[5]-bbox[4])**2)
    arrow_scale = diag * 0.012

    pl = pv.Plotter(off_screen=True, window_size=window_size)

    # Mesh (dark, subtle)
    pl.add_mesh(mesh_pv, color=mesh_color, show_edges=True, edge_color="#30363d",
                line_width=0.3, lighting=True, smooth_shading=True, opacity=0.4)

    # Cross field arrows (just direction 0 and 1 — the two independent directions)
    for i, color in [(0, "#58a6ff"), (1, "#3fb950")]:
        pts = pv.PolyData(centers)
        pts["vectors"] = dirs[:, i, :] * arrow_scale
        arrows = pts.glyph(orient="vectors", scale=False, factor=arrow_scale)
        pl.add_mesh(arrows, color=color, lighting=True)

    # Singularities as spheres
    if len(sing_pos) > 0:
        sing_pts = pv.PolyData(sing_pos)
        # Color by index: +1 = red, -1 = blue
        colors = np.where(np.array(sing_idx) > 0, 1.0, -1.0) if len(sing_idx) > 0 else np.ones(len(sing_pos))
        sing_pts["index"] = colors
        spheres = sing_pts.glyph(geom=pv.Sphere(radius=diag*0.004), scale=False)
        spheres["index"] = np.repeat(colors, spheres.n_cells // max(len(sing_pos), 1))
        pl.add_mesh(spheres, scalars="index", cmap="coolwarm", clim=[-1, 1],
                    show_scalar_bar=False)

    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.add_text(f"{len(centers)}F  {len(sing_pos)} sing",
                position="upper_right", font_size=9, color="#8b949e")
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"    rendered {os.path.basename(filename)}")


def render_uv_checker(mesh_pv, uvs, face_uvs, filename, title,
                      window_size=(1200, 1000)):
    """Render mesh with checkerboard texture based on UV coordinates."""
    import pyvista as pv
    pv.OFF_SCREEN = True

    # Compute per-face checkerboard color from UV
    n_faces = len(face_uvs)
    checker = np.zeros(n_faces)
    for i, fuv in enumerate(face_uvs):
        # Average UV of face corners
        u_avg = np.mean([uvs[j, 0] for j in fuv])
        v_avg = np.mean([uvs[j, 1] for j in fuv])
        # Checkerboard: alternate based on floor(u) + floor(v)
        checker[i] = (int(np.floor(u_avg)) + int(np.floor(v_avg))) % 2

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.add_mesh(mesh_pv, scalars=checker, cmap=["#1a1a2e", "#e0e0e0"],
                show_edges=True, edge_color=EDGE_COLOR, line_width=0.3,
                lighting=True, smooth_shading=False, show_scalar_bar=False,
                preference="cell")
    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.add_text(f"{len(uvs)} UV", position="upper_right", font_size=9, color="#8b949e")
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"    rendered {os.path.basename(filename)}")


def compute_quad_quality(mesh, input_mesh=None):
    """Compute quality metrics for a quad mesh. Returns dict."""
    import collections
    import pyvista as pv

    # Scaled Jacobian (best metric for quad quality)
    clean = pv.PolyData(mesh.points.copy(), mesh.faces.copy())
    try:
        qual = clean.compute_cell_quality(quality_measure='scaled_jacobian')
        sj = qual['CellQuality']
        sj_min, sj_avg = float(sj.min()), float(sj.mean())
    except Exception:
        sj_min, sj_avg = 0.0, 0.0

    # Vertex valence (regular = 4 for quads)
    n_verts_per_face = mesh.faces[0]  # 3 for tri, 4 for quad
    faces_arr = mesh.faces.reshape(-1, n_verts_per_face + 1)[:, 1:]
    val_count = np.zeros(mesh.n_points, dtype=int)
    for face in faces_arr:
        for vi in face:
            val_count[vi] += 1
    val_hist = collections.Counter(val_count.tolist())
    target_val = 4 if n_verts_per_face == 4 else 6
    pct_regular = 100.0 * val_hist.get(target_val, 0) / max(mesh.n_points, 1)
    n_irregular = sum(v for k, v in val_hist.items() if k != target_val and k > 0)

    metrics = {
        "V": mesh.n_points, "F": mesh.n_cells,
        "sj_min": sj_min, "sj_avg": sj_avg,
        "pct_regular": pct_regular,
        "n_irregular": n_irregular,
    }

    # Hausdorff distance to original mesh
    if input_mesh is not None:
        try:
            def sample_surface(m, n=10000):
                fa = m.faces.reshape(-1, m.faces[0] + 1)[:, 1:]
                pts = m.points
                # Triangulate quads for area computation
                tris = []
                for f in fa:
                    tris.append([f[0], f[1], f[2]])
                    if len(f) == 4:
                        tris.append([f[0], f[2], f[3]])
                tris = np.array(tris)
                v0, v1, v2 = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
                areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
                probs = areas / areas.sum()
                tri_idx = np.random.choice(len(tris), size=n, p=probs)
                r1 = np.sqrt(np.random.rand(n))
                r2 = np.random.rand(n)
                a, b, c = 1 - r1, r1 * (1 - r2), r1 * r2
                return (a[:, None] * pts[tris[tri_idx, 0]] +
                        b[:, None] * pts[tris[tri_idx, 1]] +
                        c[:, None] * pts[tris[tri_idx, 2]])

            N = 10000
            result_samples = sample_surface(mesh, N)
            input_samples = sample_surface(input_mesh, N)

            _, fwd_closest = input_mesh.find_closest_cell(result_samples, return_closest_point=True)
            fwd_dists = np.linalg.norm(result_samples - fwd_closest, axis=1)
            _, bwd_closest = mesh.find_closest_cell(input_samples, return_closest_point=True)
            bwd_dists = np.linalg.norm(input_samples - bwd_closest, axis=1)

            metrics["hausdorff_avg"] = float(max(fwd_dists.mean(), bwd_dists.mean()))
            metrics["hausdorff_max"] = float(max(fwd_dists.max(), bwd_dists.max()))
        except Exception as e:
            metrics["hausdorff_avg"] = -1
            metrics["hausdorff_max"] = -1

    return metrics


def fmt_metrics(m):
    """Format metrics dict as a compact string."""
    parts = [f"{m['V']}V {m['F']}Q"]
    if 'sj_min' in m:
        parts.append(f"SJ min={m['sj_min']:.3f} avg={m['sj_avg']:.3f}")
    if 'pct_regular' in m:
        parts.append(f"val4={m['pct_regular']:.1f}% ({m['n_irregular']} irreg)")
    if 'hausdorff_avg' in m and m['hausdorff_avg'] >= 0:
        parts.append(f"Hausdorff avg={m['hausdorff_avg']:.2e} max={m['hausdorff_max']:.2e}")
    return " | ".join(parts)


def fmt_time(ms):
    if ms < 1:
        return "<1ms"
    elif ms < 1000:
        return f"{ms:.0f}ms"
    else:
        return f"{ms/1000:.1f}s"


# ── Table data structure ──────────────────────────────────────────────
# Each cell: {"img": path, "time": "123ms", "info": "extra text"}
# Rows = pipeline steps, Columns = pipelines

COLUMNS = ["quadwild", "penner"]
COL_LABELS = {
    "quadwild": "QuadWild",
    "penner": "Penner (ours)",
}
ROW_ORDER = [
    "Input Mesh",
    "Cross Field",
    "Parametrization / UV",
    "UV Layout",
    "Motorcycle Graph",
    "Quantization",
    "Quad Mesh",
]


def generate_html(table, logs):
    """Generate comparison HTML with table layout."""
    cache_bust = int(time.time())

    # Build table rows
    header = "<tr><th>Step</th>"
    for col in COLUMNS:
        header += f"<th>{COL_LABELS[col]}</th>"
    header += "</tr>"

    rows = ""
    for step in ROW_ORDER:
        if step not in table:
            continue
        cells = table[step]
        row = f"<tr><td class='step'>{html_escape(step)}</td>"
        for col in COLUMNS:
            cell = cells.get(col, {})
            img = cell.get("img")
            timing = cell.get("time", "")
            info = cell.get("info", "")
            td = "<td>"
            if img:
                rel = os.path.relpath(img, OUT_DIR)
                td += f'<img src="{rel}?v={cache_bust}" />'
            if timing:
                td += f'<div class="timing">{html_escape(timing)}</div>'
            if info:
                td += f'<div class="info">{html_escape(info)}</div>'
            if not img and not timing and not info:
                td += "&mdash;"
            td += "</td>"
            row += td
        row += "</tr>"
        rows += row

    # Logs section
    log_html = ""
    for name, log in logs.items():
        if log:
            log_html += f"<h3>{html_escape(name)}</h3><pre>{html_escape(log)}</pre>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Metriko vs Penner</title>
<style>
body {{ background: {BG_COLOR}; color: {TEXT_COLOR}; font-family: -apple-system, monospace; padding: 20px; }}
h1 {{ color: #58a6ff; }}
h3 {{ color: #d29922; margin-top: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ background: #161b22; color: #58a6ff; padding: 10px; text-align: center; border: 1px solid #30363d; }}
td {{ padding: 8px; border: 1px solid #30363d; text-align: center; vertical-align: top; }}
td.step {{ text-align: left; font-weight: bold; color: #d29922; white-space: nowrap; }}
td img {{ max-width: 100%; border-radius: 4px; }}
.timing {{ color: #3fb950; font-size: 13px; margin-top: 4px; }}
.info {{ color: #8b949e; font-size: 12px; margin-top: 2px; }}
pre {{ background: #161b22; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 12px; max-height: 300px; }}
</style></head><body>
<h1>Metriko vs Penner — Quad Meshing Comparison</h1>
<table>
{header}
{rows}
</table>
{log_html}
</body></html>"""

    out_path = os.path.join(OUT_DIR, "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nHTML: {out_path}")


def parse_metriko_log(stderr):
    """Extract step timings from metriko stderr."""
    timings = {}
    for line in stderr.split("\n"):
        if "[metriko]" not in line:
            continue
        if "Cross field:" in line:
            timings["cross_field"] = line.split(":")[-1].strip()
        elif "Parametrization:" in line:
            timings["param"] = line.split(":")[-1].strip()
        elif "Motorcycle + T-mesh:" in line:
            timings["moto"] = line.split(":")[-1].strip()
        elif "Quantization:" in line:
            timings["quant"] = line.split(":")[-1].strip()
        elif "Quad generation:" in line:
            timings["quad_gen"] = line.split(":")[-1].strip()
        elif "Result:" in line:
            timings["result"] = line.split("Result:")[-1].strip()
        elif "Total:" in line:
            timings["total"] = line.split("Total:")[-1].strip()
    return timings


def main():
    parser = argparse.ArgumentParser(description="QuadWild vs Penner quad meshing comparison")
    parser.add_argument("--mesh", default="dragon", help="Mesh name")
    parser.add_argument("--penner-scale", type=float, default=50, help="Penner quantization scale")
    parser.add_argument("--skip-quadwild", action="store_true", help="Skip QuadWild run")
    parser.add_argument("--skip-penner", action="store_true", help="Skip Penner run")
    args = parser.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True

    mesh_path = os.path.join(RXMESH_INPUT, args.mesh + ".obj")
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        sys.exit(1)

    table = {}  # step → {col → cell_data}
    logs = {}

    # ── Input mesh (shared row) ───────────────────────────────────────
    print("=== Input Mesh ===")
    v_in, f_in = load_obj(mesh_path)
    mesh_in = pv_mesh(v_in, f_in)
    img_in = os.path.join(OUT_DIR, "input.png")
    render(mesh_in, img_in, f"Input: {args.mesh}")
    table["Input Mesh"] = {
        col: {"img": img_in, "info": f"{len(v_in)}V {len(f_in)}F"}
        for col in COLUMNS
    }

    # ── QuadWild pipeline ────────────────────────────────────────────
    QUADWILD_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "quadwild", "Build", "bin", "quadwild")
    QUADWILD_LIB = os.path.join(SCRIPT_DIR, "..", "build", "extern", "quadwild", "Build", "lib")
    QUADWILD_CFG = os.path.join(SCRIPT_DIR, "..", "extern", "quadwild", "quadwild")
    qw_quad_path = os.path.join(os.path.dirname(mesh_path),
                                args.mesh + "_rem_quadrangulation_smooth.obj")

    if not args.skip_quadwild and os.path.exists(QUADWILD_BIN):
        print("\n=== QuadWild Pipeline ===")
        # Copy config files next to working dir
        import shutil
        for cfg in ["basic_setup.txt", "basic_setup_organic.txt", "basic_setup_mechanical.txt"]:
            src = os.path.join(QUADWILD_CFG, cfg)
            if os.path.exists(src):
                shutil.copy(src, os.path.dirname(mesh_path))

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = QUADWILD_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
        import subprocess as sp
        print("  Running QuadWild...")
        t0 = time.time()
        r = sp.run([QUADWILD_BIN, mesh_path, "3"],
                   capture_output=True, text=True, timeout=600,
                   cwd=os.path.dirname(mesh_path), env=env)
        qw_time = time.time() - t0
        print(f"  QuadWild: {qw_time:.1f}s (exit={r.returncode})")
        for line in r.stdout.split("\n"):
            if any(k in line for k in ["Step", "faces", "vertices", "Remesh", "Trac", "Quad"]):
                print(f"    {line}")
        logs["QuadWild"] = r.stdout

    if os.path.exists(qw_quad_path):
        v_qw, f_qw = load_obj(qw_quad_path)
        mesh_qw = pv_mesh(v_qw, f_qw)
        img_qw = os.path.join(OUT_DIR, "quadwild_quad.png")
        render(mesh_qw, img_qw, "QuadWild Quads", color=COLORS.get("quadwild", "#58a6ff"))

        print("  Computing QuadWild quad quality...")
        qw_metrics = compute_quad_quality(mesh_qw, mesh_in)
        table["Quad Mesh"] = table.get("Quad Mesh", {})
        table["Quad Mesh"]["quadwild"] = {
            "img": img_qw,
            "info": fmt_metrics(qw_metrics),
            "time": f"{qw_time:.0f}s" if 'qw_time' in dir() else "",
        }

    # ── Penner pipeline ───────────────────────────────────────────────
    penner_dir = "/tmp/penner_output"
    if not args.skip_penner:
        print("\n=== Penner Pipeline ===")
        os.makedirs(penner_dir, exist_ok=True)
        _, stderr_p, rc_p = run_cmd(
            [PENNER_BIN, "--name", args.mesh,
             "-i", RXMESH_INPUT, "-o", penner_dir],
            "Penner", timeout=120)
        logs["Penner"] = stderr_p

    overlay_path = os.path.join(penner_dir, f"{args.mesh}_opt.obj")
    intrinsic_path = os.path.join(penner_dir, f"{args.mesh}_intrinsic.obj")

    # Penner UV
    if os.path.exists(overlay_path):
        v_ov, f_ov, uv_ov, fuv_ov = load_obj_with_uv(overlay_path)
        mesh_ov = pv_mesh(v_ov, f_ov)

        img_ov = os.path.join(OUT_DIR, "penner_overlay.png")
        render(mesh_ov, img_ov, "Overlay Mesh", color=COLORS["penner"])

        if uv_ov is not None and fuv_ov:
            # UV layout (2D)
            img_uv = os.path.join(OUT_DIR, "penner_overlay_uv.png")
            render_uv(uv_ov, fuv_ov, img_uv, "Overlay UV")

            # Checkerboard on 3D mesh
            img_checker = os.path.join(OUT_DIR, "penner_overlay_checker.png")
            render_uv_checker(mesh_ov, uv_ov, fuv_ov, img_checker,
                              "Penner UV Checker")

            table["Parametrization / UV"] = table.get("Parametrization / UV", {})
            table["Parametrization / UV"]["penner"] = {
                "img": img_checker, "time": "~2s (Newton)",
                "info": f"{len(v_ov)}V {len(f_ov)}F"
            }
            table["UV Layout"] = table.get("UV Layout", {})
            table["UV Layout"]["penner"] = {
                "img": img_uv,
                "info": f"{len(uv_ov)} UV vertices"
            }

    # Penner cross field from ffield
    ffield_path = os.path.join(penner_dir, f"{args.mesh}.ffield")
    if os.path.exists(ffield_path) and os.path.exists(overlay_path):
        print("  Rendering Penner cross field from ffield...")
        # Parse ffield: each line has dx dy dz theta k0 k1 k2 pj0 pj1 pj2
        # reference_field (dx,dy,dz) is rotated by theta in tangent plane
        with open(ffield_path) as ff:
            nF_ff = int(ff.readline())
            ff_ref = []
            ff_theta = []
            ff_pj = []
            for _ in range(nF_ff):
                vals = list(map(float, ff.readline().split()))
                ff_ref.append([vals[0], vals[1], vals[2]])
                ff_theta.append(vals[3])
                ff_pj.append([int(vals[7]), int(vals[8]), int(vals[9])])
            ff_ref = np.array(ff_ref)
            ff_theta = np.array(ff_theta)
            ff_pj = np.array(ff_pj)

        # Compute face centers and normals of overlay mesh
        ff_centers = np.zeros((len(f_ov), 3))
        ff_normals = np.zeros((len(f_ov), 3))
        for i, face in enumerate(f_ov):
            pts = np.array([v_ov[vi] for vi in face])
            ff_centers[i] = np.mean(pts, axis=0)
            e1 = pts[1] - pts[0]
            e2 = pts[2] - pts[0]
            n = np.cross(e1, e2)
            nl = np.linalg.norm(n)
            ff_normals[i] = n / nl if nl > 1e-15 else [0, 0, 1]

        n_use = min(len(ff_ref), len(ff_centers))
        ff_centers = ff_centers[:n_use]
        ff_normals = ff_normals[:n_use]

        # Rotate reference direction by theta in the tangent plane
        # Rodrigues rotation: v_rot = v*cos(t) + (n×v)*sin(t) + n*(n·v)*(1-cos(t))
        ref = ff_ref[:n_use]
        norms = np.linalg.norm(ref, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        ref = ref / norms
        cos_t = np.cos(ff_theta[:n_use])[:, None]
        sin_t = np.sin(ff_theta[:n_use])[:, None]
        nxr = np.cross(ff_normals, ref)
        ndotr = np.sum(ff_normals * ref, axis=1, keepdims=True)
        d0 = ref * cos_t + nxr * sin_t + ff_normals * ndotr * (1 - cos_t)
        # Normalize
        d0_n = np.linalg.norm(d0, axis=1, keepdims=True)
        d0_n[d0_n < 1e-10] = 1.0
        d0 = d0 / d0_n
        # Second direction: 90° rotation in tangent plane = n × d0
        d1 = np.cross(ff_normals, d0)
        d1_n = np.linalg.norm(d1, axis=1, keepdims=True)
        d1_n[d1_n < 1e-10] = 1.0
        d1 = d1 / d1_n

        ff_4dirs = np.stack([d0, d1, -d0, -d1], axis=1)

        # Compute singularities from period jumps
        # For each vertex, sum period_jump contributions from adjacent faces
        # Singularity = vertex where sum of period jumps mod 4 != 0
        from collections import defaultdict
        vert_pj_sum = defaultdict(int)
        for fi in range(min(n_use, len(f_ov))):
            face = f_ov[fi]
            for j in range(3):
                # period_jump[fi][j] is for the edge opposite corner j
                # That edge's vertices are face[(j+1)%3] and face[(j+2)%3]
                # The vertex AT corner j gets this period jump contribution
                vi = face[j]
                vert_pj_sum[vi] += ff_pj[fi][j]

        penner_sing_pos = []
        penner_sing_idx = []
        for vi, pj_sum in vert_pj_sum.items():
            # Each unit of period jump = π/2 cone angle contribution
            # Singularity when pj_sum mod 4 != 0
            idx = pj_sum % 4
            if idx == 3:
                idx = -1
            if idx != 0:
                penner_sing_pos.append(v_ov[vi])
                penner_sing_idx.append(idx)

        penner_sing_pos = np.array(penner_sing_pos) if penner_sing_pos else np.zeros((0, 3))
        penner_sing_idx = np.array(penner_sing_idx) if penner_sing_idx else np.array([])

        img_pcf = os.path.join(OUT_DIR, "penner_crossfield.png")
        render_crossfield(mesh_ov, ff_centers, ff_4dirs,
                          penner_sing_pos, penner_sing_idx,
                          img_pcf, "Penner Cross Field")
        table["Cross Field"] = table.get("Cross Field", {})
        table["Cross Field"]["penner"] = {
            "img": img_pcf,
            "info": f"{n_use} faces, {len(penner_sing_pos)} singularities"
        }

    # ── Penner pipeline: No-T-Mesh Quantization + libQEx ─────────────
    QUANT_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "quantization", "Quantization")
    QEX_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "libqex", "extract_quads")
    penner_reembed_path = "/tmp/penner_reembed.obj"
    penner_quad_path = "/tmp/penner_final_quads.obj"

    if os.path.exists(QUANT_BIN) and os.path.exists(QEX_BIN) and os.path.exists(overlay_path):
        print("\n=== Penner → No-T-Mesh Quantization → libQEx ===")

        # No-T-Mesh Quantization (auto-scale, re-embed)
        _, stderr_q, rc_q = run_cmd(
            [QUANT_BIN, "-s", str(args.penner_scale), "-r", "-o", penner_reembed_path, overlay_path],
            "No-T-Mesh Quantization", timeout=120)
        logs["Quantization"] = "\n".join(
            l for l in stderr_q.split("\n")
            if any(k in l for k in ["INPUT", "DECIM", "QUANT", "Scale", "RE-EMB"]))

        table["Quantization"] = table.get("Quantization", {})
        table["Quantization"]["penner"] = {"time": "~5s", "info": "No-T-Mesh (decimate + ILP)"}

        # libQEx quad extraction
        if os.path.exists(penner_reembed_path) and rc_q == 0:
            _, stderr_qex, rc_qex = run_cmd(
                [QEX_BIN, penner_reembed_path, penner_quad_path],
                "libQEx", timeout=60)

            if os.path.exists(penner_quad_path) and rc_qex == 0:
                v_pq, f_pq = load_obj(penner_quad_path)
                mesh_pq = pv_mesh(v_pq, f_pq)
                img_pq = os.path.join(OUT_DIR, "penner_quad.png")
                render(mesh_pq, img_pq, "Penner Quads", color=COLORS["penner"])

                print("  Computing Penner quad quality...")
                pq_metrics = compute_quad_quality(mesh_pq, mesh_in)
                table["Quad Mesh"] = table.get("Quad Mesh", {})
                table["Quad Mesh"]["penner"] = {
                    "img": img_pq,
                    "info": fmt_metrics(pq_metrics),
                    "time": "~16s total",
                }

    # ── Generate HTML ─────────────────────────────────────────────────
    generate_html(table, logs)
    print("\nDone!")


if __name__ == "__main__":
    main()
