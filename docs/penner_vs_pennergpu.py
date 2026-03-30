"""
Penner CPU vs GPU pipeline — side-by-side comparison.

For now both sides run the same CPU binary. GPU kernels will replace
pieces incrementally; this scaffold lets you verify visual match at
every step.

Usage:
    python docs/penner_vs_pennergpu.py
    python docs/penner_vs_pennergpu.py --mesh fandisk
"""

import os
import sys
import time
import argparse
import subprocess
import warnings
import collections
import numpy as np
from html import escape as html_escape

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RXMESH_INPUT = os.path.join(SCRIPT_DIR, "..", "RXMesh", "input")
PENNER_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "penner", "bin", "parameterize_aligned")
QUANT_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "quantization", "Quantization")
QEX_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "libqex", "extract_quads")
OUT_DIR = os.path.join(SCRIPT_DIR, "_site", "penner_vs_pennergpu")

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
EDGE_COLOR = "#30363d"
FEAT_COLOR = "#f85149"

os.makedirs(OUT_DIR, exist_ok=True)


# ── Loaders ──────────────────────────────────────────────────────────

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


def load_obj_with_uv(path):
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
    return np.array(verts), faces, np.array(uvs) if uvs else None, face_uvs if face_uvs else None


def pv_mesh(verts, faces):
    import pyvista as pv
    pv_faces = []
    for face in faces:
        pv_faces.append(len(face))
        pv_faces.extend(face)
    return pv.PolyData(verts, np.array(pv_faces, dtype=np.int32))


# ── Rendering ────────────────────────────────────────────────────────

def render(mesh, filename, title, color="#58a6ff", edge_lines=None,
           scalars=None, cmap="viridis", clim=None, window_size=(800, 600)):
    import pyvista as pv
    pv.OFF_SCREEN = True

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    kwargs = dict(show_edges=True, edge_color=EDGE_COLOR,
                  line_width=0.3, lighting=True, smooth_shading=True)
    if scalars is not None:
        pl.add_mesh(mesh, scalars=scalars, cmap=cmap, clim=clim,
                    show_scalar_bar=False, **kwargs)
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

    pl.add_text(title, position="upper_left", font_size=10, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"    rendered {os.path.basename(filename)}")


def render_uv_checker(mesh, uvs, face_uvs, filename, title, window_size=(800, 600)):
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
    pl.add_text(title, position="upper_left", font_size=10, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()
    print(f"    rendered {os.path.basename(filename)}")


# ── Quality metrics ──────────────────────────────────────────────────

def compute_quad_quality(mesh, input_mesh=None):
    import pyvista as pv
    clean = pv.PolyData(mesh.points.copy(), mesh.faces.copy())
    try:
        qual = clean.compute_cell_quality(quality_measure='scaled_jacobian')
        sj = qual['CellQuality']
        sj_min, sj_avg = float(sj.min()), float(sj.mean())
    except Exception:
        sj_min, sj_avg = 0.0, 0.0

    npf = mesh.faces[0]
    faces_arr = mesh.faces.reshape(-1, npf + 1)[:, 1:]
    val_count = np.zeros(mesh.n_points, dtype=int)
    for face in faces_arr:
        for vi in face:
            val_count[vi] += 1
    val_hist = collections.Counter(val_count.tolist())
    pct_regular = 100.0 * val_hist.get(4, 0) / max(mesh.n_points, 1)
    n_irregular = sum(v for k, v in val_hist.items() if k != 4 and k > 0)

    metrics = {
        "V": mesh.n_points, "F": mesh.n_cells,
        "sj_min": sj_min, "sj_avg": sj_avg,
        "pct_regular": pct_regular, "n_irregular": n_irregular,
    }

    if input_mesh is not None:
        try:
            def sample_surface(m, n=10000):
                fa = m.faces.reshape(-1, m.faces[0] + 1)[:, 1:]
                pts = m.points
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

            np.random.seed(42)
            rs = sample_surface(mesh, 10000)
            ins = sample_surface(input_mesh, 10000)
            _, fc = input_mesh.find_closest_cell(rs, return_closest_point=True)
            _, bc = mesh.find_closest_cell(ins, return_closest_point=True)
            fd = np.linalg.norm(rs - fc, axis=1)
            bd = np.linalg.norm(ins - bc, axis=1)
            metrics["hausdorff_avg"] = float(max(fd.mean(), bd.mean()))
            metrics["hausdorff_max"] = float(max(fd.max(), bd.max()))
        except Exception:
            metrics["hausdorff_avg"] = -1
            metrics["hausdorff_max"] = -1

    return metrics


def fmt_metrics(m):
    parts = [f"{m['V']}V {m['F']}Q"]
    parts.append(f"SJ min={m['sj_min']:.3f} avg={m['sj_avg']:.3f}")
    parts.append(f"val4={m['pct_regular']:.1f}% ({m['n_irregular']} irreg)")
    if 'hausdorff_avg' in m and m['hausdorff_avg'] >= 0:
        parts.append(f"Haus avg={m['hausdorff_avg']:.2e} max={m['hausdorff_max']:.2e}")
    return " | ".join(parts)


# ── Pipeline runner ──────────────────────────────────────────────────

def run_cmd(cmd, label, timeout_s=300):
    print(f"  Running {label}...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    dt = time.time() - t0
    print(f"  {label}: {dt:.1f}s (exit={r.returncode})")
    return r.stdout, r.stderr, r.returncode, dt


def run_penner_pipeline(mesh_name, work_dir, label, extra_args=None):
    """Run full Penner pipeline (parameterize → quantize → extract quads).
    Returns dict of paths and timings."""
    os.makedirs(work_dir, exist_ok=True)
    result = {"label": label, "work_dir": work_dir}

    # Step 1: Penner parameterization
    cmd = [PENNER_BIN, "--name", mesh_name, "-i", RXMESH_INPUT, "-o", work_dir]
    if extra_args:
        cmd.extend(extra_args)
    stdout, stderr, rc, dt = run_cmd(cmd, f"{label} parameterize", timeout_s=300)
    result["penner_time"] = dt
    result["penner_rc"] = rc
    result["penner_log"] = ""
    for line in (stdout + stderr).split("\n"):
        if any(k in line for k in ["Feature edge", "prune_small", "erode_dilate",
                                    "itr(", "Stopping", "Refined", "Cut mesh", "component"]):
            result["penner_log"] += line.strip() + "\n"

    # Intermediate files
    result["refined"] = os.path.join(work_dir, mesh_name + "_refined.obj")
    result["cut"] = os.path.join(work_dir, mesh_name + "_cut.obj")
    result["opt"] = os.path.join(work_dir, mesh_name + "_opt.obj")

    # Step 2: Quantization
    reembed = os.path.join(work_dir, "reembed.obj")
    result["reembed"] = reembed
    if rc == 0 and os.path.exists(result["opt"]):
        _, _, rc_q, dt_q = run_cmd(
            [QUANT_BIN, "-s", "a", "-sa", "3", "-r", "-o", reembed, result["opt"]],
            f"{label} quantization", timeout_s=300)
        result["quant_time"] = dt_q
        result["quant_rc"] = rc_q
    else:
        result["quant_time"] = 0
        result["quant_rc"] = -1

    # Step 3: Quad extraction
    quads = os.path.join(work_dir, "quads.obj")
    result["quads"] = quads
    if os.path.exists(reembed):
        _, _, rc_qex, dt_qex = run_cmd(
            [QEX_BIN, reembed, quads], f"{label} libQEx", timeout_s=60)
        result["qex_time"] = dt_qex
        result["qex_rc"] = rc_qex
    else:
        result["qex_time"] = 0
        result["qex_rc"] = -1

    result["total_time"] = result["penner_time"] + result["quant_time"] + result["qex_time"]
    return result


# ── Main ─────────────────────────────────────────────────────────────

COLUMNS = ["cpu", "gpu"]
COL_LABELS = {"cpu": "Penner CPU", "gpu": "Penner GPU"}
COL_COLORS = {"cpu": "#58a6ff", "gpu": "#3fb950"}
ROW_ORDER = [
    "Input Mesh",
    "Feature Detection",
    "Cut Mesh",
    "UV (checker)",
    "Quantization",
    "Quad Mesh",
    "Quality",
]


def main():
    parser = argparse.ArgumentParser(description="Penner CPU vs GPU comparison")
    parser.add_argument("--mesh", default="dragon")
    parser.add_argument("--sa", default="3", help="Quantization auto-scale")
    args = parser.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True

    mesh_path = os.path.join(RXMESH_INPUT, args.mesh + ".obj")
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        sys.exit(1)

    table = {}  # step → {col → cell_data}
    logs = {}

    # ── Input mesh ───────────────────────────────────────────────────
    print("=== Input Mesh ===")
    v_in, f_in, _ = load_obj(mesh_path)
    mesh_in = pv_mesh(v_in, f_in)
    img_in = os.path.join(OUT_DIR, "input.png")
    render(mesh_in, img_in, f"Input: {args.mesh}")
    table["Input Mesh"] = {
        col: {"img": img_in, "info": f"{len(v_in)}V {len(f_in)}F"}
        for col in COLUMNS
    }

    # ── Run both pipelines ───────────────────────────────────────────
    cpu_work = os.path.join(OUT_DIR, "_work_cpu")
    gpu_work = os.path.join(OUT_DIR, "_work_gpu")

    print("\n=== Penner CPU ===")
    cpu = run_penner_pipeline(args.mesh, cpu_work, "CPU")
    logs["Penner CPU"] = cpu.get("penner_log", "")

    print("\n=== Penner GPU (CPU copy for now) ===")
    gpu = run_penner_pipeline(args.mesh, gpu_work, "GPU")
    logs["Penner GPU"] = gpu.get("penner_log", "")

    # ── Render comparison rows ───────────────────────────────────────
    print("\n=== Rendering ===")
    for col, res in [("cpu", cpu), ("gpu", gpu)]:
        color = COL_COLORS[col]
        prefix = col

        # Feature detection
        if os.path.exists(res["refined"]):
            v, f, lines = load_obj(res["refined"])
            m = pv_mesh(v, f)
            img = os.path.join(OUT_DIR, f"{prefix}_features.png")
            render(m, img, f"{COL_LABELS[col]}: Features", color="#404040", edge_lines=lines)
            table.setdefault("Feature Detection", {})[col] = {
                "img": img, "info": f"{len(v)}V {len(f)}F | {len(lines)} features"
            }

        # Cut mesh
        if os.path.exists(res["cut"]):
            v, f, lines = load_obj(res["cut"])
            m = pv_mesh(v, f)
            img = os.path.join(OUT_DIR, f"{prefix}_cut.png")
            render(m, img, f"{COL_LABELS[col]}: Cut", color="#404040", edge_lines=lines)
            table.setdefault("Cut Mesh", {})[col] = {
                "img": img, "info": f"{len(v)}V | {len(lines)} boundary"
            }

        # UV checker
        if os.path.exists(res["opt"]):
            v, f, uv, fuv = load_obj_with_uv(res["opt"])
            if uv is not None and fuv:
                m = pv_mesh(v, f)
                img = os.path.join(OUT_DIR, f"{prefix}_uv.png")
                render_uv_checker(m, uv, fuv, img, f"{COL_LABELS[col]}: UV")
                table.setdefault("UV (checker)", {})[col] = {
                    "img": img, "time": f"{res['penner_time']:.1f}s",
                    "info": f"{len(v)}V {len(f)}F | {len(uv)} UV"
                }

        # Quantization
        if os.path.exists(res["reembed"]):
            v, f, uv, fuv = load_obj_with_uv(res["reembed"])
            if uv is not None and fuv:
                m = pv_mesh(v, f)
                img = os.path.join(OUT_DIR, f"{prefix}_quant.png")
                render_uv_checker(m, uv, fuv, img, f"{COL_LABELS[col]}: Quantized")
                table.setdefault("Quantization", {})[col] = {
                    "img": img, "time": f"{res['quant_time']:.1f}s",
                    "info": f"{len(v)}V {len(f)}F"
                }

        # Quad mesh
        if os.path.exists(res["quads"]):
            v, f, _ = load_obj(res["quads"])
            m = pv_mesh(v, f)
            img = os.path.join(OUT_DIR, f"{prefix}_quads.png")
            render(m, img, f"{COL_LABELS[col]}: Quads", color=color)

            metrics = compute_quad_quality(m, mesh_in)
            table.setdefault("Quad Mesh", {})[col] = {
                "img": img, "time": f"{res['total_time']:.1f}s total",
                "info": fmt_metrics(metrics)
            }

            # Quality row (text only)
            table.setdefault("Quality", {})[col] = {
                "info": (f"SJ avg={metrics['sj_avg']:.4f} min={metrics['sj_min']:.4f}\n"
                         f"val4={metrics['pct_regular']:.1f}% | {metrics['n_irregular']} irreg\n"
                         f"Haus avg={metrics.get('hausdorff_avg', -1):.2e}")
            }

    # ── Generate HTML ────────────────────────────────────────────────
    cache = int(time.time())
    header = "<tr><th>Step</th>"
    for col in COLUMNS:
        header += f"<th>{COL_LABELS[col]}</th>"
    header += "</tr>"

    rows_html = ""
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
                td += f'<img src="{rel}?v={cache}" />'
            if timing:
                td += f'<div class="timing">{html_escape(timing)}</div>'
            if info:
                td += f'<div class="info">{html_escape(info)}</div>'
            if not img and not timing and not info:
                td += "&mdash;"
            td += "</td>"
            row += td
        row += "</tr>"
        rows_html += row

    log_html = ""
    for name, log in logs.items():
        if log:
            log_html += f"<h3>{html_escape(name)}</h3><pre>{html_escape(log)}</pre>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Penner CPU vs GPU</title>
<style>
body {{ background: {BG_COLOR}; color: {TEXT_COLOR}; font-family: -apple-system, monospace; padding: 20px; }}
h1 {{ color: #58a6ff; }}
h3 {{ color: #d29922; margin-top: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ background: #161b22; color: #58a6ff; padding: 10px; text-align: center; border: 1px solid #30363d; }}
td {{ padding: 8px; border: 1px solid #30363d; text-align: center; vertical-align: top; }}
td.step {{ text-align: left; font-weight: bold; color: #d29922; white-space: nowrap; width: 140px; }}
td img {{ max-width: 100%; border-radius: 4px; }}
.timing {{ color: #3fb950; font-size: 13px; margin-top: 4px; }}
.info {{ color: #8b949e; font-size: 12px; margin-top: 2px; white-space: pre-line; }}
pre {{ background: #161b22; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 12px; max-height: 300px; }}
</style></head><body>
<h1>Penner CPU vs GPU — {args.mesh}</h1>
<p>Side-by-side comparison. GPU column is currently a CPU copy — kernels will be swapped in incrementally.</p>
<table>
{header}
{rows_html}
</table>
{log_html}
</body></html>"""

    html_path = os.path.join(OUT_DIR, "index.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nHTML: {html_path}")


if __name__ == "__main__":
    main()
