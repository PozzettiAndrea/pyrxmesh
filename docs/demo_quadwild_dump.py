"""
QuadWild pipeline step-by-step with full intermediate dumps.
Uses pyrxmesh's CPU reimplementation of each QuadWild preprocessing step.

Steps:
  1. Feature detection (dihedral angle)
  2. Erode/dilate
  3. Isotropic remeshing (15 iterations)
  4. Micro collapse
  5. Re-detect features
  6. Adaptive remeshing
  7. Clean mesh
  8. Refine (split faces with 3 sharp edges)
  9. Cross field + trace + quadrangulate (QuadWild binary)

Usage:
    python docs/demo_quadwild_dump.py
    python docs/demo_quadwild_dump.py --mesh dragon
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RXMESH_INPUT = os.path.join(SCRIPT_DIR, "..", "RXMesh", "input")
QUADWILD_BIN = os.path.join(SCRIPT_DIR, "..", "build", "extern", "quadwild", "Build", "bin", "quadwild")
QUADWILD_LIB = os.path.join(SCRIPT_DIR, "..", "build", "extern", "quadwild", "Build", "lib")
QUADWILD_CFG = os.path.join(SCRIPT_DIR, "..", "extern", "quadwild", "quadwild")
OUT_DIR = os.path.join(SCRIPT_DIR, "_site", "demo_quadwild_dump")

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
EDGE_COLOR = "#30363d"
FEAT_COLOR = "#f85149"

os.makedirs(OUT_DIR, exist_ok=True)


def pv_mesh(verts, faces):
    import pyvista as pv
    n = len(faces)
    pv_faces = np.column_stack([np.full(n, 3, dtype=np.int32), faces]).ravel()
    return pv.PolyData(verts, pv_faces)


def render(mesh, filename, title, color="#58a6ff", show_edges=True,
           scalars=None, cmap="viridis", clim=None, window_size=(1200, 900)):
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

    pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
    pl.add_text(subtitle, position="upper_right", font_size=9, color="#8b949e")
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    try:
        pl.export_html(filename.replace(".png", "_3d.html"))
    except Exception:
        pass
    pl.close()
    print(f"  rendered {os.path.basename(filename)}")


def render_features(v, f, feat, filename, title):
    mesh = pv_mesh(v, f)
    mesh.point_data["feature"] = feat.vertex_is_feature.astype(float)
    render(mesh, filename, f"{title} ({feat.num_feature_edges} features)",
           scalars="feature", cmap=["#404040", FEAT_COLOR], clim=[0, 1])


def render_quality(v, f, filename, title):
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
           clim=[1, min(float(q.max()), 10)])


def main():
    parser = argparse.ArgumentParser(description="QuadWild pipeline with full dumps")
    parser.add_argument("--mesh", default="dragon")
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()

    import pyrxmesh
    pyrxmesh.init()

    mesh_path = os.path.join(RXMESH_INPUT, args.mesh + ".obj")
    v, f = pyrxmesh.load_obj(mesh_path)
    print(f"Mesh: {args.mesh} ({len(v)}V, {len(f)}F)")

    rows = []

    # ── Step 0: Input ─────────────────────────────────────────────
    print("\n=== Step 0: Input ===")
    img = os.path.join(OUT_DIR, "step0_input.png")
    render(pv_mesh(v, f), img, f"Input ({args.mesh})")
    rows.append(("Input", img, "", f"{len(v)}V {len(f)}F"))

    # ── Step 1: Raw feature detection ─────────────────────────────
    print("\n=== Step 1: Feature Detection ===")
    t0 = time.time()
    feat_raw = pyrxmesh.detect_features(v, f, crease_angle_deg=35.0, erode_dilate_steps=0)
    t = time.time() - t0
    img = os.path.join(OUT_DIR, "step1_features_raw.png")
    render_features(v, f, feat_raw, img, "Step 1: Raw Features")
    rows.append(("Raw Features", img, f"{t*1000:.0f}ms",
                 f"{feat_raw.num_feature_edges} feature edges"))

    # ── Step 2: Erode/Dilate ──────────────────────────────────────
    print("\n=== Step 2: Erode/Dilate ===")
    t0 = time.time()
    feat_ed = pyrxmesh.detect_features(v, f, crease_angle_deg=35.0, erode_dilate_steps=4)
    t = time.time() - t0
    img = os.path.join(OUT_DIR, "step2_erode_dilate.png")
    render_features(v, f, feat_ed, img, "Step 2: After Erode/Dilate")
    rows.append(("Erode/Dilate", img, f"{t*1000:.0f}ms",
                 f"{feat_raw.num_feature_edges} → {feat_ed.num_feature_edges} features"))

    # ── Step 3: Isotropic Remeshing (CPU) ─────────────────────────
    print("\n=== Step 3: Isotropic Remeshing ===")
    t0 = time.time()
    rv, rf = pyrxmesh.vcg_remesh(v, f, iterations=args.iterations,
                                  crease_angle_deg=35.0, verbose=False)
    t = time.time() - t0
    img = os.path.join(OUT_DIR, "step3_remesh.png")
    render_quality(rv, rf, img, f"Step 3: Isotropic Remesh ({args.iterations} iters)")
    rows.append(("Isotropic Remesh", img, f"{t:.1f}s",
                 f"{len(v)}V → {len(rv)}V  |  {len(f)}F → {len(rf)}F"))

    # ── Step 4: Micro Collapse ────────────────────────────────────
    print("\n=== Step 4: Micro Collapse ===")
    t0 = time.time()
    mv, mf = pyrxmesh.vcg_micro_collapse(rv, rf)
    t = time.time() - t0
    img = os.path.join(OUT_DIR, "step4_micro.png")
    render_quality(mv, mf, img, "Step 4: Micro Collapse")
    rows.append(("Micro Collapse", img, f"{t*1000:.0f}ms",
                 f"{len(rv)}V → {len(mv)}V"))

    # ── Step 5: Re-detect Features ────────────────────────────────
    print("\n=== Step 5: Re-detect Features ===")
    t0 = time.time()
    feat2 = pyrxmesh.detect_features(mv, mf, crease_angle_deg=35.0, erode_dilate_steps=4)
    t = time.time() - t0
    img = os.path.join(OUT_DIR, "step5_redetect.png")
    render_features(mv, mf, feat2, img, "Step 5: Re-detect Features")
    rows.append(("Re-detect Features", img, f"{t*1000:.0f}ms",
                 f"{feat2.num_feature_edges} features"))

    # ── Step 6: Adaptive Remeshing ────────────────────────────────
    print("\n=== Step 6: Adaptive Remeshing ===")
    t0 = time.time()
    av, af = pyrxmesh.vcg_remesh_adaptive(mv, mf, iterations=args.iterations,
                                           crease_angle_deg=35.0, verbose=False)
    t = time.time() - t0
    img = os.path.join(OUT_DIR, "step6_adaptive.png")
    render_quality(av, af, img, "Step 6: Adaptive Remesh")
    rows.append(("Adaptive Remesh", img, f"{t:.1f}s",
                 f"{len(mv)}V → {len(av)}V  |  {len(mf)}F → {len(af)}F"))

    # ── Step 7: Clean Mesh ────────────────────────────────────────
    print("\n=== Step 7: Clean Mesh ===")
    t0 = time.time()
    cv, cf = pyrxmesh.vcg_clean_mesh(av, af)
    t = time.time() - t0
    img = os.path.join(OUT_DIR, "step7_clean.png")
    render_quality(cv, cf, img, "Step 7: Clean Mesh")
    rows.append(("Clean", img, f"{t*1000:.0f}ms",
                 f"{len(av)}V → {len(cv)}V"))

    # ── Step 8: Refine ────────────────────────────────────────────
    print("\n=== Step 8: Refine ===")
    t0 = time.time()
    xv, xf = pyrxmesh.vcg_refine_if_needed(cv, cf, crease_angle_deg=35.0)
    t = time.time() - t0
    img = os.path.join(OUT_DIR, "step8_refine.png")
    render_quality(xv, xf, img, "Step 8: Refine")
    rows.append(("Refine", img, f"{t*1000:.0f}ms",
                 f"{len(cv)}V → {len(xv)}V  |  {len(cf)}F → {len(xf)}F"))

    # ── Step 9: QuadWild (cross field → trace → quad) ─────────────
    if os.path.exists(QUADWILD_BIN):
        work_dir = os.path.join(OUT_DIR, "_work")
        os.makedirs(work_dir, exist_ok=True)

        # Save preprocessed mesh
        preproc_obj = os.path.join(work_dir, args.mesh + ".obj")
        with open(preproc_obj, "w") as fout:
            for vi in range(len(xv)):
                fout.write(f"v {xv[vi,0]} {xv[vi,1]} {xv[vi,2]}\n")
            for fi in range(len(xf)):
                fout.write(f"f {xf[fi,0]+1} {xf[fi,1]+1} {xf[fi,2]+1}\n")

        for cfg in ["basic_setup.txt", "basic_setup_organic.txt", "basic_setup_mechanical.txt"]:
            src = os.path.join(QUADWILD_CFG, cfg)
            if os.path.exists(src): shutil.copy(src, work_dir)

        print("\n=== Step 9: QuadWild (cross field → trace → quad) ===")
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = QUADWILD_LIB + ":" + env.get("LD_LIBRARY_PATH", "")
        t0 = time.time()
        r = subprocess.run([QUADWILD_BIN, preproc_obj, "3"],
                           capture_output=True, text=True, timeout=600,
                           cwd=work_dir, env=env)
        t = time.time() - t0
        for line in r.stdout.split("\n"):
            if line.strip(): print(f"    {line.strip()}")

        # Render quad mesh
        quad_path = os.path.join(work_dir, args.mesh + "_rem_quadrangulation_smooth.obj")
        if not os.path.exists(quad_path):
            quad_path = os.path.join(work_dir, args.mesh + "_rem_quadrangulation.obj")
        if os.path.exists(quad_path):
            qv, qf = [], []
            with open(quad_path) as qfile:
                for line in qfile:
                    if line.startswith("v "):
                        qv.append([float(x) for x in line.split()[1:4]])
                    elif line.startswith("f "):
                        qf.append([int(p.split("/")[0])-1 for p in line.split()[1:]])
            qv = np.array(qv)
            import pyvista as pv
            pv_faces = []
            for face in qf:
                pv_faces.append(len(face))
                pv_faces.extend(face)
            qmesh = pv.PolyData(qv, np.array(pv_faces, dtype=np.int32))
            img = os.path.join(OUT_DIR, "step9_quads.png")
            render(qmesh, img, "Step 9: Final Quad Mesh", color="#3fb950")
            rows.append(("Quad Mesh", img, f"{t:.0f}s",
                         f"{len(qv)}V {len(qf)} quads"))

    # ── Generate HTML ─────────────────────────────────────────────
    cache = int(time.time())
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>QuadWild Pipeline (Full Dump)</title>
<style>
body {{ background: {BG_COLOR}; color: {TEXT_COLOR}; font-family: monospace; padding: 20px; max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #58a6ff; }}
table {{ border-collapse: collapse; width: 100%; }}
th {{ background: #161b22; color: #58a6ff; padding: 10px; text-align: left; border: 1px solid #30363d; }}
td {{ padding: 8px; border: 1px solid #30363d; vertical-align: top; }}
td.step {{ font-weight: bold; color: #d29922; width: 180px; }}
td img {{ width: 100%; border-radius: 4px; cursor: pointer; }}
.timing {{ color: #3fb950; }}
.info {{ color: #8b949e; font-size: 13px; }}
.btn3d {{ display: inline-block; padding: 4px 10px; background: #238636; color: white; border-radius: 4px; text-decoration: none; font-size: 12px; margin-top: 4px; }}
.btn3d:hover {{ background: #2ea043; }}
</style></head><body>
<h1>QuadWild Pipeline (Full Dump) — {args.mesh}</h1>
<p>Features → Erode/Dilate → Isotropic Remesh → Micro Collapse → Re-detect → Adaptive Remesh → Clean → Refine → Cross Field → Trace → Quads</p>
<table>
<tr><th>Step</th><th>Result</th><th>Time</th><th>Info</th></tr>
"""
    for step_name, img_path, timing, info in rows:
        rel = os.path.relpath(img_path, OUT_DIR)
        rel_3d = rel.replace(".png", "_3d.html")
        has_3d = os.path.exists(os.path.join(OUT_DIR, rel_3d))
        t_html = f'<span class="timing">{timing}</span>' if timing else ""
        btn_3d = f'<a class="btn3d" href="{rel_3d}" target="_blank">3D</a>' if has_3d else ""
        html += f"""<tr>
<td class="step">{step_name}</td>
<td><img src="{rel}?v={cache}" />{btn_3d}</td>
<td>{t_html}</td>
<td class="info">{info}</td>
</tr>\n"""

    html += "</table></body></html>"
    html_path = os.path.join(OUT_DIR, "index.html")
    with open(html_path, "w") as fh:
        fh.write(html)
    print(f"\nHTML: {html_path}")
    print(f"View: http://localhost:8001/demo_quadwild_dump/index.html")


if __name__ == "__main__":
    main()
