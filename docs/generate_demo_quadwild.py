"""
Generate QuadWild pipeline CPU vs GPU comparison demo.

Usage:
    python docs/generate_demo_quadwild.py --help
    python docs/generate_demo_quadwild.py --only erode
    python docs/generate_demo_quadwild.py --only remesh
    python docs/generate_demo_quadwild.py --all
"""

import os
import sys
import shutil
import argparse
import time
import warnings
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*CellQuality.*")

import pyrxmesh

# Import shared utilities from generate_demo
from generate_demo import (
    OUT_DIR, RXMESH_INPUT, RUNS_DIR,
    BG_COLOR, MESH_COLOR_IN, MESH_COLOR_OUT, EDGE_COLOR, TEXT_COLOR,
    pv_mesh_from_numpy, render_mesh, run_before_after,
    setup_logging, generate_html,
)

def compute_mesh_quality(mesh, input_mesh=None, label=""):
    """Compute quality metrics for a triangle mesh. Returns dict of metrics."""
    import collections

    # Triangle quality (aspect ratio)
    # Use a fresh copy so existing scalars don't interfere
    clean = pv.PolyData(mesh.points.copy(), mesh.faces.copy())
    qual = clean.compute_cell_quality(quality_measure='aspect_ratio')
    q = qual['CellQuality']

    # Edge lengths
    edges = mesh.extract_all_edges()
    pts = mesh.points
    edge_lens = []
    lines = edges.lines.reshape(-1, 3)
    for ln in lines:
        v0, v1 = ln[1], ln[2]
        edge_lens.append(np.linalg.norm(pts[v0] - pts[v1]))
    edge_lens = np.array(edge_lens)

    # Vertex valence
    faces_arr = mesh.faces.reshape(-1, 4)[:, 1:]
    val_count = np.zeros(mesh.n_points, dtype=int)
    for face in faces_arr:
        for vi in face:
            val_count[vi] += 1
    val_hist = collections.Counter(val_count.tolist())
    pct_val6 = 100 * val_hist.get(6, 0) / max(mesh.n_points, 1)

    metrics = {
        "V": mesh.n_points, "F": mesh.n_cells,
        "quality_min": float(q.min()), "quality_avg": float(q.mean()), "quality_max": float(q.max()),
        "edge_min": float(edge_lens.min()), "edge_avg": float(edge_lens.mean()),
        "edge_max": float(edge_lens.max()), "edge_std": float(edge_lens.std()),
        "pct_val6": pct_val6,
        "val_mode": val_hist.most_common(1)[0],
    }

    # Distance to original mesh
    if input_mesh is not None:
        closest = input_mesh.find_closest_cell(mesh.points)
        closest_pts = input_mesh.cell_centers().points[closest]
        dists = np.linalg.norm(mesh.points - closest_pts, axis=1)
        metrics["dist_avg"] = float(dists.mean())
        metrics["dist_max"] = float(dists.max())

    return metrics


def compute_quality_for_step(step_name, input_mesh):
    """Compute and print quality metrics for both CPU and GPU outputs of a step."""
    print(f"  Quality metrics: {step_name}")
    results = {}
    for side in ["cpu", "gpu"]:
        vtk_path = os.path.join(OUT_DIR, f"qw_{side}_{step_name}_after.vtk")
        if os.path.exists(vtk_path):
            mesh = pv.read(vtk_path)
            results[side] = compute_mesh_quality(mesh, input_mesh, label=side)
    return results


def attach_metrics_to_demos(demos, input_mesh):
    """Attach quality metrics to all demos that have VTK output files."""
    to_compute = [d for d in demos if "metrics" not in d
                  and os.path.exists(os.path.join(OUT_DIR, f"{d['name']}_after.vtk"))]
    if not to_compute:
        return
    print(f"  Computing quality metrics ({len(to_compute)} meshes)...", end="", flush=True)
    for i, d in enumerate(to_compute):
        vtk_path = os.path.join(OUT_DIR, f"{d['name']}_after.vtk")
        try:
            m = pv.read(vtk_path)
            d["metrics"] = compute_mesh_quality(m, input_mesh, label=f"{d.get('side','')}")
        except Exception as e:
            print(f"\n    [warn] metrics failed for {d['name']}: {e}", end="")
    print(" done.")


def gen_quadwild(dragon_v, dragon_f, stop_after=None):
    """Generate QuadWild pipeline 3-column comparison: cpu_orig, cpu_ours, gpu.

    Each pipeline column is truly sequential -- step N output feeds step N+1.
    Table has 3 result columns (one image per cell, no before/after).

    Args:
        stop_after: "features", "erode", "remesh_isotropic", "micro",
                    "remesh_adaptive", or None (runs everything).
    """
    qw_all = []
    input_mesh = pv_mesh_from_numpy(dragon_v, dragon_f)

    # -- Helpers (rendering) --

    def render_gpu_features(verts, faces, feature_data, out_png, title):
        n = len(faces)
        pv_faces = np.column_stack([np.full(n, 3, dtype=np.int32), faces]).ravel()
        mesh = pv.PolyData(verts, pv_faces)
        nE = mesh.extract_all_edges().n_cells

        # Build edge index -> vertex pair mapping from faces
        edge_map = {}
        eidx = 0
        seen = set()
        for fi in range(len(faces)):
            for j in range(3):
                v0, v1 = int(faces[fi][j]), int(faces[fi][(j+1) % 3])
                key = (min(v0, v1), max(v0, v1))
                if key not in seen:
                    seen.add(key)
                    edge_map[key] = eidx
                    eidx += 1

        edge_is_feat = feature_data.edge_is_feature
        feat_pairs = []
        for (v0, v1), ei in edge_map.items():
            if ei < len(edge_is_feat) and edge_is_feat[ei]:
                feat_pairs.append([v0, v1])
        n_feat = len(feat_pairs)

        print(f"    GPU features: {len(verts)}V, {nE}E, {n}F, "
              f"{n_feat} feature edges (reported: {feature_data.num_feature_edges})")

        subtitle = f"{len(verts):,}V  {nE:,}E  {n:,}F  |  {n_feat} feature edges"

        vtk_path = out_png.replace(".png", ".vtk")
        mesh.cell_data["face_id"] = np.arange(n)
        mesh.point_data["vertex_is_feature"] = feature_data.vertex_is_feature
        mesh.save(vtk_path)
        if feat_pairs:
            ea = np.array(feat_pairs)
            lines = np.column_stack([np.full(len(ea), 2), ea]).ravel()
            edge_mesh = pv.PolyData(verts, lines=lines)
            edge_mesh.save(vtk_path.replace(".vtk", "_edges.vtk"))

        pl = pv.Plotter(off_screen=True, window_size=(800, 600))
        pl.add_mesh(mesh, color=MESH_COLOR_IN,
                    show_edges=True, edge_color=EDGE_COLOR, line_width=0.3,
                    lighting=True, smooth_shading=True,
                    opacity=0.85 if feat_pairs else 1.0)
        if feat_pairs:
            ea = np.array(feat_pairs)
            lines = np.column_stack([np.full(len(ea), 2), ea]).ravel()
            edge_mesh = pv.PolyData(verts, lines=lines)
            pl.add_mesh(edge_mesh, color="red", line_width=4)
        pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
        pl.add_text(subtitle, position="upper_right", font_size=10, color="#8b949e")
        pl.set_background(BG_COLOR)
        pl.camera_position = "iso"
        pl.screenshot(out_png, transparent_background=False)
        pl.close()
        return mesh, n_feat

    def parse_sharp(sharp_path):
        edges = []
        if not sharp_path or not os.path.exists(sharp_path):
            return edges
        with open(sharp_path) as f:
            n = int(f.readline().strip())
            for _ in range(n):
                parts = f.readline().strip().split(",")
                if len(parts) >= 3:
                    edges.append((int(parts[1]), int(parts[2])))
        return edges

    def render_with_features(obj_path, sharp_path, out_png, title):
        mesh = pv.read(obj_path)
        nE = mesh.extract_all_edges().n_cells
        sharp_edges = parse_sharp(sharp_path)

        unique_edges = set()
        edge_pairs = []
        if len(sharp_edges) > 0:
            faces_arr = mesh.faces.reshape(-1, 4)
            for fidx, eidx in sharp_edges:
                if fidx < len(faces_arr):
                    v0 = int(faces_arr[fidx][1 + eidx])
                    v1 = int(faces_arr[fidx][1 + (eidx + 1) % 3])
                    key = (min(v0, v1), max(v0, v1))
                    if key not in unique_edges:
                        unique_edges.add(key)
                        edge_pairs.append([v0, v1])
        n_feat = len(unique_edges)

        print(f"    CPU features: {mesh.n_points}V, {nE}E, {mesh.n_cells}F, "
              f"{n_feat} feature edges ({len(sharp_edges)} half-edges)")

        subtitle = f"{mesh.n_points:,}V  {nE:,}E  {mesh.n_cells:,}F  |  {n_feat} feature edges"

        vtk_path = out_png.replace(".png", ".vtk")
        mesh.save(vtk_path)
        if edge_pairs:
            ea = np.array(edge_pairs)
            lines = np.column_stack([np.full(len(ea), 2), ea]).ravel()
            edge_mesh = pv.PolyData(mesh.points, lines=lines)
            edge_mesh.save(vtk_path.replace(".vtk", "_edges.vtk"))

        pl = pv.Plotter(off_screen=True, window_size=(800, 600))
        pl.add_mesh(mesh, color=MESH_COLOR_IN,
                    show_edges=True, edge_color=EDGE_COLOR, line_width=0.3,
                    lighting=True, smooth_shading=True,
                    opacity=0.85 if n_feat > 0 else 1.0)
        if edge_pairs:
            ea = np.array(edge_pairs)
            lines = np.column_stack([np.full(len(ea), 2), ea]).ravel()
            edge_mesh = pv.PolyData(mesh.points, lines=lines)
            pl.add_mesh(edge_mesh, color="red", line_width=4)
        pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
        pl.add_text(subtitle, position="upper_right", font_size=10, color="#8b949e")
        pl.set_background(BG_COLOR)
        pl.camera_position = "iso"
        pl.screenshot(out_png, transparent_background=False)
        pl.close()
        return mesh, n_feat

    def render_crossfield(obj_path, rosy_path, out_png, title):
        mesh = pv.read(obj_path)
        with open(rosy_path) as f:
            nf = int(f.readline().strip())
            symm = int(f.readline().strip())
            dirs = []
            for _ in range(nf):
                vals = list(map(float, f.readline().strip().split()))
                dirs.append(vals[:3])
        dirs = np.array(dirs)
        centers = mesh.cell_centers().points
        stride = max(1, len(centers) // 2000)
        idx = np.arange(0, min(len(centers), len(dirs)), stride)
        avg_edge = np.mean([np.linalg.norm(
            mesh.points[mesh.faces.reshape(-1,4)[i,1]] -
            mesh.points[mesh.faces.reshape(-1,4)[i,2]])
            for i in range(min(100, mesh.n_cells))])
        scale = avg_edge * 1.2
        pts = centers[idx]
        vecs = dirs[idx]
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)
        am1 = pv.PolyData(pts)
        am1["vectors"] = vecs * scale
        arrows1 = am1.glyph(orient="vectors", scale=False, factor=scale)
        fnormals = mesh.cell_normals[idx]
        vecs90 = np.cross(fnormals, vecs)
        vecs90 = vecs90 / (np.linalg.norm(vecs90, axis=1, keepdims=True) + 1e-10)
        am2 = pv.PolyData(pts)
        am2["vectors"] = vecs90 * scale
        arrows2 = am2.glyph(orient="vectors", scale=False, factor=scale)
        pl = pv.Plotter(off_screen=True, window_size=(800, 600))
        pl.add_mesh(mesh, color=MESH_COLOR_IN, show_edges=True, edge_color=EDGE_COLOR,
                    line_width=0.3, lighting=True, smooth_shading=True, opacity=0.5)
        pl.add_mesh(arrows1, color="red", lighting=False)
        pl.add_mesh(arrows2, color="#4488ff", lighting=False)
        pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
        pl.set_background(BG_COLOR)
        pl.camera_position = "iso"
        pl.screenshot(out_png, transparent_background=False)
        pl.close()
        return mesh

    def count_sharp_edges(sharp_path):
        if not sharp_path or not os.path.exists(sharp_path):
            return 0
        with open(sharp_path) as f:
            return int(f.readline().strip())

    # -- Demo-dict builders --
    # PNG is {name}.png (for _render_table), VTK is {name}_after.vtk (for attach_metrics).

    def make_demo(name, side, step_name, v, f, elapsed=0, label="",
                  feat_data=None):
        """Render mesh to {name}.png, save {name}_after.vtk, return demo dict."""
        pv_m = pv_mesh_from_numpy(v, f)
        png_path = os.path.join(OUT_DIR, f"{name}.png")
        vtk_path = os.path.join(OUT_DIR, f"{name}_after.vtk")

        if feat_data is not None:
            # render_gpu_features saves VTK to out_png.replace(".png",".vtk"),
            # so render to {name}_after.png — the VTK becomes {name}_after.vtk
            tmp_png = os.path.join(OUT_DIR, f"{name}_after.png")
            _, n_feat = render_gpu_features(v, f, feat_data, tmp_png,
                                            label or step_name)
            # Copy to {name}.png for the HTML table
            if os.path.exists(tmp_png):
                shutil.copy2(tmp_png, png_path)
            lbl = label or f"{len(v):,}V {len(f):,}F | {n_feat} features"
        else:
            render_mesh(pv_m, png_path, label or f"{len(v):,}V {len(f):,}F")
            pv_m.save(vtk_path)
            lbl = label or f"{len(v):,}V {len(f):,}F"

        return {"name": name, "side": side, "step_name": step_name,
                "elapsed": elapsed, "after_label": lbl}

    def make_demo_from_checkpoint(name, side, step_name, ckpt, label=""):
        """Render a QuadWild checkpoint to {name}.png, save VTK, return demo dict."""
        png_path = os.path.join(OUT_DIR, f"{name}.png")
        vtk_path = os.path.join(OUT_DIR, f"{name}_after.vtk")
        lbl = label or step_name
        if ckpt.get("obj") and os.path.exists(ckpt["obj"]):
            mesh = pv.read(ckpt["obj"])
            render_with_features(ckpt["obj"], ckpt.get("sharp"), png_path,
                                 label or step_name)
            mesh.save(vtk_path)
            lbl = label or f"{mesh.n_points:,}V {mesh.n_cells:,}F"
        return {"name": name, "side": side, "step_name": step_name,
                "elapsed": 0, "after_label": lbl}

    # Map step names to stop_after values for gpu_fast pipeline ordering
    STEP_ORDER = ["features", "erode", "remesh_isotropic", "micro",
                  "remesh_adaptive", "clean", "refine"]

    def run_gpu_fast(up_to):
        """Run GPU pipeline end-to-end in subprocess (crash-safe), no logging."""
        import subprocess, textwrap, json
        print(f"  [4/4] Running GPU fast pipeline (up to {up_to})...")

        step_map = {
            "features": "Raw Features", "erode": "Erode/Dilate",
            "remesh_isotropic": "Isotropic Remesh",
            "micro": "Re-detect Features",
            "remesh_adaptive": "Adaptive Remesh",
            "clean": "Clean", "refine": "Refine",
        }

        out_vtk = os.path.join(OUT_DIR, f"qw_fast_{up_to}_after.vtk")
        out_json = os.path.join(OUT_DIR, f"qw_fast_{up_to}_info.json")

        script = textwrap.dedent(f"""\
            import numpy as np, time, json, pyrxmesh
            pyrxmesh.init()
            v, f = pyrxmesh.load_obj("{os.path.join(RXMESH_INPUT, 'dragon.obj')}")

            steps = {repr(STEP_ORDER[:STEP_ORDER.index(up_to) + 1])}
            t0 = time.time()

            fd = pyrxmesh.detect_features(v, f, crease_angle_deg=35.0, erode_dilate_steps=4)
            el = pyrxmesh.expected_edge_length(v, f)

            if "remesh_isotropic" in steps:
                v, f = pyrxmesh.feature_remesh(v, f,
                    relative_len=el.target_edge_length / el.avg_edge_length,
                    iterations=1, crease_angle_deg=35.0, max_passes=1, verbose=False)
            if "micro" in steps:
                v, f = pyrxmesh.vcg_micro_collapse(v, f, verbose=False)
                pyrxmesh.detect_features(v, f, crease_angle_deg=35.0, erode_dilate_steps=0)
            if "remesh_adaptive" in steps:
                v, f = pyrxmesh.feature_remesh(v, f,
                    relative_len=el.target_edge_length / el.avg_edge_length,
                    iterations=1, crease_angle_deg=35.0, max_passes=1, verbose=False)
            if "clean" in steps:
                v, f = pyrxmesh.vcg_clean_mesh(v, f, verbose=False)
            if "refine" in steps:
                v, f = pyrxmesh.vcg_refine_if_needed(v, f, verbose=False)

            t_total = time.time() - t0

            # Save result
            import pyvista as pv
            n = len(f)
            pv_f = np.column_stack([np.full(n, 3, dtype=np.int32), f]).ravel()
            m = pv.PolyData(v, pv_f)
            m.save("{out_vtk}")
            json.dump({{"elapsed": t_total, "V": len(v), "F": len(f)}},
                      open("{out_json}", "w"))
        """)

        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode == 0 and os.path.exists(out_json):
            info = json.load(open(out_json))
            t_total = info["elapsed"]
            nv, nf = info["V"], info["F"]

            # Render the result
            name = f"qw_fast_{up_to}"
            m = pv.read(out_vtk)
            render_mesh(m, os.path.join(OUT_DIR, f"{name}.png"),
                        f"GPU fast: {nv:,}V {nf:,}F")

            qw_all.append({
                "name": name, "side": "gpu_fast",
                "step_name": step_map.get(up_to, up_to),
                "elapsed": t_total,
                "after_label": f"total: {t_total*1000:.0f}ms",
            })
            print(f"    GPU fast done: {t_total*1000:.0f}ms total, {nv}V {nf}F")
            os.remove(out_json)
        else:
            print(f"  [gpu_fast FAILED] exit {proc.returncode}")
            if proc.stderr:
                for line in proc.stderr.strip().split('\n')[-3:]:
                    print(f"    {line}")

    def make_result(subtitle, partial=False, fast_up_to=None):
        """Build the return dict. Runs gpu_fast pipeline first."""
        if fast_up_to:
            try:
                run_gpu_fast(fast_up_to)
            except Exception as e:
                print(f"  [gpu_fast FAILED] {e}")
        print("  Computing quality metrics...")
        attach_metrics_to_demos(qw_all, input_mesh)
        sub = subtitle if partial else (
            "4-column comparison: CPU (QuadWild binary), CPU (our VCG wrappers), "
            "GPU (our GPU kernels), GPU fast (no logging). Each column is a sequential pipeline.")
        return {
            "title": "QuadWild Pipeline: CPU orig vs CPU ours vs GPU",
            "subtitle": sub,
            "layout": "table",
            "demos": qw_all,
        }

    # ===================================================================
    #  Run pipelines
    # ===================================================================

    def should_run(step_name):
        """Check if this step should run given stop_after."""
        if stop_after is None:
            return True
        order = ["features", "erode", "remesh_isotropic", "micro",
                 "remesh_adaptive", "clean", "refine"]
        if step_name not in order or stop_after not in order:
            return True
        return order.index(step_name) <= order.index(stop_after)

    # ── Pipeline 1: CPU original (QuadWild binary) ─────────────────────
    print("\n  [1/4] Running CPU original (QuadWild binary)...")
    qw_result = pyrxmesh.quadwild_pipeline(
        os.path.join(RXMESH_INPUT, "dragon.obj"),
        output_dir="/tmp/qw_demo_full",
        steps=3,
    )
    ckpts = qw_result.get("checkpoints", {})

    # Render all available checkpoints as cpu_orig demos
    ckpt_steps = [
        ("step_1_0_input", "Input"),
        ("step_1_1a_features_raw", "Raw Features"),
        ("step_1_1b_features_eroded", "Erode/Dilate"),
        # QuadWild's remeshed = after both passes (iso+micro+adaptive)
        ("step_1_2_remeshed", "Adaptive Remesh"),
        ("step_1_3_cleaned", "Clean"),
        ("step_1_4_refined", "Refine"),
    ]
    for ckpt_key, step_name in ckpt_steps:
        if ckpt_key in ckpts:
            label = "Full remesh (iso+micro+adaptive)" if ckpt_key == "step_1_2_remeshed" else ""
            qw_all.append(make_demo_from_checkpoint(
                f"qw_orig_{ckpt_key}", "cpu_orig", step_name,
                ckpts[ckpt_key], label=label))
            print(f"    {step_name}")

    # Cross field, trace, quad (cpu_orig only — only if running full pipeline)
    if stop_after is None:
        if "step_1_4_refined" in ckpts and "step_1_5_field" in ckpts:
            print("    Cross Field")
            ac = ckpts["step_1_5_field"]
            ac_mesh = pv.read(ac["obj"])
            png_path = os.path.join(OUT_DIR, "qw_orig_field.png")
            vtk_path = os.path.join(OUT_DIR, "qw_orig_field_after.vtk")
            if ac.get("rosy"):
                render_crossfield(ac["obj"], ac["rosy"], png_path,
                                  f"Cross Field: {ac_mesh.n_points:,}V")
            else:
                render_with_features(ac["obj"], ac.get("sharp"), png_path,
                                     f"Cross Field: {ac_mesh.n_points:,}V")
            ac_mesh.save(vtk_path)
            qw_all.append({
                "name": "qw_orig_field", "side": "cpu_orig",
                "step_name": "Cross Field", "elapsed": 0,
                "after_label": f"Cross Field ({ac_mesh.n_points:,}V)",
            })

        if qw_result.get("traced") and "step_1_5_field" in ckpts:
            print("    Trace")
            traced = pv.read(qw_result["traced"])
            png_path = os.path.join(OUT_DIR, "qw_orig_traced.png")
            vtk_path = os.path.join(OUT_DIR, "qw_orig_traced_after.vtk")
            pl = pv.Plotter(off_screen=True, window_size=(800, 600))
            pl.add_mesh(traced, color=MESH_COLOR_OUT,
                        show_edges=True, edge_color=EDGE_COLOR, line_width=0.5,
                        lighting=True, smooth_shading=True)
            pl.add_text(f"Traced: {traced.n_points:,}V, {traced.n_cells:,}F",
                        position="upper_left", font_size=12, color=TEXT_COLOR)
            pl.set_background(BG_COLOR)
            pl.camera_position = "iso"
            pl.screenshot(png_path, transparent_background=False)
            pl.close()
            traced.save(vtk_path)
            qw_all.append({
                "name": "qw_orig_traced", "side": "cpu_orig",
                "step_name": "Trace", "elapsed": 0,
                "after_label": f"Traced: {traced.n_points:,}V {traced.n_cells:,}F",
            })

        if qw_result.get("quad_smooth"):
            print("    Quad Output")
            quad = pv.read(qw_result["quad_smooth"])
            png_path = os.path.join(OUT_DIR, "qw_orig_quad.png")
            vtk_path = os.path.join(OUT_DIR, "qw_orig_quad_after.vtk")
            pl = pv.Plotter(off_screen=True, window_size=(800, 600))
            pl.add_mesh(quad, color=MESH_COLOR_OUT,
                        show_edges=True, edge_color=EDGE_COLOR, line_width=0.5,
                        lighting=True, smooth_shading=True)
            pl.add_text(f"Quad: {quad.n_points:,}V, {quad.n_cells:,} quads",
                        position="upper_left", font_size=12, color=TEXT_COLOR)
            pl.set_background(BG_COLOR)
            pl.camera_position = "iso"
            pl.screenshot(png_path, transparent_background=False)
            pl.close()
            quad.save(vtk_path)
            qw_all.append({
                "name": "qw_orig_quad", "side": "cpu_orig",
                "step_name": "Quad Output", "elapsed": 0,
                "after_label": f"Quad: {quad.n_points:,}V {quad.n_cells:,} quads",
            })

    # ── Pipeline 2: CPU ours ───────────────────────────────────────────
    print("  [2/4] Running CPU ours...")
    ours_v, ours_f = dragon_v.copy(), dragon_f.copy()

    # Input
    print("    input...", end="", flush=True)
    qw_all.append(make_demo("qw_ours_input", "cpu_ours", "Input",
                            ours_v, ours_f, label="Input"))

    # Raw Features
    if should_run("features"):
        print(" features...", end="", flush=True)
        t0 = time.time()
        fd = pyrxmesh.detect_features(ours_v, ours_f,
                                      crease_angle_deg=35.0,
                                      erode_dilate_steps=0)
        t_feat = time.time() - t0
        qw_all.append(make_demo("qw_ours_features_raw", "cpu_ours",
                                "Raw Features", ours_v, ours_f,
                                elapsed=t_feat, feat_data=fd))

    # Erode/Dilate
    if should_run("erode"):
        print(" erode...", end="", flush=True)
        t0 = time.time()
        fd = pyrxmesh.detect_features(ours_v, ours_f,
                                      crease_angle_deg=35.0,
                                      erode_dilate_steps=4)
        t_erode = time.time() - t0
        qw_all.append(make_demo("qw_ours_erode", "cpu_ours", "Erode/Dilate",
                                ours_v, ours_f, elapsed=t_erode, feat_data=fd))

    # Isotropic Remesh
    if should_run("remesh_isotropic"):
        print(" iso remesh...", end="", flush=True)
        t0 = time.time()
        ck = pyrxmesh.vcg_remesh_checkpoints(ours_v, ours_f,
                                             target_faces=10000,
                                             iterations=1, adaptive=False,
                                             verbose=True)
        ours_v, ours_f = ck["after_pass1"]
        t_iso = time.time() - t0
        qw_all.append(make_demo("qw_ours_iso", "cpu_ours", "Isotropic Remesh",
                                ours_v, ours_f, elapsed=t_iso))

    # Micro Collapse
    if should_run("micro"):
        print(" micro...", end="", flush=True)
        t0 = time.time()
        ours_v, ours_f = pyrxmesh.vcg_micro_collapse(ours_v, ours_f,
                                                      verbose=True)
        t_micro = time.time() - t0
        qw_all.append(make_demo("qw_ours_micro", "cpu_ours", "Micro Collapse",
                                ours_v, ours_f, elapsed=t_micro))

    # Re-detect Features (part of micro step)
    if should_run("micro"):
        print(" re-detect...", end="", flush=True)
        fd_re = pyrxmesh.detect_features(ours_v, ours_f,
                                         crease_angle_deg=35.0,
                                         erode_dilate_steps=0)
        qw_all.append(make_demo("qw_ours_redetect", "cpu_ours",
                                "Re-detect Features",
                                ours_v, ours_f, feat_data=fd_re))

    # Adaptive Remesh
    if should_run("remesh_adaptive"):
        print(" adaptive...", end="", flush=True)
        t0 = time.time()
        ours_v, ours_f = pyrxmesh.vcg_remesh(ours_v, ours_f,
                                             target_faces=10000,
                                             adaptive=True, verbose=True)
        t_adapt = time.time() - t0
        qw_all.append(make_demo("qw_ours_adaptive", "cpu_ours",
                                "Adaptive Remesh", ours_v, ours_f,
                                elapsed=t_adapt))

    # Clean
    if should_run("clean"):
        print(" clean...", end="", flush=True)
        t0 = time.time()
        ours_v, ours_f = pyrxmesh.vcg_clean_mesh(ours_v, ours_f,
                                                  verbose=True)
        t_clean = time.time() - t0
        qw_all.append(make_demo("qw_ours_clean", "cpu_ours", "Clean",
                                ours_v, ours_f, elapsed=t_clean))

    # Refine
    if should_run("refine"):
        print(" refine...", end="", flush=True)
        t0 = time.time()
        ours_v, ours_f = pyrxmesh.vcg_refine_if_needed(ours_v, ours_f,
                                                        verbose=True)
        t_refine = time.time() - t0
        qw_all.append(make_demo("qw_ours_refine", "cpu_ours", "Refine",
                                ours_v, ours_f, elapsed=t_refine))

    print(" done.")

    # ── Pipeline 3: GPU ────────────────────────────────────────────────
    print("  [3/4] Running GPU...")
    gpu_v, gpu_f = dragon_v.copy(), dragon_f.copy()
    el_gpu = pyrxmesh.expected_edge_length(gpu_v, gpu_f, verbose=True)

    # Input
    print("    input...", end="", flush=True)
    qw_all.append(make_demo("qw_gpu_input", "gpu", "Input",
                            gpu_v, gpu_f, label="Input"))

    # Raw Features
    if should_run("features"):
        print(" features...", end="", flush=True)
        t0 = time.time()
        fd = pyrxmesh.detect_features(gpu_v, gpu_f,
                                      crease_angle_deg=35.0,
                                      erode_dilate_steps=0)
        t_feat = time.time() - t0
        qw_all.append(make_demo("qw_gpu_features_raw", "gpu", "Raw Features",
                                gpu_v, gpu_f, elapsed=t_feat, feat_data=fd))

    # Erode/Dilate
    if should_run("erode"):
        print(" erode...", end="", flush=True)
        t0 = time.time()
        fd = pyrxmesh.detect_features(gpu_v, gpu_f,
                                      crease_angle_deg=35.0,
                                      erode_dilate_steps=4)
        t_erode = time.time() - t0
        qw_all.append(make_demo("qw_gpu_erode", "gpu", "Erode/Dilate",
                                gpu_v, gpu_f, elapsed=t_erode, feat_data=fd))

    # Isotropic Remesh
    if should_run("remesh_isotropic"):
        print(" iso remesh...", end="", flush=True)
        t0 = time.time()
        gpu_v, gpu_f = pyrxmesh.feature_remesh(gpu_v, gpu_f,
            relative_len=el_gpu.target_edge_length / el_gpu.avg_edge_length,
            iterations=1, crease_angle_deg=35.0, max_passes=1, verbose=True)
        t_iso = time.time() - t0
        qw_all.append(make_demo("qw_gpu_iso", "gpu", "Isotropic Remesh",
                                gpu_v, gpu_f, elapsed=t_iso))

    # Micro Collapse
    if should_run("micro"):
        print(" micro...", end="", flush=True)
        t0 = time.time()
        gpu_v, gpu_f = pyrxmesh.vcg_micro_collapse(gpu_v, gpu_f,
                                                    verbose=True)
        t_micro = time.time() - t0
        qw_all.append(make_demo("qw_gpu_micro", "gpu", "Micro Collapse",
                                gpu_v, gpu_f, elapsed=t_micro))

    # Re-detect Features
    if should_run("micro"):
        print(" re-detect...", end="", flush=True)
        fd_re = pyrxmesh.detect_features(gpu_v, gpu_f,
                                         crease_angle_deg=35.0,
                                         erode_dilate_steps=0)
        qw_all.append(make_demo("qw_gpu_redetect", "gpu",
                                "Re-detect Features",
                                gpu_v, gpu_f, feat_data=fd_re))

    # Adaptive Remesh (subprocess — crashes with double free)
    if should_run("remesh_adaptive"):
        print(" adaptive (subprocess)...", end="", flush=True)
        # Save current GPU mesh to VTK for subprocess to read
        gpu_pre_vtk = os.path.join(OUT_DIR, "qw_gpu_pre_adaptive.vtk")
        n_gpu = len(gpu_f)
        pv_faces_tmp = np.column_stack(
            [np.full(n_gpu, 3, dtype=np.int32), gpu_f]).ravel()
        pv.PolyData(gpu_v, pv_faces_tmp).save(gpu_pre_vtk)

        import subprocess as _sp, textwrap as _tw
        out_vtk = os.path.join(OUT_DIR, "qw_gpu_adaptive_after.vtk")
        script = _tw.dedent(f"""\
            import numpy as np, pyvista as pv, pyrxmesh, time, sys
            pyrxmesh.init()
            m = pv.read("{gpu_pre_vtk}")
            v = np.array(m.points, dtype=np.float64)
            f = np.array(m.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)
            el = pyrxmesh.expected_edge_length(v, f)
            t0 = time.time()
            rv, rf = pyrxmesh.feature_remesh(v, f,
                relative_len=el.target_edge_length / el.avg_edge_length,
                iterations=1, crease_angle_deg=35.0, max_passes=1, verbose=True)
            elapsed = time.time() - t0
            n = len(rf)
            pv_f = np.column_stack([np.full(n, 3, dtype=np.int32), rf]).ravel()
            out = pv.PolyData(rv, pv_f)
            out.save("{out_vtk}")
            print(f"ELAPSED={{elapsed:.4f}}")
        """)
        proc = _sp.run([sys.executable, "-c", script],
                       capture_output=True, text=True, timeout=60)
        if proc.returncode == 0 and os.path.exists(out_vtk):
            t_gpu_adaptive = 0
            for line in proc.stdout.strip().split('\n'):
                if line.startswith("ELAPSED="):
                    t_gpu_adaptive = float(line.split("=")[1])
            am = pv.read(out_vtk)
            gpu_v = np.array(am.points, dtype=np.float64)
            gpu_f = np.array(am.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)
            png_path = os.path.join(OUT_DIR, "qw_gpu_adaptive.png")
            render_mesh(am, png_path,
                        f"Adaptive: {am.n_points:,}V {am.n_cells:,}F")
            qw_all.append({
                "name": "qw_gpu_adaptive", "side": "gpu",
                "step_name": "Adaptive Remesh",
                "elapsed": t_gpu_adaptive,
                "after_label": (f"Adaptive: {am.n_points:,}V "
                                f"{am.n_cells:,}F"),
            })
        else:
            print(f"\n    [FAILED] GPU Adaptive -- subprocess crashed "
                  f"(exit {proc.returncode})")
            if proc.stderr:
                for line in proc.stderr.strip().split('\n')[-5:]:
                    print(f"      {line}")

    # Clean
    if should_run("clean"):
        print(" clean...", end="", flush=True)
        t0 = time.time()
        gpu_v, gpu_f = pyrxmesh.vcg_clean_mesh(gpu_v, gpu_f, verbose=True)
        t_clean = time.time() - t0
        qw_all.append(make_demo("qw_gpu_clean", "gpu", "Clean",
                                gpu_v, gpu_f, elapsed=t_clean))

    # Refine
    if should_run("refine"):
        print(" refine...", end="", flush=True)
        t0 = time.time()
        gpu_v, gpu_f = pyrxmesh.vcg_refine_if_needed(gpu_v, gpu_f,
                                                      verbose=True)
        t_refine = time.time() - t0
        qw_all.append(make_demo("qw_gpu_refine", "gpu", "Refine",
                                gpu_v, gpu_f, elapsed=t_refine))

    print(" done.")

    # ── Pipeline 4: GPU fast (subprocess, no intermediate renders) ─────
    fast_map = {"features": "features", "erode": "erode",
                "remesh_isotropic": "remesh_isotropic", "micro": "micro",
                "remesh_adaptive": "remesh_adaptive",
                "clean": "clean", "refine": "refine",
                None: "refine"}
    fast_up_to = fast_map.get(stop_after, stop_after or "refine")

    return make_result("", fast_up_to=fast_up_to)



QUADWILD_STEPS = [
    "features",           # 1. Raw feature detection
    "erode",              # 2. Erode/dilate
    "remesh_isotropic",   # 3. Isotropic remesh (pass 1)
    "micro",              # 4. Micro-edge collapse + feature re-detect
    "remesh_adaptive",    # 5. Adaptive remesh (pass 2)
    "all",
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate QuadWild CPU vs GPU comparison demo.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--only", choices=QUADWILD_STEPS,
        help="Stop after this step:\n"
             "  erode   — features + erode/dilate only\n"
             "  remesh  — up to non-adaptive remesh\n"
             "  all     — full pipeline")
    parser.add_argument("--list", action="store_true",
        help="List available steps and exit.")

    args = parser.parse_args()

    if args.list:
        print("QuadWild pipeline steps:")
        print("  erode   — feature detection + erode/dilate (CPU + GPU)")
        print("  remesh  — + non-adaptive remesh (CPU + GPU)")
        print("  all     — full pipeline including CPU-only steps")
        sys.exit(0)

    if not args.only:
        parser.print_help()
        print("\nError: specify --only {erode,remesh,all}")
        sys.exit(1)

    tee = setup_logging()

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    pyrxmesh.init()

    print("Loading dragon mesh...")
    dragon_v, dragon_f = pyrxmesh.load_obj(os.path.join(RXMESH_INPUT, "dragon.obj"))
    print(f"  dragon: {len(dragon_v)} verts, {len(dragon_f)} faces")

    stop = None if args.only == "all" else args.only

    print(f"\n=== Generating: QuadWild Pipeline (stop_after={stop}) ===")
    section = gen_quadwild(dragon_v, dragon_f, stop_after=stop)

    if section:
        print("\n=== Writing HTML ===")
        generate_html([section])

        # Print summary table
        demos = section.get("demos", [])
        has_metrics = any(d.get("metrics") for d in demos)
        if has_metrics:
            print("\n=== Quality Summary ===")
            hdr = (f"{'Step':<25} {'Side':<10} {'V':>7} {'F':>7} "
                   f"{'Q avg':>7} {'Q max':>10} {'edge std':>10} "
                   f"{'%val6':>6} {'dist':>8} {'time':>8}")
            sep = "-" * len(hdr)
            print(hdr)
            print(sep)
            from collections import OrderedDict
            ROW_ORDER = ["Input", "Raw Features", "Erode/Dilate",
                         "Isotropic Remesh", "Micro Collapse",
                         "Re-detect Features", "Adaptive Remesh",
                         "Clean", "Refine", "Cross Field", "Trace", "Quad Output"]
            steps = OrderedDict((s, {}) for s in ROW_ORDER)
            for d in demos:
                step = d.get("step_name", "?")
                if step not in steps:
                    steps[step] = {}
                steps[step][d.get("side", "?")] = d
            prev_printed = False
            for step, sides in steps.items():
                if not sides:
                    continue
                has_any = False
                for side in ("cpu_orig", "cpu_ours", "gpu", "gpu_fast"):
                    d = sides.get(side)
                    if not d or not d.get("metrics"):
                        continue
                    has_any = True
                if not has_any:
                    continue
                if prev_printed:
                    print(sep)
                for side in ("cpu_orig", "cpu_ours", "gpu", "gpu_fast"):
                    d = sides.get(side)
                    if not d:
                        continue
                    m = d.get("metrics")
                    if not m:
                        continue
                    t = d.get("elapsed", 0)
                    t_str = f"{t*1000:.0f}ms" if t > 0 else ""
                    dist = f"{m['dist_avg']:.4f}" if "dist_avg" in m else ""
                    print(f"{step:<25} {side:<10} {m['V']:>7,} {m['F']:>7,} "
                          f"{m['quality_avg']:>7.2f} {m['quality_max']:>10.0f} "
                          f"{m['edge_std']:>10.4f} {m['pct_val6']:>5.0f}% "
                          f"{dist:>8} {t_str:>8}")
                prev_printed = True

        import glob
        n_files = len(glob.glob(os.path.join(OUT_DIR, "*")))
        print(f"\nDone! Generated {n_files} files in {OUT_DIR}/")


if __name__ == "__main__":
    main()
