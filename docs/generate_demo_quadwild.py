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
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

import pyrxmesh

# Import shared utilities from generate_demo
from generate_demo import (
    OUT_DIR, RXMESH_INPUT, RUNS_DIR,
    BG_COLOR, MESH_COLOR_IN, MESH_COLOR_OUT, EDGE_COLOR, TEXT_COLOR,
    pv_mesh_from_numpy, render_mesh, run_before_after,
    setup_logging, generate_html,
)

def gen_quadwild(dragon_v, dragon_f, stop_after=None):
    """Generate QuadWild pipeline CPU vs GPU comparison demos.

    Args:
        stop_after: "erode" stops after erode/dilate, "remesh" stops after
                    non-adaptive remesh, None runs everything.
    """
    qw_all = []

    # -- Helpers --

    dragon_pv_qw = pv_mesh_from_numpy(dragon_v, dragon_f)

    def render_feat(scalars, filename, title):
        pl = pv.Plotter(off_screen=True, window_size=(800, 600))
        pl.add_mesh(dragon_pv_qw, scalars=scalars, cmap="coolwarm",
                    show_edges=True, edge_color=EDGE_COLOR, line_width=0.3,
                    lighting=True, smooth_shading=True)
        pl.add_text(title, position="upper_left", font_size=12, color=TEXT_COLOR)
        pl.set_background(BG_COLOR)
        pl.camera_position = "iso"
        pl.screenshot(filename, transparent_background=False)
        pl.close()

    def render_gpu_features(verts, faces, feature_data, out_png, title):
        n = len(faces)
        pv_faces = np.column_stack([np.full(n, 3, dtype=np.int32), faces]).ravel()
        mesh = pv.PolyData(verts, pv_faces)
        nE = mesh.extract_all_edges().n_cells

        # Build edge index -> vertex pair mapping from faces
        edge_map = {}  # (min_v, max_v) -> edge_index
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

        # Collect feature edge vertex pairs
        edge_is_feat = feature_data.edge_is_feature
        feat_pairs = []
        for (v0, v1), ei in edge_map.items():
            if ei < len(edge_is_feat) and edge_is_feat[ei]:
                feat_pairs.append([v0, v1])
        n_feat = len(feat_pairs)

        print(f"    GPU features: {len(verts)}V, {nE}E, {n}F, "
              f"{n_feat} feature edges (reported: {feature_data.num_feature_edges})")

        subtitle = f"{len(verts):,}V  {nE:,}E  {n:,}F  |  {n_feat} feature edges"

        # Save mesh + feature data as VTK
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

        # Deduplicate half-edges to unique edges
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

        # Save mesh + feature edges as VTK
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

    # -- Run GPU operations --
    print("  Running GPU feature detection...")

    fd_gpu_raw = pyrxmesh.detect_features(dragon_v, dragon_f, crease_angle_deg=35.0, erode_dilate_steps=0)
    fd_gpu_ed = pyrxmesh.detect_features(dragon_v, dragon_f, crease_angle_deg=35.0, erode_dilate_steps=4)
    el_gpu = pyrxmesh.expected_edge_length(dragon_v, dragon_f, verbose=True)

    # -- Step 0: Compare target edge lengths --
    print(f"\n  === Target Edge Length Comparison ===")
    print(f"  GPU: target={el_gpu.target_edge_length:.6f}, avg={el_gpu.avg_edge_length:.6f}, "
          f"sphericity={el_gpu.sphericity:.4f}")
    # CPU computes this internally in vcg_remesh -- the verbose output prints it.
    # Both use the same formula (expected_edge_length / IdealL0/IdealL1).

    print("  Running CPU QuadWild pipeline...")
    # -- Run CPU pipeline --

    qw_result = pyrxmesh.quadwild_pipeline(
        os.path.join(RXMESH_INPUT, "dragon.obj"),
        output_dir="/tmp/qw_demo_full",
        steps=3
    )
    ckpts = qw_result.get("checkpoints", {})

    print("  Rendering: Raw Features (CPU + GPU)")
    # -- Step 1: Raw Features (CPU + GPU) --
    # Note: CPU .sharp files count face-edges (half-edges), ~2x unique edges.
    # GPU counts unique edges. We show both for clarity.

    # CPU side
    if "step_1_0_input" in ckpts and "step_1_1a_features_raw" in ckpts:
        bc, ac = ckpts["step_1_0_input"], ckpts["step_1_1a_features_raw"]
        prefix = os.path.join(OUT_DIR, "qw_cpu_raw_feat")
        render_with_features(bc["obj"], bc["sharp"], f"{prefix}_before.png", "Input")
        _, cpu_raw_n = render_with_features(ac["obj"], ac["sharp"], f"{prefix}_after.png",
            "Raw Features")
        qw_all.append({
            "name": "qw_cpu_raw_feat", "type": "before_after",
            "step_name": "Raw Features", "side": "cpu",
            "verts_in": len(dragon_v), "faces_in": len(dragon_f),
            "verts_out": len(dragon_v), "faces_out": len(dragon_f),
            "elapsed": 0, "after_label": f"Raw ({cpu_raw_n} feature edges)",
            "code": "Detect dihedral angle > 35\u00b0", "verbose": "",
        })

    # GPU side
    prefix = os.path.join(OUT_DIR, "qw_gpu_raw_feat")
    render_mesh(dragon_pv_qw, f"{prefix}_before.png", "Input")
    _, gpu_raw_n = render_gpu_features(dragon_v, dragon_f, fd_gpu_raw,
                        f"{prefix}_after.png", "Raw Features")
    qw_all.append({
        "name": "qw_gpu_raw_feat", "type": "before_after",
        "step_name": "Raw Features", "side": "gpu",
        "verts_in": len(dragon_v), "faces_in": len(dragon_f),
        "verts_out": len(dragon_v), "faces_out": len(dragon_f),
        "elapsed": 0, "after_label": f"Raw ({gpu_raw_n} feature edges)",
        "code": "pyrxmesh.detect_features(v, f, erode_dilate_steps=0)",
        "verbose": "",
    })

    print("  Rendering: Erode/Dilate (CPU + GPU)")
    # -- Step 2: Erode/Dilate (CPU + GPU) --

    if "step_1_1a_features_raw" in ckpts and "step_1_1b_features_eroded" in ckpts:
        bc, ac = ckpts["step_1_1a_features_raw"], ckpts["step_1_1b_features_eroded"]
        prefix = os.path.join(OUT_DIR, "qw_cpu_erode")
        render_with_features(bc["obj"], bc["sharp"], f"{prefix}_before.png", "Raw Features")
        _, cpu_eroded_n = render_with_features(ac["obj"], ac["sharp"], f"{prefix}_after.png",
            "Eroded")
        qw_all.append({
            "name": "qw_cpu_erode", "type": "before_after",
            "step_name": "Erode/Dilate", "side": "cpu",
            "verts_in": len(dragon_v), "faces_in": len(dragon_f),
            "verts_out": len(dragon_v), "faces_out": len(dragon_f),
            "elapsed": 0, "after_label": f"Eroded ({cpu_eroded_n} feature edges)",
            "code": "Erode/dilate(4) removes noise", "verbose": "",
        })

    prefix = os.path.join(OUT_DIR, "qw_gpu_erode")
    render_gpu_features(dragon_v, dragon_f, fd_gpu_raw,
                        f"{prefix}_before.png", "Raw Features")
    _, gpu_eroded_n = render_gpu_features(dragon_v, dragon_f, fd_gpu_ed,
                        f"{prefix}_after.png", "Eroded")
    qw_all.append({
        "name": "qw_gpu_erode", "type": "before_after",
        "step_name": "Erode/Dilate", "side": "gpu",
        "verts_in": len(dragon_v), "faces_in": len(dragon_f),
        "verts_out": len(dragon_v), "faces_out": len(dragon_f),
        "elapsed": 0, "after_label": f"Eroded ({gpu_eroded_n} feature edges)",
        "code": "pyrxmesh.detect_features(v, f, erode_dilate_steps=4)",
        "verbose": "",
    })

    if stop_after == "erode":
        return {
            "title": "QuadWild Pipeline: CPU vs GPU",
            "subtitle": "Side-by-side comparison (partial — stopped after erode/dilate)",
            "layout": "table",
            "demos": qw_all,
        }

    print("  Rendering: Remesh Non-Adaptive (CPU + GPU)")
    # -- Step 3a: Remesh Non-Adaptive (CPU + GPU) --

    # CPU non-adaptive
    d = run_before_after("qw_cpu_remesh_p1",
        lambda v, f: pyrxmesh.vcg_remesh(v, f, target_faces=10000, iterations=15, adaptive=False, verbose=True),
        dragon_v, dragon_f,
        "pyrxmesh.vcg_remesh(v, f, target_faces=10000, iterations=15, adaptive=False)",
        after_label="CPU Non-Adaptive")
    d["step_name"] = "Remesh (non-adaptive)"
    d["side"] = "cpu"
    qw_all.append(d)

    # GPU non-adaptive (pass 1 only)
    d = run_before_after("qw_gpu_remesh_p1",
        lambda v, f: pyrxmesh.feature_remesh(v, f,
            relative_len=el_gpu.target_edge_length / el_gpu.avg_edge_length,
            iterations=15, crease_angle_deg=35.0, max_passes=1, verbose=True),
        dragon_v, dragon_f,
        "pyrxmesh.feature_remesh(v, f, ..., max_passes=1)",
        after_label="GPU Non-Adaptive")
    d["step_name"] = "Remesh (non-adaptive)"
    d["side"] = "gpu"
    qw_all.append(d)

    if stop_after == "remesh":
        return {
            "title": "QuadWild Pipeline: CPU vs GPU",
            "subtitle": "Side-by-side comparison (partial — stopped after non-adaptive remesh)",
            "layout": "table",
            "demos": qw_all,
        }

    print("  Rendering: Remesh Adaptive (CPU + GPU)")
    # -- Step 3b: Remesh Adaptive (CPU + GPU) --

    # CPU adaptive (full two-pass)
    d = run_before_after("qw_cpu_remesh_p2",
        lambda v, f: pyrxmesh.vcg_remesh(v, f, target_faces=10000, adaptive=True, verbose=True),
        dragon_v, dragon_f,
        "pyrxmesh.vcg_remesh(v, f, target_faces=10000, adaptive=True)",
        after_label="CPU Adaptive")
    d["step_name"] = "Remesh (adaptive)"
    d["side"] = "cpu"
    qw_all.append(d)

    # GPU adaptive -- DISABLED: pass 2 crashes (collapse kernel illegal memory access)
    # TODO: debug pass 2 crash (see boffins_room/19_debugging_persistent_crash.md)
    print("  [SKIPPED] GPU Adaptive — pass 2 crashes")

    # -- Step 3c: Re-detect features on remeshed outputs --
    print("  Rendering: Re-detected Features on Remeshed Mesh")

    # CPU: run detect_features on the CPU non-adaptive remeshed output
    if hasattr(qw_all[-3] if len(qw_all) >= 3 else None, '__getitem__'):
        # Find the CPU non-adaptive remesh result
        cpu_remesh_d = next((d for d in qw_all if d.get("name") == "qw_cpu_remesh_p1"), None)
        gpu_remesh_d = next((d for d in qw_all if d.get("name") == "qw_gpu_remesh_p1"), None)

        if cpu_remesh_d:
            # Load CPU remeshed mesh from VTK
            cpu_remesh_vtk = os.path.join(OUT_DIR, "qw_cpu_remesh_p1_after.vtk")
            if os.path.exists(cpu_remesh_vtk):
                cpu_rm = pv.read(cpu_remesh_vtk)
                cpu_rm_v = np.array(cpu_rm.points, dtype=np.float64)
                cpu_rm_f = np.array(cpu_rm.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)
                fd_cpu_redetect = pyrxmesh.detect_features(cpu_rm_v, cpu_rm_f,
                    crease_angle_deg=35.0, erode_dilate_steps=0)
                prefix = os.path.join(OUT_DIR, "qw_cpu_redetect")
                render_mesh(pv_mesh_from_numpy(cpu_rm_v, cpu_rm_f),
                            f"{prefix}_before.png", "CPU Remeshed")
                _, cpu_redetect_n = render_gpu_features(cpu_rm_v, cpu_rm_f, fd_cpu_redetect,
                    f"{prefix}_after.png", "Re-detected Features")
                qw_all.append({
                    "name": "qw_cpu_redetect", "type": "before_after",
                    "step_name": "Re-detect Features", "side": "cpu",
                    "verts_in": len(cpu_rm_v), "faces_in": len(cpu_rm_f),
                    "verts_out": len(cpu_rm_v), "faces_out": len(cpu_rm_f),
                    "elapsed": 0, "after_label": f"Re-detected ({cpu_redetect_n} features)",
                    "code": "detect_features on remeshed mesh", "verbose": "",
                })

        if gpu_remesh_d:
            gpu_remesh_vtk = os.path.join(OUT_DIR, "qw_gpu_remesh_p1_after.vtk")
            if os.path.exists(gpu_remesh_vtk):
                gpu_rm = pv.read(gpu_remesh_vtk)
                gpu_rm_v = np.array(gpu_rm.points, dtype=np.float64)
                gpu_rm_f = np.array(gpu_rm.faces.reshape(-1, 4)[:, 1:], dtype=np.int32)
                fd_gpu_redetect = pyrxmesh.detect_features(gpu_rm_v, gpu_rm_f,
                    crease_angle_deg=35.0, erode_dilate_steps=0)
                prefix = os.path.join(OUT_DIR, "qw_gpu_redetect")
                render_mesh(pv_mesh_from_numpy(gpu_rm_v, gpu_rm_f),
                            f"{prefix}_before.png", "GPU Remeshed")
                _, gpu_redetect_n = render_gpu_features(gpu_rm_v, gpu_rm_f, fd_gpu_redetect,
                    f"{prefix}_after.png", "Re-detected Features")
                qw_all.append({
                    "name": "qw_gpu_redetect", "type": "before_after",
                    "step_name": "Re-detect Features", "side": "gpu",
                    "verts_in": len(gpu_rm_v), "faces_in": len(gpu_rm_f),
                    "verts_out": len(gpu_rm_v), "faces_out": len(gpu_rm_f),
                    "elapsed": 0, "after_label": f"Re-detected ({gpu_redetect_n} features)",
                    "code": "detect_features on remeshed mesh", "verbose": "",
                })

    print("  Rendering: Clean (CPU only)")
    # -- Step 4: Clean (CPU only) --

    if "step_1_2_remeshed" in ckpts and "step_1_3_cleaned" in ckpts:
        bc, ac = ckpts["step_1_2_remeshed"], ckpts["step_1_3_cleaned"]
        bc_mesh, ac_mesh = pv.read(bc["obj"]), pv.read(ac["obj"])
        prefix = os.path.join(OUT_DIR, "qw_cpu_clean")
        render_with_features(bc["obj"], bc["sharp"], f"{prefix}_before.png",
            f"Remeshed: {bc_mesh.n_points:,}V, {bc_mesh.n_cells:,}F")
        render_with_features(ac["obj"], ac["sharp"], f"{prefix}_after.png",
            f"Cleaned: {ac_mesh.n_points:,}V, {ac_mesh.n_cells:,}F")
        qw_all.append({
            "name": "qw_cpu_clean", "type": "before_after",
            "step_name": "Clean", "side": "cpu",
            "verts_in": bc_mesh.n_points, "faces_in": bc_mesh.n_cells,
            "verts_out": ac_mesh.n_points, "faces_out": ac_mesh.n_cells,
            "elapsed": 0, "after_label": f"Cleaned ({ac_mesh.n_points:,}V, {ac_mesh.n_cells:,}F)",
            "code": "SolveGeometricArtifacts", "verbose": "",
        })

    print("  Rendering: Refine (CPU only)")
    # -- Step 5: Refine (CPU only) --

    if "step_1_3_cleaned" in ckpts and "step_1_4_refined" in ckpts:
        bc, ac = ckpts["step_1_3_cleaned"], ckpts["step_1_4_refined"]
        bc_mesh, ac_mesh = pv.read(bc["obj"]), pv.read(ac["obj"])
        prefix = os.path.join(OUT_DIR, "qw_cpu_refine")
        render_with_features(bc["obj"], bc["sharp"], f"{prefix}_before.png",
            f"Cleaned: {bc_mesh.n_points:,}V, {bc_mesh.n_cells:,}F")
        render_with_features(ac["obj"], ac["sharp"], f"{prefix}_after.png",
            f"Refined: {ac_mesh.n_points:,}V, {ac_mesh.n_cells:,}F")
        qw_all.append({
            "name": "qw_cpu_refine", "type": "before_after",
            "step_name": "Refine", "side": "cpu",
            "verts_in": bc_mesh.n_points, "faces_in": bc_mesh.n_cells,
            "verts_out": ac_mesh.n_points, "faces_out": ac_mesh.n_cells,
            "elapsed": 0, "after_label": f"Refined ({ac_mesh.n_points:,}V, {ac_mesh.n_cells:,}F)",
            "code": "RefineIfNeeded", "verbose": "",
        })

    print("  Rendering: Cross Field (CPU only)")
    # -- Step 6: Cross Field (CPU only) --

    if "step_1_4_refined" in ckpts and "step_1_5_field" in ckpts:
        bc, ac = ckpts["step_1_4_refined"], ckpts["step_1_5_field"]
        bc_mesh, ac_mesh = pv.read(bc["obj"]), pv.read(ac["obj"])
        prefix = os.path.join(OUT_DIR, "qw_cpu_field")
        render_with_features(bc["obj"], bc["sharp"], f"{prefix}_before.png",
            f"Refined: {bc_mesh.n_points:,}V, {bc_mesh.n_cells:,}F")
        if ac.get("rosy"):
            render_crossfield(ac["obj"], ac["rosy"], f"{prefix}_after.png",
                f"Cross Field: {ac_mesh.n_points:,}V, {ac_mesh.n_cells:,}F")
        else:
            render_with_features(ac["obj"], ac["sharp"], f"{prefix}_after.png",
                f"Cross Field: {ac_mesh.n_points:,}V, {ac_mesh.n_cells:,}F")
        qw_all.append({
            "name": "qw_cpu_field", "type": "before_after",
            "step_name": "Cross Field", "side": "cpu",
            "verts_in": bc_mesh.n_points, "faces_in": bc_mesh.n_cells,
            "verts_out": ac_mesh.n_points, "faces_out": ac_mesh.n_cells,
            "elapsed": 0, "after_label": f"Cross Field ({ac_mesh.n_points:,}V)",
            "code": "Cross field + singularity detection", "verbose": "",
        })

    print("  Rendering: Trace (CPU only)")
    # -- Step 7: Trace (CPU only) --

    if qw_result.get("traced") and "step_1_5_field" in ckpts:
        field_mesh = pv.read(ckpts["step_1_5_field"]["obj"])
        prefix = os.path.join(OUT_DIR, "qw_cpu_traced")
        render_with_features(ckpts["step_1_5_field"]["obj"], None,
            f"{prefix}_before.png", f"Cross Field: {field_mesh.n_points:,}V")
        traced = pv.read(qw_result["traced"])
        pl = pv.Plotter(off_screen=True, window_size=(800, 600))
        pl.add_mesh(traced, color=MESH_COLOR_OUT,
                    show_edges=True, edge_color=EDGE_COLOR, line_width=0.5,
                    lighting=True, smooth_shading=True)
        pl.add_text(f"Traced: {traced.n_points:,}V, {traced.n_cells:,}F",
                    position="upper_left", font_size=12, color=TEXT_COLOR)
        pl.set_background(BG_COLOR)
        pl.camera_position = "iso"
        pl.screenshot(f"{prefix}_after.png", transparent_background=False)
        pl.close()
        qw_all.append({
            "name": "qw_cpu_traced", "type": "before_after",
            "step_name": "Trace", "side": "cpu",
            "verts_in": pv.read(ckpts["step_1_5_field"]["obj"]).n_points,
            "faces_in": pv.read(ckpts["step_1_5_field"]["obj"]).n_cells,
            "verts_out": traced.n_points, "faces_out": traced.n_cells,
            "elapsed": 0, "after_label": "CPU Traced",
            "code": "Field tracing \u2192 patch layout", "verbose": "",
        })

    print("  Rendering: Quad Output (CPU only)")
    # -- Step 8: Quad Output (CPU only) --

    if qw_result.get("quad_smooth"):
        prefix = os.path.join(OUT_DIR, "qw_cpu_quad")
        render_mesh(pv_mesh_from_numpy(dragon_v, dragon_f),
                    f"{prefix}_before.png", f"Input: {len(dragon_v):,} tris")
        quad = pv.read(qw_result["quad_smooth"])
        pl = pv.Plotter(off_screen=True, window_size=(800, 600))
        pl.add_mesh(quad, color=MESH_COLOR_OUT,
                    show_edges=True, edge_color=EDGE_COLOR, line_width=0.5,
                    lighting=True, smooth_shading=True)
        pl.add_text(f"Quad: {quad.n_points:,}V, {quad.n_cells:,} quads",
                    position="upper_left", font_size=12, color=TEXT_COLOR)
        pl.set_background(BG_COLOR)
        pl.camera_position = "iso"
        pl.screenshot(f"{prefix}_after.png", transparent_background=False)
        pl.close()
        qw_all.append({
            "name": "qw_cpu_quad", "type": "before_after",
            "step_name": "Quad Output", "side": "cpu",
            "verts_in": len(dragon_v), "faces_in": len(dragon_f),
            "verts_out": quad.n_points, "faces_out": quad.n_cells,
            "elapsed": 0, "after_label": "CPU Quad Output",
            "code": "100% quad mesh", "verbose": "",
        })

    return {
        "title": "QuadWild Pipeline: CPU vs GPU",
        "subtitle": "Side-by-side comparison of CPU (QuadWild binary) and GPU (pyrxmesh) pipeline steps. "
                    "Steps after Remesh are CPU-only for now.",
        "layout": "table",
        "demos": qw_all,
    }



QUADWILD_STEPS = ["erode", "remesh", "all"]


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

        import glob
        files = sorted(glob.glob(os.path.join(OUT_DIR, "*")))
        print(f"\nDone! Generated {len(files)} files in {OUT_DIR}/")
        for f in files:
            size = os.path.getsize(f)
            print(f"  {os.path.basename(f)} ({size // 1024}KB)")


if __name__ == "__main__":
    main()
