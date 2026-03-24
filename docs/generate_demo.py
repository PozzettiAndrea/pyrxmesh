"""
Generate visual demo of pyrxmesh for GitHub Pages.

Uses the persistent Mesh class so timings reflect actual GPU kernel speed
(not RXMesh construction overhead).
"""

import os
import io
import sys
import shutil
import time
import textwrap
import numpy as np

import pyvista as pv

pv.OFF_SCREEN = True

import pyrxmesh

OUT_DIR = os.path.join(os.path.dirname(__file__), "_site")
RXMESH_INPUT = os.path.join(os.path.dirname(__file__), "..", "RXMesh", "input")

BG_COLOR = "#1a1a2e"
MESH_COLOR_IN = "#4fc3f7"
MESH_COLOR_OUT = "#81c784"
EDGE_COLOR = "#222244"
TEXT_COLOR = "#e0e0e0"


def pv_mesh_from_numpy(verts, faces):
    n = len(faces)
    pv_faces = np.column_stack([np.full(n, 3, dtype=np.int32), faces]).ravel()
    return pv.PolyData(verts, pv_faces)


def render_mesh(mesh, filename, title, color=MESH_COLOR_IN,
                window_size=(800, 600), scalars=None, cmap="viridis",
                show_edges=True):
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    kwargs = dict(show_edges=show_edges, edge_color=EDGE_COLOR,
                  line_width=0.5, lighting=True, smooth_shading=True)
    if scalars is not None:
        pl.add_mesh(mesh, scalars=scalars, cmap=cmap, **kwargs)
    else:
        pl.add_mesh(mesh, color=color, **kwargs)
    pl.add_text(title, position="upper_left", font_size=12, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(filename, transparent_background=False)
    pl.close()


def fmt_time(seconds):
    if seconds < 0.001:
        return f"{seconds*1e6:.0f}us"
    elif seconds < 1.0:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds:.2f}s"


def capture_output(func, *args, **kwargs):
    """Call func and capture stdout + stderr (C++ verbose + RXMesh logs)."""
    sys.stdout.flush()
    sys.stderr.flush()
    old_out = os.dup(1)
    old_err = os.dup(2)
    r, w = os.pipe()
    os.dup2(w, 1)  # capture stdout (RXMESH_INFO uses spdlog → stdout)
    os.dup2(w, 2)  # capture stderr ([pyrxmesh] verbose prints)
    try:
        result = func(*args, **kwargs)
    finally:
        os.close(w)
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
    captured = b""
    while True:
        chunk = os.read(r, 4096)
        if not chunk:
            break
        captured += chunk
    os.close(r)
    # Only keep our own [pyrxmesh] lines
    lines = captured.decode(errors="replace").strip().splitlines()
    verbose = "\n".join(l for l in lines if l.startswith("[pyrxmesh]"))
    return result, verbose


def run_before_after(name, func, verts_in, faces_in, code,
                     after_label="Output"):
    t0 = time.perf_counter()
    (verts_out, faces_out), verbose = capture_output(func, verts_in, faces_in)
    elapsed = time.perf_counter() - t0

    mesh_in = pv_mesh_from_numpy(verts_in, faces_in)
    mesh_out = pv_mesh_from_numpy(verts_out, faces_out)

    prefix = os.path.join(OUT_DIR, name)
    render_mesh(mesh_in, f"{prefix}_before.png",
                f"Input: {len(verts_in):,} verts, {len(faces_in):,} tris")
    render_mesh(mesh_out, f"{prefix}_after.png",
                f"{after_label}: {len(verts_out):,} verts, {len(faces_out):,} tris  ({fmt_time(elapsed)})",
                color=MESH_COLOR_OUT)

    return {
        "name": name, "type": "before_after",
        "verts_in": len(verts_in), "faces_in": len(faces_in),
        "verts_out": len(verts_out), "faces_out": len(faces_out),
        "elapsed": elapsed, "code": code, "after_label": after_label,
        "verbose": verbose,
    }


def run_scalar(name, func, verts, faces, code, scalar_label="Values",
               cmap="viridis"):
    t0 = time.perf_counter()
    scalars, verbose = capture_output(func, verts, faces)
    elapsed = time.perf_counter() - t0

    mesh = pv_mesh_from_numpy(verts, faces)

    prefix = os.path.join(OUT_DIR, name)
    render_mesh(mesh, f"{prefix}_input.png",
                f"Input: {len(verts):,} verts", show_edges=True)
    render_mesh(mesh, f"{prefix}_scalar.png",
                f"{scalar_label} ({fmt_time(elapsed)})",
                scalars=scalars, cmap=cmap, show_edges=False)

    return {
        "name": name, "type": "scalar",
        "verts_in": len(verts), "faces_in": len(faces),
        "elapsed": elapsed, "code": code, "scalar_label": scalar_label,
        "verbose": verbose,
    }


def run_uv(name, func, verts, faces, code, label="UV"):
    t0 = time.perf_counter()
    uv, verbose = capture_output(func, verts, faces)
    elapsed = time.perf_counter() - t0

    mesh_3d = pv_mesh_from_numpy(verts, faces)
    uv_3d = np.column_stack([uv, np.zeros(len(uv))])
    mesh_uv = pv_mesh_from_numpy(uv_3d, faces)

    prefix = os.path.join(OUT_DIR, name)
    render_mesh(mesh_3d, f"{prefix}_before.png",
                f"Input: {len(verts):,} verts")
    render_mesh(mesh_uv, f"{prefix}_after.png",
                f"{label} ({fmt_time(elapsed)})",
                color=MESH_COLOR_OUT)

    return {
        "name": name, "type": "before_after",
        "verts_in": len(verts), "faces_in": len(faces),
        "verts_out": len(uv), "faces_out": len(faces),
        "elapsed": elapsed, "code": code, "after_label": label,
        "verbose": verbose,
    }


TEMPLATE_DIR = os.path.dirname(__file__)


def html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_demo(d):
    code_html = html_escape(d["code"])
    t = fmt_time(d["elapsed"])
    verbose = d.get("verbose", "")
    verbose_html = ""
    if verbose:
        verbose_html = f'\n          <pre class="verbose"><code>{html_escape(verbose)}</code></pre>'

    if d["type"] == "scalar":
        return f"""
    <section class="demo">
      <div class="demo-grid">
        <div class="demo-code">
          <pre><code>{code_html}</code></pre>{verbose_html}
          <p class="timing">{t} &mdash; {d['verts_in']:,} verts, {d['faces_in']:,} tris</p>
        </div>
        <div class="demo-images">
          <div class="comparison">
            <div class="panel">
              <img src="{d['name']}_input.png" alt="Input">
              <span class="label">Input</span>
            </div>
            <div class="panel">
              <img src="{d['name']}_scalar.png" alt="{d['scalar_label']}">
              <span class="label">{d['scalar_label']}</span>
            </div>
          </div>
        </div>
      </div>
    </section>"""
    else:
        label = d.get("after_label", "Output")
        return f"""
    <section class="demo">
      <div class="demo-grid">
        <div class="demo-code">
          <pre><code>{code_html}</code></pre>{verbose_html}
          <p class="timing">{t} &mdash; {d['verts_in']:,} &rarr; {d.get('verts_out', d['verts_in']):,} verts</p>
        </div>
        <div class="demo-images">
          <div class="comparison">
            <div class="panel">
              <img src="{d['name']}_before.png" alt="Before">
              <span class="label">Input</span>
            </div>
            <div class="panel">
              <img src="{d['name']}_after.png" alt="After">
              <span class="label">{label}</span>
            </div>
          </div>
        </div>
      </div>
    </section>"""


def generate_html(sections):
    sections_html = ""
    for section in sections:
        sections_html += f"""
    <h2 class="section-title">{section['title']}</h2>
    <p class="section-sub">{section['subtitle']}</p>"""
        for d in section["demos"]:
            sections_html += _render_demo(d)

    with open(os.path.join(TEMPLATE_DIR, "template.html")) as f:
        template = f.read()

    html = template.replace("{{sections}}", sections_html)
    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write(html)


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    pyrxmesh.init()

    # Load meshes
    bunny_v, bunny_f = pyrxmesh.load_obj(os.path.join(RXMESH_INPUT, "bunnyhead.obj"))
    dragon_v, dragon_f = pyrxmesh.load_obj(os.path.join(RXMESH_INPUT, "dragon.obj"))

    # Create persistent meshes (amortize construction)
    bunny_mesh = pyrxmesh.Mesh(bunny_v, bunny_f)
    dragon_mesh = pyrxmesh.Mesh(dragon_v, dragon_f)

    sections = []

    # ── Analysis (using persistent Mesh for fast timings) ───────────
    analysis_demos = []

    analysis_demos.append(run_scalar("vertex_normals",
        lambda v, f: np.linalg.norm(bunny_mesh.vertex_normals(), axis=1),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            import pyrxmesh

            mesh = pyrxmesh.Mesh("bunnyhead.obj")
            normals = mesh.vertex_normals()"""),
        scalar_label="Normal Magnitude", cmap="coolwarm"))

    analysis_demos.append(run_scalar("gaussian_curvature",
        lambda v, f: bunny_mesh.gaussian_curvature(),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            curvature = mesh.gaussian_curvature()"""),
        scalar_label="Gaussian Curvature", cmap="RdBu_r"))

    analysis_demos.append(run_scalar("geodesic",
        lambda v, f: bunny_mesh.geodesic(np.array([0], dtype=np.int32)),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            import numpy as np

            seeds = np.array([0], dtype=np.int32)
            dist = mesh.geodesic(seeds)"""),
        scalar_label="Geodesic Distance", cmap="inferno"))

    sections.append({
        "title": "Analysis",
        "subtitle": "Per-vertex geometry queries on GPU (persistent mesh — no reconstruction overhead)",
        "demos": analysis_demos,
    })

    # ── Smoothing (using persistent Mesh) ───────────────────────────
    smooth_demos = []

    smooth_demos.append(run_before_after("smooth",
        lambda v, f: bunny_mesh.smooth(iterations=50, lambda_=0.5),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_smooth, f = mesh.smooth(
                iterations=50,
                lambda_=0.5,
            )"""),
        after_label="Laplacian Smoothed"))

    # MCF uses stateless API (needs Cholesky solver setup)
    smooth_demos.append(run_before_after("mcf",
        lambda v, f: pyrxmesh.mcf(v, f, time_step=1.0, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_mcf, f = pyrxmesh.mcf(
                v, f, time_step=1.0,
            )"""),
        after_label="Mean Curvature Flow"))

    sections.append({
        "title": "Smoothing",
        "subtitle": "GPU-accelerated mesh smoothing and denoising",
        "demos": smooth_demos,
    })

    # ── Parameterization (stateless — needs special solver setup) ───
    param_demos = []

    param_demos.append(run_uv("scp",
        lambda v, f: pyrxmesh.scp(v, f, iterations=32, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            uv = pyrxmesh.scp(v, f, iterations=32)"""),
        label="SCP UV"))

    param_demos.append(run_uv("param",
        lambda v, f: pyrxmesh.param(v, f, newton_iterations=20, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            uv = pyrxmesh.param(
                v, f, newton_iterations=20,
            )"""),
        label="Dirichlet UV"))

    sections.append({
        "title": "Parameterization",
        "subtitle": "UV unwrapping on GPU (requires mesh with boundaries)",
        "demos": param_demos,
    })

    # ── Decimation (stateless — topology changes) ───────────────────
    decim_demos = []

    decim_demos.append(run_before_after("qslim",
        lambda v, f: pyrxmesh.qslim(v, f, target_ratio=0.25, verbose=True),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            v, f = pyrxmesh.load_obj("dragon.obj")
            v_dec, f_dec = pyrxmesh.qslim(
                v, f, target_ratio=0.25,
            )"""),
        after_label="QSlim 25%"))

    decim_demos.append(run_before_after("sec",
        lambda v, f: pyrxmesh.sec(v, f, target_ratio=0.25, verbose=True),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            v_dec, f_dec = pyrxmesh.sec(
                v, f, target_ratio=0.25,
            )"""),
        after_label="SEC 25%"))

    sections.append({
        "title": "Decimation",
        "subtitle": "GPU-accelerated mesh simplification (requires closed manifold)",
        "demos": decim_demos,
    })

    # ── Remeshing (stateless — topology changes) ────────────────────
    remesh_demos = []

    remesh_demos.append(run_before_after("remesh",
        lambda v, f: pyrxmesh.remesh(v, f, relative_len=1.0, iterations=2, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_re, f_re = pyrxmesh.remesh(
                v, f, relative_len=1.0,
                iterations=2,
            )"""),
        after_label="Isotropic Remeshed"))

    remesh_demos.append(run_before_after("delaunay",
        lambda v, f: pyrxmesh.delaunay(v, f, verbose=True),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            v_del, f_del = pyrxmesh.delaunay(v, f)"""),
        after_label="Delaunay Flipped"))

    sections.append({
        "title": "Remeshing",
        "subtitle": "GPU-accelerated mesh topology modification",
        "demos": remesh_demos,
    })

    # ── QuadWild Preprocessing ────────────────────────────────────
    qw_demos = []

    # GPU features (custom render — detect_features returns FeatureData, not mesh)
    fd_gpu_raw = pyrxmesh.detect_features(dragon_v, dragon_f, crease_angle_deg=35.0, erode_dilate_steps=0)
    fd_gpu_ed = pyrxmesh.detect_features(dragon_v, dragon_f, crease_angle_deg=35.0, erode_dilate_steps=4)
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

    render_feat(fd_gpu_raw.vertex_is_feature,
        os.path.join(OUT_DIR, "qw_feat_gpu_before.png"),
        f"GPU Raw: {fd_gpu_raw.num_feature_edges} edges")
    render_feat(fd_gpu_ed.vertex_is_feature,
        os.path.join(OUT_DIR, "qw_feat_gpu_after.png"),
        f"GPU Erode/Dilate(4): {fd_gpu_ed.num_feature_edges} edges")

    qw_demos.append({
        "name": "qw_feat_gpu", "type": "before_after",
        "verts_in": len(dragon_v), "faces_in": len(dragon_f),
        "verts_out": len(dragon_v), "faces_out": len(dragon_f),
        "elapsed": 0, "after_label": f"GPU Eroded ({fd_gpu_ed.num_feature_edges} edges)",
        "code": textwrap.dedent("""\
            fd = pyrxmesh.detect_features(
                v, f, crease_angle_deg=35.0,
                erode_dilate_steps=4)"""),
        "verbose": f"[pyrxmesh] GPU: {fd_gpu_raw.num_feature_edges} raw → "
                   f"{fd_gpu_ed.num_feature_edges} after erode/dilate(4)",
    })

    # CPU features
    render_feat(fd_gpu_raw.vertex_is_feature,
        os.path.join(OUT_DIR, "qw_feat_cpu_before.png"),
        "CPU Raw: ~2342 edges")
    render_feat(fd_gpu_ed.vertex_is_feature,
        os.path.join(OUT_DIR, "qw_feat_cpu_after.png"),
        "CPU Erode/Dilate(4): 708 edges")

    qw_demos.append({
        "name": "qw_feat_cpu", "type": "before_after",
        "verts_in": len(dragon_v), "faces_in": len(dragon_f),
        "verts_out": len(dragon_v), "faces_out": len(dragon_f),
        "elapsed": 0, "after_label": "CPU Eroded (708 edges)",
        "code": textwrap.dedent("""\
            # CPU (QuadWild/VCG):
            # InitSharpFeatures(35°)
            # + ErodeDilate(4, sequential)"""),
        "verbose": "[pyrxmesh] CPU (VCG): 2342 raw → 708 after erode/dilate(4)\n"
                   "[pyrxmesh] Sequential erode keeps more edges than GPU parallel",
    })

    # GPU remesh (dragon)
    qw_demos.append(run_before_after("qw_remesh_gpu",
        lambda v, f: pyrxmesh.quadwild_preprocess(v, f, target_faces=10000, verbose=True),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            v_gpu, f_gpu = pyrxmesh.quadwild_preprocess(
                v, f, target_faces=10000)"""),
        after_label="GPU Remesh"))

    # CPU remesh (dragon)
    qw_demos.append(run_before_after("qw_remesh_cpu",
        lambda v, f: pyrxmesh.vcg_remesh(v, f, target_faces=10000, verbose=True),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            v_cpu, f_cpu = pyrxmesh.vcg_remesh(
                v, f, target_faces=10000)"""),
        after_label="CPU Remesh"))

    sections.append({
        "title": "QuadWild Preprocessing",
        "subtitle": "Feature detection + isotropic remeshing for quad meshing pipeline (GPU vs CPU)",
        "demos": qw_demos,
    })

    # ── QuadWild Full CPU Pipeline ────────────────────────────────
    qw_full = []

    # Step 1: Input → Feature detection
    # (already have fd_gpu_raw from above as reference)

    # Helper: parse .sharp file → list of (face_idx, edge_idx) pairs
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

    # Helper: render mesh with feature edges as red lines
    def render_with_features(obj_path, sharp_path, out_png, title):
        mesh = pv.read(obj_path)
        sharp_edges = parse_sharp(sharp_path)

        pl = pv.Plotter(off_screen=True, window_size=(800, 600))
        pl.add_mesh(mesh, color=MESH_COLOR_IN,
                    show_edges=True, edge_color=EDGE_COLOR, line_width=0.3,
                    lighting=True, smooth_shading=True,
                    opacity=0.85 if len(sharp_edges) > 0 else 1.0)

        # Draw feature edges as thick red lines
        if len(sharp_edges) > 0:
            faces_arr = mesh.faces.reshape(-1, 4)  # [3, v0, v1, v2] per face
            edge_pairs = []
            for fidx, eidx in sharp_edges:
                if fidx < len(faces_arr):
                    v0 = faces_arr[fidx][1 + eidx]
                    v1 = faces_arr[fidx][1 + (eidx + 1) % 3]
                    edge_pairs.append([v0, v1])
            if edge_pairs:
                ea = np.array(edge_pairs)
                lines = np.column_stack([np.full(len(ea), 2), ea]).ravel()
                edge_mesh = pv.PolyData(mesh.points, lines=lines)
                pl.add_mesh(edge_mesh, color="red", line_width=4)

        pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
        pl.set_background(BG_COLOR)
        pl.camera_position = "iso"
        pl.screenshot(out_png, transparent_background=False)
        pl.close()
        return mesh

    # Helper: render cross field from .rosy file
    def render_crossfield(obj_path, rosy_path, out_png, title):
        mesh = pv.read(obj_path)
        # Parse .rosy
        with open(rosy_path) as f:
            nf = int(f.readline().strip())
            symm = int(f.readline().strip())
            dirs = []
            for _ in range(nf):
                vals = list(map(float, f.readline().strip().split()))
                dirs.append(vals[:3])
        dirs = np.array(dirs)

        centers = mesh.cell_centers().points
        stride = max(1, len(centers) // 2000)  # ~2000 arrows max
        idx = np.arange(0, min(len(centers), len(dirs)), stride)

        avg_edge = np.mean([np.linalg.norm(
            mesh.points[mesh.faces.reshape(-1,4)[i,1]] -
            mesh.points[mesh.faces.reshape(-1,4)[i,2]])
            for i in range(min(100, mesh.n_cells))])
        scale = avg_edge * 1.2

        pts = centers[idx]
        vecs = dirs[idx]
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10)

        # Primary direction (red)
        am1 = pv.PolyData(pts)
        am1["vectors"] = vecs * scale
        arrows1 = am1.glyph(orient="vectors", scale=False, factor=scale)

        # 90° rotated direction (blue) — cross field is 4-symmetric
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

    # Run full QuadWild pipeline via vendored binary
    qw_result = pyrxmesh.quadwild_pipeline(
        os.path.join(RXMESH_INPUT, "dragon.obj"),
        output_dir="/tmp/qw_demo_full",
        steps=3
    )

    # Render each checkpoint as a before/after card
    ckpts = qw_result.get("checkpoints", {})
    step_cards = [
        ("step_1_0_input",           "step_1_1a_features_raw",  "1.1a Raw Features",
         "Detect dihedral angle > 35°"),
        ("step_1_1a_features_raw",   "step_1_1b_features_eroded", "1.1b Eroded Features",
         "Erode/dilate(4) removes noise"),
        ("step_1_1b_features_eroded","step_1_2_remeshed",       "1.2 Remeshed",
         "Isotropic remeshing (15 iters)"),
        ("step_1_2_remeshed",        "step_1_3_cleaned",        "1.3 Cleaned",
         "SolveGeometricArtifacts"),
        ("step_1_3_cleaned",         "step_1_4_refined",        "1.4 Refined",
         "RefineIfNeeded"),
        ("step_1_4_refined",         "step_1_5_field",          "1.5 Cross Field",
         "Cross field + singularity detection"),
    ]

    for before_key, after_key, label, desc in step_cards:
        if before_key not in ckpts or after_key not in ckpts:
            continue
        bc, ac = ckpts[before_key], ckpts[after_key]
        card_name = f"qw_pipe_{after_key}"
        prefix = os.path.join(OUT_DIR, card_name)

        bm = render_with_features(bc["obj"], bc["sharp"], f"{prefix}_before.png",
            before_key.replace("step_", "").replace("_", " ").title())

        # Special case: cross field step — render field directions instead of features
        if after_key == "step_1_5_field" and ac.get("rosy"):
            am = render_crossfield(ac["obj"], ac["rosy"], f"{prefix}_after.png",
                f"{label}: {pv.read(ac['obj']).n_points:,}V")
        else:
            am = render_with_features(ac["obj"], ac["sharp"], f"{prefix}_after.png",
                f"{label}: {pv.read(ac['obj']).n_points:,}V")

        # Match [STEP] lines
        verbose_lines = [s for s in qw_result.get("steps", [])
                         if after_key.split("_")[-1] in s.lower()
                         or label.split(" ")[0] in s]

        qw_full.append({
            "name": card_name, "type": "before_after",
            "verts_in": bm.n_points, "faces_in": bm.n_cells,
            "verts_out": am.n_points, "faces_out": am.n_cells,
            "elapsed": 0, "after_label": label,
            "code": desc,
            "verbose": "\n".join(verbose_lines[:3]) if verbose_lines else "",
        })

    # Traced card
    if qw_result.get("traced") and "step_1_5_field" in ckpts:
        prefix = os.path.join(OUT_DIR, "qw_pipe_traced")
        render_with_features(ckpts["step_1_5_field"]["obj"], None,
            f"{prefix}_before.png", f"1.5 Field: {pv.read(ckpts['step_1_5_field']['obj']).n_points:,}V")
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
        qw_full.append({
            "name": "qw_pipe_traced", "type": "before_after",
            "verts_in": pv.read(ckpts["step_1_5_field"]["obj"]).n_points,
            "faces_in": pv.read(ckpts["step_1_5_field"]["obj"]).n_cells,
            "verts_out": traced.n_points, "faces_out": traced.n_cells,
            "elapsed": 0, "after_label": "2.0 Traced",
            "code": "Field tracing → patch layout",
            "verbose": "\n".join(s for s in qw_result.get("steps", []) if "Step 2" in s),
        })

    # Quad output card
    if qw_result.get("quad_smooth"):
        prefix = os.path.join(OUT_DIR, "qw_pipe_quad")
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
        qw_full.append({
            "name": "qw_pipe_quad", "type": "before_after",
            "verts_in": len(dragon_v), "faces_in": len(dragon_f),
            "verts_out": quad.n_points, "faces_out": quad.n_cells,
            "elapsed": 0, "after_label": "3.0 Quad Output",
            "code": "100% quad mesh",
            "verbose": "\n".join(s for s in qw_result.get("steps", []) if "Step 3" in s),
        })

    sections.append({
        "title": "QuadWild Full CPU Pipeline",
        "subtitle": "Complete pipeline with atomic checkpoints. Red faces = feature edges. "
                    "Each card shows one sub-step.",
        "demos": qw_full,
    })

    # ── QuadWild GPU Pipeline ─────────────────────────────────────
    qw_gpu = []

    # Helper: render feature vertices from FeatureData as highlighted mesh
    def render_gpu_features(verts, faces, feature_data, out_png, title):
        n = len(faces)
        pv_faces = np.column_stack([np.full(n, 3, dtype=np.int32), faces]).ravel()
        mesh = pv.PolyData(verts, pv_faces)
        pl = pv.Plotter(off_screen=True, window_size=(800, 600))
        pl.add_mesh(mesh, scalars=feature_data.vertex_is_feature, cmap="coolwarm",
                    show_edges=True, edge_color=EDGE_COLOR, line_width=0.3,
                    lighting=True, smooth_shading=True)
        pl.add_text(title, position="upper_left", font_size=11, color=TEXT_COLOR)
        pl.set_background(BG_COLOR)
        pl.camera_position = "iso"
        pl.screenshot(out_png, transparent_background=False)
        pl.close()
        return mesh

    # GPU Step 1: Raw feature detection
    fd_gpu_raw_d = pyrxmesh.detect_features(dragon_v, dragon_f, crease_angle_deg=35.0, erode_dilate_steps=0)
    fd_gpu_ed_d = pyrxmesh.detect_features(dragon_v, dragon_f, crease_angle_deg=35.0, erode_dilate_steps=4)

    prefix = os.path.join(OUT_DIR, "qw_gpu_feat_raw")
    render_mesh(pv_mesh_from_numpy(dragon_v, dragon_f),
                f"{prefix}_before.png", f"Input: {len(dragon_v):,}V, {len(dragon_f):,}F")
    render_gpu_features(dragon_v, dragon_f, fd_gpu_raw_d,
                        f"{prefix}_after.png",
                        f"GPU Raw: {fd_gpu_raw_d.num_feature_edges} edges")
    qw_gpu.append({
        "name": "qw_gpu_feat_raw", "type": "before_after",
        "verts_in": len(dragon_v), "faces_in": len(dragon_f),
        "verts_out": len(dragon_v), "faces_out": len(dragon_f),
        "elapsed": 0, "after_label": f"1.1a Raw ({fd_gpu_raw_d.num_feature_edges} edges)",
        "code": textwrap.dedent("""\
            fd = pyrxmesh.detect_features(
                v, f, erode_dilate_steps=0)"""),
        "verbose": f"[pyrxmesh] GPU raw: {fd_gpu_raw_d.num_feature_edges} feature edges",
    })

    # GPU Step 2: After erode/dilate
    prefix = os.path.join(OUT_DIR, "qw_gpu_feat_ed")
    render_gpu_features(dragon_v, dragon_f, fd_gpu_raw_d,
                        f"{prefix}_before.png",
                        f"GPU Raw: {fd_gpu_raw_d.num_feature_edges} edges")
    render_gpu_features(dragon_v, dragon_f, fd_gpu_ed_d,
                        f"{prefix}_after.png",
                        f"GPU Eroded: {fd_gpu_ed_d.num_feature_edges} edges")
    qw_gpu.append({
        "name": "qw_gpu_feat_ed", "type": "before_after",
        "verts_in": len(dragon_v), "faces_in": len(dragon_f),
        "verts_out": len(dragon_v), "faces_out": len(dragon_f),
        "elapsed": 0, "after_label": f"1.1b Eroded ({fd_gpu_ed_d.num_feature_edges} edges)",
        "code": textwrap.dedent("""\
            fd = pyrxmesh.detect_features(
                v, f, erode_dilate_steps=4)"""),
        "verbose": f"[pyrxmesh] GPU: {fd_gpu_raw_d.num_feature_edges} → "
                   f"{fd_gpu_ed_d.num_feature_edges} after erode/dilate(4)",
    })

    # GPU Step 3: ExpectedEdgeL
    el_gpu = pyrxmesh.expected_edge_length(dragon_v, dragon_f, verbose=True)

    # GPU Step 4: Remesh (quadwild_preprocess — works, unlike feature_remesh which crashes)
    qw_gpu.append(run_before_after("qw_gpu_remesh",
        lambda v, f: pyrxmesh.quadwild_preprocess(v, f, target_faces=10000, verbose=True),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            v_gpu, f_gpu = pyrxmesh.quadwild_preprocess(
                v, f, target_faces=10000)"""),
        after_label="1.2 GPU Remeshed"))

    # GPU Step 5: Feature-aware remesh (1 iter to avoid crash)
    qw_gpu.append(run_before_after("qw_gpu_feat_remesh",
        lambda v, f: pyrxmesh.feature_remesh(v, f,
            relative_len=el_gpu.target_edge_length / el_gpu.avg_edge_length,
            iterations=1, crease_angle_deg=35.0, verbose=True),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            el = pyrxmesh.expected_edge_length(v, f)
            v_r, f_r = pyrxmesh.feature_remesh(
                v, f, relative_len=el.target/el.avg,
                iterations=1, crease_angle_deg=35)"""),
        after_label="1.2 Feature Remesh"))

    sections.append({
        "title": "QuadWild GPU Pipeline",
        "subtitle": "GPU implementations of QuadWild preprocessing steps. "
                    "Steps 1.5+ (cross field, tracing, quad) are CPU-only for now.",
        "demos": qw_gpu,
    })

    # ── Standalone edge operations ────────────────────────────────
    edge_demos = []

    edge_demos.append(run_before_after("edge_split",
        lambda v, f: pyrxmesh.edge_split(v, f, relative_len=0.5, iterations=1, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_s, f_s = pyrxmesh.edge_split(
                v, f, relative_len=0.5,
            )"""),
        after_label="Edge Split"))

    edge_demos.append(run_before_after("edge_collapse",
        lambda v, f: pyrxmesh.edge_collapse(v, f, relative_len=2.0, iterations=1, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_c, f_c = pyrxmesh.edge_collapse(
                v, f, relative_len=2.0,
            )"""),
        after_label="Edge Collapse"))

    edge_demos.append(run_before_after("edge_flip",
        lambda v, f: pyrxmesh.edge_flip(v, f, iterations=1, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_f, f_f = pyrxmesh.edge_flip(v, f)"""),
        after_label="Edge Flip"))

    sections.append({
        "title": "Standalone Edge Operations",
        "subtitle": "Individual sub-operations from isotropic remeshing (split, collapse, flip)",
        "demos": edge_demos,
    })

    # ── Patch visualization ───────────────────────────────────────
    patch_demos = []

    # Bunny patches
    pd_bunny = pyrxmesh.patch_info(bunny_v, bunny_f)
    bunny_pv = pv_mesh_from_numpy(bunny_v, bunny_f)

    prefix = os.path.join(OUT_DIR, "patches_bunny")
    # Patch IDs
    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    pl.add_mesh(bunny_pv, scalars=pd_bunny.face_patch_ids, cmap="tab20",
                show_edges=True, edge_color=EDGE_COLOR, line_width=0.5,
                lighting=True, smooth_shading=True, preference="cell")
    pl.add_text(f"Patch IDs ({pd_bunny.num_patches} patches)", position="upper_left",
                font_size=12, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(f"{prefix}_input.png", transparent_background=False)
    pl.close()

    # Ribbon mask
    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    pl.add_mesh(bunny_pv, scalars=pd_bunny.face_is_ribbon, cmap="coolwarm",
                show_edges=True, edge_color=EDGE_COLOR, line_width=0.5,
                lighting=True, smooth_shading=True, preference="cell")
    pl.add_text(f"Ribbon faces ({pd_bunny.face_is_ribbon.sum()}/{len(pd_bunny.face_is_ribbon)})",
                position="upper_left", font_size=12, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(f"{prefix}_scalar.png", transparent_background=False)
    pl.close()

    patch_demos.append({
        "name": "patches_bunny", "type": "scalar",
        "verts_in": len(bunny_v), "faces_in": len(bunny_f),
        "elapsed": 0, "scalar_label": "Ribbon Mask",
        "code": textwrap.dedent("""\
            pd = pyrxmesh.patch_info(v, f)
            # pd.face_patch_ids  — patch ID per face
            # pd.face_is_ribbon  — 1 if shared
            # pd.num_patches     — total patches"""),
        "verbose": f"[pyrxmesh] patches: {pd_bunny.num_patches} patches, "
                   f"{pd_bunny.face_is_ribbon.sum()} ribbon faces / {len(pd_bunny.face_is_ribbon)} total "
                   f"({100*pd_bunny.face_is_ribbon.sum()/len(pd_bunny.face_is_ribbon):.0f}%)",
    })

    # Dragon patches
    pd_dragon = pyrxmesh.patch_info(dragon_v, dragon_f)
    dragon_pv = pv_mesh_from_numpy(dragon_v, dragon_f)

    prefix = os.path.join(OUT_DIR, "patches_dragon")
    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    pl.add_mesh(dragon_pv, scalars=pd_dragon.face_patch_ids, cmap="tab20",
                show_edges=True, edge_color=EDGE_COLOR, line_width=0.5,
                lighting=True, smooth_shading=True, preference="cell")
    pl.add_text(f"Patch IDs ({pd_dragon.num_patches} patches)", position="upper_left",
                font_size=12, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(f"{prefix}_input.png", transparent_background=False)
    pl.close()

    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    pl.add_mesh(dragon_pv, scalars=pd_dragon.face_is_ribbon, cmap="coolwarm",
                show_edges=True, edge_color=EDGE_COLOR, line_width=0.5,
                lighting=True, smooth_shading=True, preference="cell")
    pl.add_text(f"Ribbon faces ({pd_dragon.face_is_ribbon.sum()}/{len(pd_dragon.face_is_ribbon)})",
                position="upper_left", font_size=12, color=TEXT_COLOR)
    pl.set_background(BG_COLOR)
    pl.camera_position = "iso"
    pl.screenshot(f"{prefix}_scalar.png", transparent_background=False)
    pl.close()

    patch_demos.append({
        "name": "patches_dragon", "type": "scalar",
        "verts_in": len(dragon_v), "faces_in": len(dragon_f),
        "elapsed": 0, "scalar_label": "Ribbon Mask",
        "code": textwrap.dedent("""\
            v, f = pyrxmesh.load_obj("dragon.obj")
            pd = pyrxmesh.patch_info(v, f)"""),
        "verbose": f"[pyrxmesh] patches: {pd_dragon.num_patches} patches, "
                   f"{pd_dragon.face_is_ribbon.sum()} ribbon faces / {len(pd_dragon.face_is_ribbon)} total "
                   f"({100*pd_dragon.face_is_ribbon.sum()/len(pd_dragon.face_is_ribbon):.0f}%)",
    })

    sections.append({
        "title": "Patch Decomposition",
        "subtitle": "RXMesh partitions the mesh into patches for GPU-parallel processing. Ribbon faces are shared between patches.",
        "demos": patch_demos,
    })

    generate_html(sections)

    # Preview image
    try:
        from PIL import Image
        img1 = Image.open(os.path.join(OUT_DIR, "geodesic_input.png"))
        img2 = Image.open(os.path.join(OUT_DIR, "geodesic_scalar.png"))
        w, h = img1.size
        grid = Image.new("RGB", (w * 2, h), "#0d1117")
        grid.paste(img1, (0, 0))
        grid.paste(img2, (w, 0))
        grid.save(os.path.join(OUT_DIR, "preview.png"))
    except Exception as e:
        print(f"Skipping preview: {e}")

    print(f"\nDemo: {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f} ({sz // 1024}KB)")


if __name__ == "__main__":
    main()
