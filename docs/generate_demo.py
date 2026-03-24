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
import faulthandler
import datetime
import numpy as np

faulthandler.enable()

import pyvista as pv

pv.OFF_SCREEN = True

import pyrxmesh

OUT_DIR = os.path.join(os.path.dirname(__file__), "_site")
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
RXMESH_INPUT = os.path.join(os.path.dirname(__file__), "..", "RXMesh", "input")


class TeeLogger:
    """Write to both terminal and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

    def write(self, msg):
        self.terminal.write(msg)
        self.terminal.flush()
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logging():
    os.makedirs(RUNS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(RUNS_DIR, f"run_{timestamp}.log")
    tee = TeeLogger(log_path)
    sys.stdout = tee
    sys.stderr = tee
    print(f"Logging to {log_path}")
    print("NOTE: C++ stderr goes to terminal only. Run with '2>&1 | tee ...' to capture.")
    return tee

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
    nV = mesh.n_points
    nF = mesh.n_cells
    nE = mesh.extract_all_edges().n_cells if nF > 0 else 0
    subtitle = f"{nV:,}V  {nE:,}E  {nF:,}F"

    # Save VTK
    vtk_path = filename.replace(".png", ".vtk")
    mesh.save(vtk_path)

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    kwargs = dict(show_edges=show_edges, edge_color=EDGE_COLOR,
                  line_width=0.5, lighting=True, smooth_shading=True)
    if scalars is not None:
        pl.add_mesh(mesh, scalars=scalars, cmap=cmap, **kwargs)
    else:
        pl.add_mesh(mesh, color=color, **kwargs)
    pl.add_text(title, position="upper_left", font_size=12, color=TEXT_COLOR)
    pl.add_text(subtitle, position="upper_right", font_size=10, color="#8b949e")
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
    """Call func, letting C++ output go to terminal. Returns (result, verbose_str)."""
    result = func(*args, **kwargs)
    return result, ""


def run_before_after(name, func, verts_in, faces_in, code,
                     after_label="Output"):
    t0 = time.perf_counter()
    try:
        (verts_out, faces_out), verbose = capture_output(func, verts_in, faces_in)
    except Exception:
        # capture_output can interfere with CUDA's stderr; fall back to direct call
        verbose = ""
        verts_out, faces_out = func(verts_in, faces_in)
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


def _render_table(demos):
    """Render demos as a CPU vs GPU comparison table with 4 result columns.

    Demos should have "side" (cpu/gpu) and "step_name" fields.
    Each side shows input (before) and result (after) images.
    Demos with the same step_name are paired into one row.
    """
    from collections import OrderedDict
    steps = OrderedDict()
    for d in demos:
        step = d.get("step_name", d.get("after_label", ""))
        if step not in steps:
            steps[step] = {}
        side = d.get("side", "gpu")
        steps[step][side] = d

    rows = ""
    for step, sides in steps.items():
        step_html = html_escape(step)
        cells = []
        for side in ("cpu", "gpu"):
            d = sides.get(side)
            if d is None:
                cells.append("<td>&mdash;</td><td>&mdash;</td>")
                continue
            timing = fmt_time(d["elapsed"]) if d["elapsed"] > 0 else ""
            timing_html = f'<span class="cell-label">{timing}</span>' if timing else ""
            if d["type"] == "before_after":
                before_img = f"{d['name']}_before.png"
                after_img = f"{d['name']}_after.png"
            elif d["type"] == "scalar":
                before_img = f"{d['name']}_input.png"
                after_img = f"{d['name']}_scalar.png"
            else:
                before_img = f"{d['name']}_before.png"
                after_img = f"{d['name']}_after.png"
            cells.append(
                f'<td><img src="{before_img}" alt="{side} input"></td>'
                f'<td><img src="{after_img}" alt="{side} result">{timing_html}</td>')

        rows += f"""
      <tr>
        <td>{step_html}</td>
        {cells[0]}
        {cells[1]}
      </tr>"""

    return f"""
    <table class="comparison-table">
      <thead><tr>
        <th>Step</th>
        <th>CPU Input</th><th>CPU Result</th>
        <th>GPU Input</th><th>GPU Result</th>
      </tr></thead>
      <tbody>{rows}
      </tbody>
    </table>"""


def generate_html(sections):
    cache_bust = int(time.time())
    sections_html = ""
    for section in sections:
        sections_html += f"""
    <h2 class="section-title">{section['title']}</h2>
    <p class="section-sub">{section['subtitle']}</p>"""
        if section.get("layout") == "table":
            sections_html += _render_table(section["demos"])
        else:
            for d in section["demos"]:
                sections_html += _render_demo(d)

    with open(os.path.join(TEMPLATE_DIR, "template.html")) as f:
        template = f.read()

    html = template.replace("{{sections}}", sections_html)
    # Cache-bust all image URLs
    html = html.replace('.png"', f'.png?v={cache_bust}"')
    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write(html)


def main():
    tee = setup_logging()

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    pyrxmesh.init()

    # Control which sections to run. Set via env vars.
    # PYRXMESH_DEMO_STOP_AFTER: stop after this section
    # PYRXMESH_DEMO_ONLY: run ONLY this section (skips everything else)
    # Options: "analysis", "smoothing", "parameterization", "decimation",
    #          "remeshing", "quadwild_erode", "quadwild_remesh", "quadwild_all", None
    STOP_AFTER = os.environ.get("PYRXMESH_DEMO_STOP_AFTER", None)
    ONLY = os.environ.get("PYRXMESH_DEMO_ONLY", None)
    if STOP_AFTER:
        print(f"*** STOP_AFTER={STOP_AFTER} — will generate partial demo ***")
    if ONLY:
        print(f"*** ONLY={ONLY} — will skip other sections ***")

    def should_run(section_name):
        """Check if a section should run based on ONLY filter."""
        if ONLY is None:
            return True
        if ONLY.startswith("quadwild") and section_name.startswith("quadwild"):
            return True
        return section_name == ONLY

    # Skip persistent mesh construction if we only need quadwild
    skip_persistent = ONLY is not None and ONLY.startswith("quadwild")

    print("Loading meshes...")
    bunny_v, bunny_f = pyrxmesh.load_obj(os.path.join(RXMESH_INPUT, "bunnyhead.obj"))
    dragon_v, dragon_f = pyrxmesh.load_obj(os.path.join(RXMESH_INPUT, "dragon.obj"))
    print(f"  bunny: {len(bunny_v)} verts, {len(bunny_f)} faces")
    print(f"  dragon: {len(dragon_v)} verts, {len(dragon_f)} faces")

    if not skip_persistent:
        print("Building persistent meshes...")
        bunny_mesh = pyrxmesh.Mesh(bunny_v, bunny_f)
        dragon_mesh = pyrxmesh.Mesh(dragon_v, dragon_f)
    else:
        bunny_mesh = dragon_mesh = None

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

    print("\n=== Generating: Analysis ===")
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

    if STOP_AFTER == "analysis":
        print("\n*** Stopping after analysis ***")
        generate_html(sections)
        return

    print("\n=== Generating: Smoothing ===")
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

    if STOP_AFTER == "smoothing":
        print("\n*** Stopping after smoothing ***")
        generate_html(sections)
        return

    print("\n=== Generating: Parameterization ===")
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

    if STOP_AFTER == "parameterization":
        print("\n*** Stopping after parameterization ***")
        generate_html(sections)
        return

    print("\n=== Generating: Decimation ===")
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

    if STOP_AFTER == "decimation":
        print("\n*** Stopping after decimation ***")
        generate_html(sections)
        return

    print("\n=== Generating: Remeshing ===")
    sections.append({
        "title": "Remeshing",
        "subtitle": "GPU-accelerated mesh topology modification",
        "demos": remesh_demos,
    })

    # ── QuadWild Pipeline: CPU vs GPU ────────────────────────────
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

        # Build edge index → vertex pair mapping from faces
        edge_map = {}  # (min_v, max_v) → edge_index
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

    if STOP_AFTER == "remeshing":
        print("\n*** Stopping after remeshing ***")
        generate_html(sections)
        return

    print("\n=== Generating: QuadWild Pipeline CPU vs GPU ===")
    print("  Running GPU feature detection...")
    # -- Run GPU operations --

    fd_gpu_raw = pyrxmesh.detect_features(dragon_v, dragon_f, crease_angle_deg=35.0, erode_dilate_steps=0)
    fd_gpu_ed = pyrxmesh.detect_features(dragon_v, dragon_f, crease_angle_deg=35.0, erode_dilate_steps=4)
    el_gpu = pyrxmesh.expected_edge_length(dragon_v, dragon_f, verbose=True)

    # -- Step 0: Compare target edge lengths --
    print(f"\n  === Target Edge Length Comparison ===")
    print(f"  GPU: target={el_gpu.target_edge_length:.6f}, avg={el_gpu.avg_edge_length:.6f}, "
          f"sphericity={el_gpu.sphericity:.4f}")
    # CPU computes this internally in vcg_remesh — the verbose output prints it.
    # Both use the same formula (expected_edge_length / IdealL0/IdealL1).

    print("  Running CPU QuadWild pipeline...")
    # -- Run CPU pipeline --

    qw_result = pyrxmesh.quadwild_pipeline(
        os.path.join(RXMESH_INPUT, "dragon.obj"),
        output_dir="/tmp/qw_demo_full",
        steps=3
    )
    ckpts = qw_result.get("checkpoints", {})

    # -- Helper: count edges in a .sharp file --
    def count_sharp_edges(sharp_path):
        if not sharp_path or not os.path.exists(sharp_path):
            return 0
        with open(sharp_path) as f:
            return int(f.readline().strip())

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
            "code": "Detect dihedral angle > 35°", "verbose": "",
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

    if STOP_AFTER == "quadwild_erode":
        sections.append({"title": "QuadWild Pipeline: CPU vs GPU",
            "subtitle": "Side-by-side comparison (partial — stopped after erode/dilate)",
            "layout": "table", "demos": qw_all})
        print("\n*** Stopping after quadwild erode ***")
        generate_html(sections)
        return

    print("  Rendering: Remesh Non-Adaptive (CPU + GPU)")
    # -- Step 3a: Remesh Non-Adaptive (CPU + GPU) --

    # CPU non-adaptive
    d = run_before_after("qw_cpu_remesh_p1",
        lambda v, f: pyrxmesh.vcg_remesh(v, f, target_faces=10000, adaptive=False, verbose=True),
        dragon_v, dragon_f,
        "pyrxmesh.vcg_remesh(v, f, target_faces=10000, adaptive=False)",
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

    if STOP_AFTER == "quadwild_remesh":
        sections.append({"title": "QuadWild Pipeline: CPU vs GPU",
            "subtitle": "Side-by-side comparison (partial — stopped after non-adaptive remesh)",
            "layout": "table", "demos": qw_all})
        print("\n*** Stopping after quadwild remesh ***")
        generate_html(sections)
        return

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

    # GPU adaptive — DISABLED: pass 2 crashes (collapse kernel illegal memory access)
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
            "code": "Field tracing → patch layout", "verbose": "",
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

    sections.append({
        "title": "QuadWild Pipeline: CPU vs GPU",
        "subtitle": "Side-by-side comparison of CPU (QuadWild binary) and GPU (pyrxmesh) pipeline steps. "
                    "Steps after Remesh are CPU-only for now.",
        "layout": "table",
        "demos": qw_all,
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

    if STOP_AFTER == "quadwild_all":
        print("\n*** Stopping after quadwild ***")
        generate_html(sections)
        return

    print("\n=== Generating: Standalone Edge Operations ===")
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

    print("\n=== Generating: Patch Decomposition ===")
    sections.append({
        "title": "Patch Decomposition",
        "subtitle": "RXMesh partitions the mesh into patches for GPU-parallel processing. Ribbon faces are shared between patches.",
        "demos": patch_demos,
    })

    print("\n=== Writing HTML ===")
    generate_html(sections)

    # Print summary of generated files
    import glob
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*")))
    print(f"\nDone! Generated {len(files)} files in {OUT_DIR}/")
    for f in files:
        size = os.path.getsize(f)
        print(f"  {os.path.basename(f)} ({size // 1024}KB)")

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
