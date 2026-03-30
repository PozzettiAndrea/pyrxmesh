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

OUT_DIR = os.path.join(os.path.dirname(__file__), "_site", "demo_pyrxmesh")
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

    # Open log file and redirect C-level stdout (fd 1) and stderr (fd 2)
    # so that fprintf from C++/CUDA also goes to the log.
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    # Save original fds so we can also write to terminal
    orig_stdout_fd = os.dup(1)
    orig_stderr_fd = os.dup(2)
    # Redirect fd 1 and fd 2 to log file
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)

    # Python stdout/stderr: tee to both terminal and the log file
    log_file = open(log_path, "a")  # append since C already writes there
    terminal_out = os.fdopen(orig_stdout_fd, "w")
    terminal_err = os.fdopen(orig_stderr_fd, "w")

    class Tee:
        def __init__(self, terminal, log):
            self.terminal = terminal
            self.log = log
        def write(self, msg):
            self.terminal.write(msg)
            self.terminal.flush()
            self.log.write(msg)
            self.log.flush()
            # Also write to fd 1 so it goes to the log file for C code
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    tee = Tee(terminal_out, log_file)
    sys.stdout = tee
    sys.stderr = Tee(terminal_err, log_file)

    print(f"Logging to {log_path}")
    # Log the command that was used to invoke this script
    import sys as _sys
    print(f"Command: {' '.join(_sys.argv)}")
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


def _fmt_metrics(m):
    """Format a metrics dict as a compact HTML snippet."""
    if not m:
        return ""
    parts = []
    parts.append(f"{m['V']:,}V {m['F']:,}F")
    parts.append(f"Q: {m['quality_avg']:.2f} avg, {m['quality_max']:.0f} max")
    parts.append(f"edge std: {m['edge_std']:.4f}")
    parts.append(f"val6: {m['pct_val6']:.0f}%")
    if "dist_avg" in m:
        parts.append(f"dist: avg={m['dist_avg']:.4f} max={m['dist_max']:.4f}")
    return "<br>".join(parts)


def _render_table(demos):
    """Render demos as a 3-column comparison table: CPU (QuadWild), CPU (ours), GPU.

    Demos should have "side" (cpu_orig/cpu_ours/gpu) and "step_name" fields.
    Each cell shows just the result image — no before/after (pipeline is sequential).
    Demos with the same step_name are grouped into one row.
    """
    SIDES = ("cpu_orig", "cpu_ours", "gpu", "gpu_fast")
    SIDE_LABELS = {"cpu_orig": "CPU (QuadWild)", "cpu_ours": "CPU (ours)", "gpu": "GPU", "gpu_fast": "GPU (fast)"}

    from collections import OrderedDict
    # Fixed row order so columns align even when pipelines skip steps
    ROW_ORDER = ["Input", "Raw Features", "Erode/Dilate", "Isotropic Remesh",
                 "Micro Collapse", "Re-detect Features", "Adaptive Remesh",
                 "Clean", "Refine", "Cross Field", "Trace", "Quad Output"]
    steps = OrderedDict()
    # Pre-populate in canonical order
    for s in ROW_ORDER:
        steps[s] = {}
    for d in demos:
        step = d.get("step_name", d.get("after_label", ""))
        if step not in steps:
            steps[step] = {}
        side = d.get("side", "gpu")
        steps[step][side] = d
    # Remove empty rows
    steps = OrderedDict((k, v) for k, v in steps.items() if v)

    rows = ""
    for step, sides in steps.items():
        step_html = html_escape(step)
        cells = []
        has_metrics = False
        metric_cells = []
        for side in SIDES:
            d = sides.get(side)
            if d is None:
                cells.append("<td>&mdash;</td>")
                metric_cells.append("<td></td>")
                continue
            timing = fmt_time(d["elapsed"]) if d["elapsed"] > 0 else ""
            timing_html = f'<span class="cell-label">{timing}</span>' if timing else ""
            img = f"{d['name']}.png"
            label = d.get("after_label", "")
            label_html = f'<span class="cell-label">{html_escape(label)}</span>' if label else ""
            cells.append(
                f'<td><img src="{img}" alt="{side} result">{timing_html}{label_html}</td>')
            m = d.get("metrics")
            if m:
                has_metrics = True
                metric_cells.append(
                    f'<td><span class="metrics">{_fmt_metrics(m)}</span></td>')
            else:
                metric_cells.append("<td></td>")

        all_cells = "\n        ".join(cells)
        rows += f"""
      <tr>
        <td>{step_html}</td>
        {all_cells}
      </tr>"""
        if has_metrics:
            all_mcells = "\n        ".join(metric_cells)
            rows += f"""
      <tr class="metrics-row">
        <td></td>
        {all_mcells}
      </tr>"""

    header_cells = "".join(f"<th>{SIDE_LABELS[s]}</th>" for s in SIDES)
    return f"""
    <table class="comparison-table">
      <thead><tr>
        <th>Step</th>
        {header_cells}
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


# ── Section generators ─────────────────────────────────────────────


def gen_analysis(bunny_v, bunny_f, bunny_mesh):
    """Generate analysis demos using the persistent Mesh for fast timings."""
    analysis_demos = []

    print("  Rendering: vertex_normals")
    analysis_demos.append(run_scalar("vertex_normals",
        lambda v, f: np.linalg.norm(bunny_mesh.vertex_normals(), axis=1),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            import pyrxmesh

            mesh = pyrxmesh.Mesh("bunnyhead.obj")
            normals = mesh.vertex_normals()"""),
        scalar_label="Normal Magnitude", cmap="coolwarm"))

    print("  Rendering: gaussian_curvature")
    analysis_demos.append(run_scalar("gaussian_curvature",
        lambda v, f: bunny_mesh.gaussian_curvature(),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            curvature = mesh.gaussian_curvature()"""),
        scalar_label="Gaussian Curvature", cmap="RdBu_r"))

    print("  Rendering: geodesic")
    analysis_demos.append(run_scalar("geodesic",
        lambda v, f: bunny_mesh.geodesic(np.array([0], dtype=np.int32)),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            import numpy as np

            seeds = np.array([0], dtype=np.int32)
            dist = mesh.geodesic(seeds)"""),
        scalar_label="Geodesic Distance", cmap="inferno"))

    return {
        "title": "Analysis",
        "subtitle": "Per-vertex geometry queries on GPU (persistent mesh — no reconstruction overhead)",
        "demos": analysis_demos,
    }


def gen_smoothing(bunny_v, bunny_f, bunny_mesh):
    """Generate smoothing demos."""
    smooth_demos = []

    print("  Rendering: smooth")
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
    print("  Rendering: mcf")
    smooth_demos.append(run_before_after("mcf",
        lambda v, f: pyrxmesh.mcf(v, f, time_step=1.0, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_mcf, f = pyrxmesh.mcf(
                v, f, time_step=1.0,
            )"""),
        after_label="Mean Curvature Flow"))

    return {
        "title": "Smoothing",
        "subtitle": "GPU-accelerated mesh smoothing and denoising",
        "demos": smooth_demos,
    }


def gen_parameterization(bunny_v, bunny_f):
    """Generate parameterization demos."""
    param_demos = []

    print("  Rendering: scp")
    param_demos.append(run_uv("scp",
        lambda v, f: pyrxmesh.scp(v, f, iterations=32, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            uv = pyrxmesh.scp(v, f, iterations=32)"""),
        label="SCP UV"))

    print("  Rendering: param")
    param_demos.append(run_uv("param",
        lambda v, f: pyrxmesh.param(v, f, newton_iterations=20, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            uv = pyrxmesh.param(
                v, f, newton_iterations=20,
            )"""),
        label="Dirichlet UV"))

    return {
        "title": "Parameterization",
        "subtitle": "UV unwrapping on GPU (requires mesh with boundaries)",
        "demos": param_demos,
    }


def gen_decimation(dragon_v, dragon_f):
    """Generate decimation demos."""
    decim_demos = []

    print("  Rendering: qslim")
    decim_demos.append(run_before_after("qslim",
        lambda v, f: pyrxmesh.qslim(v, f, target_ratio=0.25, verbose=True),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            v, f = pyrxmesh.load_obj("dragon.obj")
            v_dec, f_dec = pyrxmesh.qslim(
                v, f, target_ratio=0.25,
            )"""),
        after_label="QSlim 25%"))

    print("  Rendering: sec")
    decim_demos.append(run_before_after("sec",
        lambda v, f: pyrxmesh.sec(v, f, target_ratio=0.25, verbose=True),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            v_dec, f_dec = pyrxmesh.sec(
                v, f, target_ratio=0.25,
            )"""),
        after_label="SEC 25%"))

    return {
        "title": "Decimation",
        "subtitle": "GPU-accelerated mesh simplification (requires closed manifold)",
        "demos": decim_demos,
    }


def gen_remeshing(bunny_v, bunny_f, dragon_v, dragon_f):
    """Generate remeshing demos."""
    remesh_demos = []

    print("  Rendering: remesh")
    remesh_demos.append(run_before_after("remesh",
        lambda v, f: pyrxmesh.remesh(v, f, relative_len=1.0, iterations=2, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_re, f_re = pyrxmesh.remesh(
                v, f, relative_len=1.0,
                iterations=2,
            )"""),
        after_label="Isotropic Remeshed"))

    print("  Rendering: delaunay")
    remesh_demos.append(run_before_after("delaunay",
        lambda v, f: pyrxmesh.delaunay(v, f, verbose=True),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            v_del, f_del = pyrxmesh.delaunay(v, f)"""),
        after_label="Delaunay Flipped"))

    return {
        "title": "Remeshing",
        "subtitle": "GPU-accelerated mesh topology modification",
        "demos": remesh_demos,
    }


def gen_edge_ops(bunny_v, bunny_f):
    """Generate standalone edge operation demos."""
    edge_demos = []

    print("  Rendering: edge_split")
    edge_demos.append(run_before_after("edge_split",
        lambda v, f: pyrxmesh.edge_split(v, f, relative_len=0.5, iterations=1, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_s, f_s = pyrxmesh.edge_split(
                v, f, relative_len=0.5,
            )"""),
        after_label="Edge Split"))

    print("  Rendering: edge_collapse")
    edge_demos.append(run_before_after("edge_collapse",
        lambda v, f: pyrxmesh.edge_collapse(v, f, relative_len=2.0, iterations=1, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_c, f_c = pyrxmesh.edge_collapse(
                v, f, relative_len=2.0,
            )"""),
        after_label="Edge Collapse"))

    print("  Rendering: edge_flip")
    edge_demos.append(run_before_after("edge_flip",
        lambda v, f: pyrxmesh.edge_flip(v, f, iterations=1, verbose=True),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_f, f_f = pyrxmesh.edge_flip(v, f)"""),
        after_label="Edge Flip"))

    return {
        "title": "Standalone Edge Operations",
        "subtitle": "Individual sub-operations from isotropic remeshing (split, collapse, flip)",
        "demos": edge_demos,
    }


def gen_patches(bunny_v, bunny_f, dragon_v, dragon_f):
    """Generate patch decomposition demos."""
    patch_demos = []

    # Bunny patches
    print("  Rendering: patches_bunny")
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
    print("  Rendering: patches_dragon")
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

    return {
        "title": "Patch Decomposition",
        "subtitle": "RXMesh partitions the mesh into patches for GPU-parallel processing. Ribbon faces are shared between patches.",
        "demos": patch_demos,
    }


# ── Section registry and main ──────────────────────────────────────

SECTIONS = [
    ("analysis", gen_analysis, ["bunny_v", "bunny_f", "bunny_mesh"]),
    ("smoothing", gen_smoothing, ["bunny_v", "bunny_f", "bunny_mesh"]),
    ("parameterization", gen_parameterization, ["bunny_v", "bunny_f"]),
    ("decimation", gen_decimation, ["dragon_v", "dragon_f"]),
    ("remeshing", gen_remeshing, ["bunny_v", "bunny_f", "dragon_v", "dragon_f"]),
    ("edge_ops", gen_edge_ops, ["bunny_v", "bunny_f"]),
    ("patches", gen_patches, ["bunny_v", "bunny_f", "dragon_v", "dragon_f"]),
]


def main():
    tee = setup_logging()

    only = os.environ.get("PYRXMESH_DEMO_ONLY")
    stop_after = os.environ.get("PYRXMESH_DEMO_STOP_AFTER")

    if only:
        print(f"*** ONLY={only} ***")
    if stop_after:
        print(f"*** STOP_AFTER={stop_after} ***")

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    pyrxmesh.init()

    print("Loading meshes...")
    bunny_v, bunny_f = pyrxmesh.load_obj(os.path.join(RXMESH_INPUT, "bunnyhead.obj"))
    dragon_v, dragon_f = pyrxmesh.load_obj(os.path.join(RXMESH_INPUT, "dragon.obj"))
    print(f"  bunny: {len(bunny_v)} verts, {len(bunny_f)} faces")
    print(f"  dragon: {len(dragon_v)} verts, {len(dragon_f)} faces")

    # Only build persistent meshes if needed
    needs_persistent = only in (None, "analysis", "smoothing")
    bunny_mesh = dragon_mesh = None
    if needs_persistent:
        print("Building persistent meshes...")
        bunny_mesh = pyrxmesh.Mesh(bunny_v, bunny_f)
        dragon_mesh = pyrxmesh.Mesh(dragon_v, dragon_f)

    all_args = {
        "bunny_v": bunny_v, "bunny_f": bunny_f, "bunny_mesh": bunny_mesh,
        "dragon_v": dragon_v, "dragon_f": dragon_f, "dragon_mesh": dragon_mesh,
    }

    sections = []
    for name, func_or_path, arg_names in SECTIONS:
        func = func_or_path

        # Check if should run
        if only:
            if name != only:
                print(f"\n  [SKIPPED] {name}")
                continue

        print(f"\n=== Generating: {name} ===")
        kwargs = {k: all_args[k] for k in arg_names}

        result = func(**kwargs)
        if result:
            sections.append(result)

        if stop_after == name:
            print(f"\n*** Stopping after {name} ***")
            break

    print("\n=== Writing HTML ===")
    generate_html(sections)

    # Print summary of generated files
    import glob
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*")))
    print(f"\nDone! Generated {len(files)} files in {OUT_DIR}/")
    for f in files:
        size = os.path.getsize(f)
        print(f"  {os.path.basename(f)} ({size // 1024}KB)")


if __name__ == "__main__":
    import argparse

    section_names = [name for name, _, _ in SECTIONS]

    parser = argparse.ArgumentParser(
        description="Generate pyrxmesh demo site.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--only", choices=section_names,
        help="Run ONLY this section.")
    parser.add_argument("--stop-after", choices=section_names,
        help="Run sections in order, stop after this one.")

    args = parser.parse_args()

    if args.only:
        os.environ["PYRXMESH_DEMO_ONLY"] = args.only
    if args.stop_after:
        os.environ["PYRXMESH_DEMO_STOP_AFTER"] = args.stop_after

    main()
