"""
Generate visual demo of pyrxmesh for GitHub Pages.

Uses the persistent Mesh class so timings reflect actual GPU kernel speed
(not RXMesh construction overhead).
"""

import os
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


def run_before_after(name, func, verts_in, faces_in, code,
                     after_label="Output"):
    t0 = time.perf_counter()
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
    }


def run_scalar(name, func, verts, faces, code, scalar_label="Values",
               cmap="viridis"):
    t0 = time.perf_counter()
    scalars = func(verts, faces)
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
    }


def run_uv(name, func, verts, faces, code, label="UV"):
    t0 = time.perf_counter()
    uv = func(verts, faces)
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
    }


TEMPLATE_DIR = os.path.dirname(__file__)


def html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_demo(d):
    code_html = html_escape(d["code"])
    t = fmt_time(d["elapsed"])

    if d["type"] == "scalar":
        return f"""
    <section class="demo">
      <div class="demo-grid">
        <div class="demo-code">
          <pre><code>{code_html}</code></pre>
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
          <pre><code>{code_html}</code></pre>
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
        lambda v, f: pyrxmesh.mcf(v, f, time_step=1.0),
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
        lambda v, f: pyrxmesh.scp(v, f, iterations=32),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            uv = pyrxmesh.scp(v, f, iterations=32)"""),
        label="SCP UV"))

    param_demos.append(run_uv("param",
        lambda v, f: pyrxmesh.param(v, f, newton_iterations=20),
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
        lambda v, f: pyrxmesh.qslim(v, f, target_ratio=0.25),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            v, f = pyrxmesh.load_obj("dragon.obj")
            v_dec, f_dec = pyrxmesh.qslim(
                v, f, target_ratio=0.25,
            )"""),
        after_label="QSlim 25%"))

    decim_demos.append(run_before_after("sec",
        lambda v, f: pyrxmesh.sec(v, f, target_ratio=0.25),
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
        lambda v, f: pyrxmesh.remesh(v, f, relative_len=1.0, iterations=2),
        bunny_v, bunny_f,
        textwrap.dedent("""\
            v_re, f_re = pyrxmesh.remesh(
                v, f, relative_len=1.0,
                iterations=2,
            )"""),
        after_label="Isotropic Remeshed"))

    remesh_demos.append(run_before_after("delaunay",
        lambda v, f: pyrxmesh.delaunay(v, f),
        dragon_v, dragon_f,
        textwrap.dedent("""\
            v_del, f_del = pyrxmesh.delaunay(v, f)"""),
        after_label="Delaunay Flipped"))

    sections.append({
        "title": "Remeshing",
        "subtitle": "GPU-accelerated mesh topology modification",
        "demos": remesh_demos,
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
