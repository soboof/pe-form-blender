"""What the Form draws over the formation in Edit Mode (Viewport.tsx):

  ghost        the module Place would put down on each picked face, in its
               colour at 0.10 (red at 0.12 where it would collide), outlined;
               in carve mode the module it would cut out, seated inside the
               formation and drawn through it, carve blue (red if refused)
  open faces   every face a module can go on, tinted by its corner count
               (FACE_TYPE_COLOR), outlined (Settings > View)

Blender draws the selection itself, so the picked face is its selected face.

The web blends its tints in sRGB (0.07 fill, 0.2 outline); Blender blends in
linear light, where the same alpha reads two or three times as strong over the
dark side of a module (measured on a screenshot: 0.07 moved a dark face as far
as 0.18 would in sRGB). So the alphas here are lower, to look as the web does.

Faces lie on the shell, so the tints are lifted along their normals to sit in
front of it: by a few steps of the depth buffer at the view's distance, which a
fixed share of an edge was not (it z-fought into stripes on faces seen
obliquely from a few edges away), and never less than a thousandth of an edge.
"""

import bpy
import gpu
from mathutils import Matrix
from gpu_extras.batch import batch_for_shader

from ..engine.geom import get_edge
from .formation import edit_formation, follow_selection

FACE_TYPE_COLOR = {3: "#ffd700", 4: "#00bcd4", 5: "#4caf50", 6: "#9c27b0", 7: "#26c6da",
                   8: "#ff9800", 9: "#ff7043", 10: "#f44336"}
FILL_ALPHA = 0.035
LINE_ALPHA = 0.09
CARVE_GHOST = "#4ca8d8"  # Viewport.tsx CARVE_GHOST
MAX_GHOSTS = 32

_handle = None


def _lift(ctx, obj, edge):
    """How far to lift a tint off the shell so the depth test keeps it in
    front: a 24-bit depth buffer resolves about d^2 / (near * 2^24) at
    distance d in perspective."""
    lo, hi = edge * 1e-3, edge * 0.03
    rv3d, sp = ctx.region_data, ctx.space_data
    if rv3d is None or sp is None or not rv3d.is_perspective:
        return lo
    eye = rv3d.view_matrix.inverted().translation
    r = max(obj.dimensions) * 0.5 if obj.dimensions.length else edge
    d = (obj.matrix_world.translation - eye).length + r
    near = max(getattr(sp, "clip_start", 0.01), 1e-4)
    return min(hi, max(lo, 6.0 * d * d / (near * 2 ** 24)))


def hexc(h, a=1.0):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


def _bl(x, y, z):
    return (x, -z, y)


def _tris(pos, col, depth):
    if not pos:
        return
    sh = gpu.shader.from_builtin("SMOOTH_COLOR")
    gpu.state.depth_test_set("LESS_EQUAL" if depth else "NONE")
    batch_for_shader(sh, "TRIS", {"pos": pos, "color": col}).draw(sh)


def _lines(pos, col, vp, depth=True):
    if not pos:
        return
    sh = gpu.shader.from_builtin("POLYLINE_SMOOTH_COLOR")
    sh.uniform_float("viewportSize", vp)
    sh.uniform_float("lineWidth", 1.0)
    gpu.state.depth_test_set("LESS_EQUAL" if depth else "NONE")
    batch_for_shader(sh, "LINES", {"pos": pos, "color": col}).draw(sh)


def _ghosts(st):
    """(module data, world vertices, refused) of the module on every picked
    face: what Place (or Carve) would put there. The active face first; at
    most MAX_GHOSTS, since each is a full seat test."""
    e = st.engine
    out = []
    for i in st.targets()[:MAX_GHOSTS]:
        if st.carve_mode:
            p = e.preview_carve(st.active_mod_id, i, st.hexa_sub, st.orientation, st.conn_sel)
            if p is not None:
                out.append((p.md, p.w_verts, not p.ok))
        else:
            p = e.preview_placement(st.active_mod_id, i, st.hexa_sub, st.orientation, st.conn_sel)
            if p is not None:
                out.append((p.md, p.w_verts, p.intersects))
    return out


def draw():
    ctx = bpy.context
    obj = edit_formation(ctx)
    if obj is None:
        return
    try:
        f = follow_selection(obj)
    except (ReferenceError, ValueError):
        return
    st = f.state
    e = st.engine
    region = ctx.region
    vp = (region.width, region.height)
    lift = _lift(ctx, obj, get_edge(e.X))
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_mask_set(False)
    gpu.matrix.push()
    # The formation's own frame under the object's: the origin is its plate.
    gpu.matrix.multiply_matrix(obj.matrix_world @ Matrix(f.frame))
    try:
        if ctx.window_manager.pe_show_open:
            fills, fcol, lines, lcol = [], [], [], []
            for i, o in enumerate(e.open_faces):
                # Not a face you modelled away (it is no longer a seat).
                if i == st.sel_idx or o.hex_sub or i in f.taken_open:
                    continue
                n = o.normal3
                vs = [_bl(v.x + n.x * lift, v.y + n.y * lift, v.z + n.z * lift) for v in o.verts3]
                base = FACE_TYPE_COLOR.get(o.face_type, "#888866")
                cf, cl = hexc(base, FILL_ALPHA), hexc(base, LINE_ALPHA)
                for t in range(1, len(vs) - 1):
                    fills.extend((vs[0], vs[t], vs[t + 1]))
                    fcol.extend((cf, cf, cf))
                for k in range(len(vs)):
                    lines.extend((vs[k], vs[(k + 1) % len(vs)]))
                    lcol.extend((cl, cl))
            _tris(fills, fcol, True)
            _lines(lines, lcol, vp)
        if st.sel_idx is not None:
            for md, w_verts, bad in f.memo("ghosts", lambda: _ghosts(st)):
                w = [_bl(v.x, v.y, v.z) for v in w_verts]
                if st.carve_mode:
                    # Seated inside the formation, so drawn through it.
                    base = "#ff3333" if bad else CARVE_GHOST
                    cf, cl, depth = hexc(base, 0.16), hexc(base, 0.95), False
                else:
                    base = "#ff3333" if bad else e.color_of(st.active_mod_id)
                    cf = hexc(base, 0.12 if bad else 0.10)
                    cl = hexc(base, 0.9 if bad else 0.75)
                    depth = True
                gf = []
                for fc in md.faces:
                    gf.extend((w[fc[0]], w[fc[1]], w[fc[2]]))
                _tris(gf, [cf] * len(gf), depth)
                seen, gl = set(), []
                for poly in md.poly_faces:
                    for k in range(len(poly)):
                        a, b = poly[k], poly[(k + 1) % len(poly)]
                        key = (a, b) if a < b else (b, a)
                        if key not in seen:
                            seen.add(key)
                            gl.extend((w[a], w[b]))
                _lines(gl, [cl] * len(gl), vp, depth)
    except Exception as ex:  # never take Blender's draw loop down
        print("PE Form overlay:", repr(ex))
    finally:
        gpu.matrix.pop()
        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("NONE")


def register():
    global _handle
    if _handle is None:
        _handle = bpy.types.SpaceView3D.draw_handler_add(draw, (), "WINDOW", "POST_VIEW")


def unregister():
    global _handle
    if _handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
        _handle = None
