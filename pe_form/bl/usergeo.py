"""The formation's mesh and Blender's own modelling tools, side by side.

The Form writes the formation's shell as mesh faces. You may model on it in
Edit Mode with Blender's tools (extrude, bevel, inset, knife, moving
vertices...), and what you made stays: the next change to the formation
replaces only the faces the Form wrote and still owns, never yours, and never
writes back a face you changed or removed.

Every face is one of three (face attribute "pe_role"):
  ENGINE  a face the Form wrote, untouched: it is what it was (its vertices at
          the points they were built from, "pe_base"), and an open one is a
          seat for a module, as ever;
  SEAT    a face you modelled that is still a seat: one of the Form's open
          faces EXTRUDED, i.e. the same face moved straight out along its
          normal (extruded again and again, still). A module placed on it
          stands there, and the seam between them is closed;
  USER    every other face you modelled: an extrusion's sides, a bevel's or
          an inset's faces, a face whose vertex you moved, and the children
          of any of them. Not a seat, like the skirts and bore rings.
Which open face a face came from rides on it ("pe_src", "pe_srcn"): Blender
copies a face's attributes to what a tool makes of it, so an extruded top
still knows its face. So does the module a face belongs to ("pe_mod", its
history index + 1, 0 for none or not known): an extrusion's sides, a bevel's
faces, belong to the module of the face they were made from, which is what a
skeleton's skin moves them with.
"""

import bmesh
import numpy as np

from ..engine import frame as FR
from ..engine.builder import OpenFace
from ..engine.meshplan import mesh_plan, to_bl, module_edges, join_tol, JOIN, THIN
from ..engine.vec import Vec3

ROLE_ATTR = "pe_role"
SRC_ATTR = "pe_src"     # base-space centre of the open face it came from
SRCN_ATTR = "pe_srcn"   # that face's corner count; 0 for none
MOD_ATTR = "pe_mod"     # the module it belongs to: history index + 1, 0 for none
ENGINE, USER, SEAT = 0, 1, 2


def to_engine(b):
    """meshplan.to_bl, inverted: Blender Z up back to the engine's Y up."""
    return Vec3(b[0], b[2], -b[1])


def _f32(p):
    return tuple(float(c) for c in np.asarray(p, dtype=np.float32))


def poly_key(pts):
    """A polygon by its points, as the mesh stores them (32-bit floats), in
    any order: the Form's faces are found again exactly."""
    return tuple(sorted(_f32(p) for p in pts))


def _round3(p, q=1e-4):
    return (round(p[0] / q), round(p[1] / q), round(p[2] / q))


# ── the plan: what the Form writes now ───────────────────────────────────────
def build_plan(st):
    """The engine's shell as mesh polygons (the modelled seats left out), each
    with its key and the open face it is ((centre, corners) or None)."""
    st.strip_free()
    try:
        e = st.engine
        mods = []
        base, polys, cols, opens, hexes = mesh_plan(e, mods)
        items = []
        for ids, col, oi, hx, m in zip(polys, cols, opens, hexes, mods):
            pts = [base[i] for i in ids]
            src = None
            if oi >= 0 and hx < 0:
                o = e.open_faces[oi]
                if not o.hex_sub and o.face_type != 6:
                    src = (to_bl(o.center3), len(o.verts3))
            items.append((pts, col, oi, hx, poly_key(pts), src, m))
        return items
    finally:
        st.inject_free()


def plan_index(items):
    return {it[4]: (it[2], it[3]) for it in items}


# ── polygons found again within the joining's reach ─────────────────────────
class Near:
    """Polygons by their points (plan keys: their corners), found from points
    that are each within reach of theirs: the corners the mesh plan joins
    (meshplan._join) move by up to JOIN of their module's edge, and never by
    THIN of the polygon's own shortest edge, so a polygon that is the same
    one is found and no other is. A file saved before the corners were
    joined keeps each module's own copy of a corner; so does a face kept
    when the copy the plan joins to changes."""

    def __init__(self, keys, e, tol):
        self.tol = tol
        self.far = JOIN * max(module_edges(e).values())
        self.cell = max(self.far * 4, tol * 4, 1e-6)
        self.grid = {}
        for k in keys:
            self.grid.setdefault(self._ck(_mean(k)), []).append(k)

    def _ck(self, p):
        c = self.cell
        return (int(p[0] // c), int(p[1] // c), int(p[2] // c))

    def find(self, pts, skip=()):
        c = self._ck(_mean(pts))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for k in self.grid.get((c[0] + dx, c[1] + dy, c[2] + dz), ()):
                        if k not in skip and len(k) == len(pts) and _same_poly(pts, k, _reach(k, self.tol, self.far)):
                            return k
        return None


def _reach(pts, tol, far):
    """How far polygon pts's corners may be from where the plan has them."""
    short = min(sum((pts[i][j] - pts[i - 1][j]) ** 2 for j in range(3)) for i in range(len(pts))) ** 0.5
    return tol + min(far, THIN * short) * 1.01


def _mean(pts):
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, sum(p[2] for p in pts) / n)


# ── layers ───────────────────────────────────────────────────────────────────
def layers(bm, names):
    """The bmesh layers the Form keeps, made when missing."""
    OPEN_ATTR, HEX_ATTR, WID_ATTR, BASE_ATTR, COLOR_ATTR = names
    fi = bm.faces.layers.int
    return {
        "open": fi.get(OPEN_ATTR) or fi.new(OPEN_ATTR),
        "hex": fi.get(HEX_ATTR) or fi.new(HEX_ATTR),
        "wid": fi.get(WID_ATTR) or fi.new(WID_ATTR),
        "role": fi.get(ROLE_ATTR) or fi.new(ROLE_ATTR),
        "srcn": fi.get(SRCN_ATTR) or fi.new(SRCN_ATTR),
        "mod": fi.get(MOD_ATTR) or fi.new(MOD_ATTR),
        "src": bm.faces.layers.float_vector.get(SRC_ATTR) or bm.faces.layers.float_vector.new(SRC_ATTR),
        "base": bm.verts.layers.float_vector.get(BASE_ATTR) or bm.verts.layers.float_vector.new(BASE_ATTR),
        "color": bm.loops.layers.color.get(COLOR_ATTR) or bm.loops.layers.color.new(COLOR_ATTR),
    }


# ── which face is which ──────────────────────────────────────────────────────
class Sorted:
    """What classify found."""

    def __init__(self):
        self.engine = []        # the Form's untouched faces
        self.user = []          # modelled, not seats
        self.seat = []          # (face, OpenFace)
        self.intact = set()     # keys of the plan's polygons still standing
        self.moved = set()      # vertices off their points


def _translated(src, pts, tol, near=None):
    """The seat a modelled face is when it is open face `src` moved straight
    out along its normal (an extrusion), else None. Its corners are matched
    within `near` (the mesh's corners may be joined copies of the face's)."""
    near = tol if near is None else near
    B = [to_bl(v) for v in src.verts3]
    if len(B) != len(pts):
        return None
    n = to_bl(src.normal3)
    ln = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5 or 1.0
    n = (n[0] / ln, n[1] / ln, n[2] / ln)
    cb = [sum(p[k] for p in B) / len(B) for k in range(3)]
    ct = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
    d = sum((ct[k] - cb[k]) * n[k] for k in range(3))
    if d <= tol:
        return None
    moved = [(b[0] + n[0] * d, b[1] + n[1] * d, b[2] + n[2] * d) for b in B]
    for p in pts:
        if min(sum((p[k] - q[k]) ** 2 for k in range(3)) for q in moved) > near * near:
            return None
    # The engine's face, carried out by d along its own normal, corner order kept.
    nv = src.normal3.clone().normalize()
    verts = [v.clone().add_scaled_vector(nv, d) for v in src.verts3]
    rec = {"v": [[v.x, v.y, v.z] for v in verts], "n": [src.normal3.x, src.normal3.y, src.normal3.z],
           "m": src.module_idx, "c": src.canonical_face_idx, "t": src.face_type, "fine": src.fine}
    return OpenFace.from_free(rec)


def classify(bm, L, M, st, plan, tol):
    """Sort the mesh's faces against `plan` (the plan the mesh was written
    from). Writes each face's role and seat number, and gives the vertices of
    modelled faces their own points (so the frame still reads off them)."""
    out = Sorted()
    e = st.engine
    lb = L["base"]
    for v in bm.verts:
        b = tuple(v[lb])
        p = FR.apply(M, b)
        c = v.co
        if (p[0] - c.x) ** 2 + (p[1] - c.y) ** 2 + (p[2] - c.z) ** 2 > tol * tol:
            out.moved.add(v)
    # The engine's own open faces, by where their centre is (the src lookup).
    st.strip_free()
    web = list(e.open_faces)
    st.inject_free()
    by_centre = {}
    for o in web:
        by_centre[_round3(to_bl(o.center3))] = o
    Mi = FR.inverse(M)
    seats = []
    edges = module_edges(e)
    rest = []
    for f in bm.faces:
        if not any(v in out.moved for v in f.verts):
            k = poly_key([tuple(v[lb]) for v in f.verts])
            hit = plan.get(k)
            if hit is not None:
                f[L["role"]] = ENGINE
                f[L["open"]], f[L["hex"]] = hit
                out.engine.append(f)
                out.intact.add(k)
                continue
        rest.append(f)
    # A face of the plan whose corners are other copies of the plan's (a file
    # saved before the plan joined them): the Form's still, and written
    # joined the next time.
    near = None
    for f in rest:
        if any(v in out.moved for v in f.verts):
            continue
        if near is None:
            near = Near([k for k in plan if k not in out.intact], e, tol)
        k = near.find([tuple(v[lb]) for v in f.verts], out.intact)
        if k is not None:
            f[L["role"]] = ENGINE
            f[L["open"]], f[L["hex"]] = plan[k]
            out.engine.append(f)
            out.intact.add(k)
    found = set(out.engine)
    for f in rest:
        if f in found:
            continue
        seat = None
        n = f[L["srcn"]]
        if n and len(f.verts) == n:
            src = by_centre.get(_round3(tuple(f[L["src"]])))
            if src is not None:
                seat = _translated(src, [FR.apply(Mi, tuple(v.co)) for v in f.verts], tol,
                                   tol + join_tol(e, src.module_idx, edges))
        if seat is not None:
            f[L["role"]] = SEAT
            f[L["hex"]] = -1
            out.seat.append((f, seat))
            seats.append(seat)
        else:
            f[L["role"]] = USER
            f[L["open"]] = -1
            f[L["hex"]] = -1
            out.user.append(f)
    st.set_free_seats(seats)
    base_n = len(e.open_faces) - len(seats)
    for k, (f, _) in enumerate(out.seat):
        f[L["open"]] = base_n + k
    # A modelled face's vertices take the points they are at now: the frame
    # (Set Origin) is read off the mesh, and they are part of it.
    eng = set()
    for f in out.engine:
        eng.update(f.verts)
    for v in bm.verts:
        if v not in eng and v in out.moved:
            v[lb] = FR.apply(Mi, tuple(v.co))
    return out


def taken_open(plan_items, intact):
    """The engine's open faces you modelled away (no longer seats)."""
    out = set()
    for pts, col, oi, hx, k, src, m in plan_items:
        if k not in intact:
            if oi >= 0:
                out.add(oi)
            if hx >= 0:
                out.add(hx)
    return out


# ── writing the mesh round what you modelled ─────────────────────────────────
def _same_poly(a, b, tol):
    if len(a) != len(b):
        return False
    return all(min(sum((p[k] - q[k]) ** 2 for k in range(3)) for q in b) <= tol * tol for p in a)


def write_around(bm, L, M, st, old_plan, items, wid, select, sealed, tol):
    """Replace the Form's faces with `items` (the new plan), leaving every
    modelled face as it is; a plan polygon the user changed or removed is not
    written again. A new face lying exactly on a modelled seat (a module
    placed on an extruded face) closes it: both go, and `sealed` remembers
    the seat so an undo of that module puts it back."""
    e = st.engine
    srt = classify(bm, L, M, st, old_plan, tol)
    taken = set(old_plan) - srt.intact
    if srt.engine:
        bmesh.ops.delete(bm, geom=srt.engine, context="FACES_ONLY")
    loose_e = [x for x in bm.edges if not x.link_faces]
    if loose_e:
        bmesh.ops.delete(bm, geom=loose_e, context="EDGES")
    loose_v = [v for v in bm.verts if not v.link_faces]
    if loose_v:
        bmesh.ops.delete(bm, geom=loose_v, context="VERTS")
    Mi = FR.inverse(M)
    lb = L["base"]
    edges = module_edges(e)
    far = JOIN * max(edges.values())
    # Vertices by their points, found within `tol` (a grid of cells and the
    # cells round them). A modelled vertex the Form's face lands on takes the
    # Form's exact point, so that face is the Form's again next time. One
    # further off, as far as the plan joins corners (`reach`: another copy of
    # the corner, kept by a modelled face), is taken too, by one point only.
    cell = max(tol * 4, far * 1.1 + tol, 1e-6)
    grid = {}
    claimed = {}

    def ck(p):
        return (int(p[0] // cell), int(p[1] // cell), int(p[2] // cell))

    for v in bm.verts:
        grid.setdefault(ck(tuple(v[lb])), []).append(v)

    def vert(p, reach=None):
        c = ck(p)
        r2 = tol * tol if reach is None else reach * reach
        best, bd = None, None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for v in grid.get((c[0] + dx, c[1] + dy, c[2] + dz), ()):
                        b = v[lb]
                        d = (b[0] - p[0]) ** 2 + (b[1] - p[1]) ** 2 + (b[2] - p[2]) ** 2
                        if (d <= tol * tol or d <= r2 and claimed.get(v, p) == p) and (bd is None or d <= bd):
                            best, bd = v, d
        if best is None:
            best = bm.verts.new(FR.apply(M, p))
            grid.setdefault(c, []).append(best)
        claimed.setdefault(best, p)
        best[lb] = p
        best.co = FR.apply(M, p)
        return best

    # A seat a module now stands on (a step placed on a modelled face) is
    # closed: the module's own face there is not written by the engine (it
    # is the face the module mated with), so the seat face goes, and is
    # remembered to come back if that module is undone.
    covered = []
    for i, h in enumerate(e.history):
        if h.free:
            covered.append((i + 1, [to_bl(Vec3(*v)) for v in h.free["v"]],
                            tol + join_tol(e, h.free.get("m", -1), edges)))
    for f, _ in srt.seat:
        pts = [FR.apply(Mi, tuple(v.co)) for v in f.verts]
        hit = next((c for c in covered if _same_poly(pts, c[1], c[2])), None)
        if hit is None:
            continue
        sealed.append({"pts": [list(p) for p in pts], "count": hit[0],
                       "src": list(f[L["src"]]), "srcn": f[L["srcn"]], "mod": f[L["mod"]],
                       "color": [list(f.loops[0][L["color"]])]})
        bmesh.ops.delete(bm, geom=[f], context="FACES_ONLY")
    active = None
    gone = Near(taken, e, tol) if taken else None
    for pts, col, oi, hx, k, src, m in items:
        if k in taken or (gone is not None and gone.find(pts) is not None):
            continue
        r = _reach(pts, tol, far)
        try:
            face = bm.faces.new([vert(p, r) for p in pts])
        except ValueError:
            continue
        face[L["role"]] = ENGINE
        face[L["mod"]] = m + 1
        face[L["open"]] = oi
        face[L["hex"]] = hx
        if src is not None:
            face[L["src"]] = src[0]
            face[L["srcn"]] = src[1]
        else:
            face[L["srcn"]] = 0
        for lp in face.loops:
            lp[L["color"]] = col
        if select is not None and (oi == select or hx == select):
            face.select_set(True)
            if oi == select or active is None:
                active = face
    # A seat a module was sealed onto, the module since undone: back.
    keep = []
    for rec in sealed:
        # Sealed when the formation had `count` modules: fewer now, and the
        # module on it was undone.
        if e.module_count >= rec["count"]:
            keep.append(rec)
            continue
        pts = [tuple(p) for p in rec["pts"]]
        if any(_same_poly(pts, [FR.apply(Mi, tuple(v.co)) for v in f.verts], tol) for f in bm.faces
               if len(f.verts) == len(pts)):
            continue
        try:
            face = bm.faces.new([vert(p) for p in pts])
        except ValueError:
            continue
        face[L["role"]] = USER
        face[L["open"]] = -1
        face[L["hex"]] = -1
        face[L["src"]] = rec["src"]
        face[L["srcn"]] = rec["srcn"]
        face[L["mod"]] = rec.get("mod", 0)
        if rec.get("color"):
            for lp in face.loops:
                lp[L["color"]] = rec["color"][0]
    sealed[:] = keep
    for f in bm.faces:
        f[L["wid"]] = wid
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    if select is not None:
        bm.faces.active = active
    bm.select_flush_mode()
    bm.normal_update()
    return srt
