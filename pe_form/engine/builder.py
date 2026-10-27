"""Port of pe-web/src/engine/builder.ts: the Form's BuilderEngine.

The placement path for the built-in modules, line for line: snapping, the
SAT seat test, the adapter skirt, wall cancellation, the interior purge, the
sealed-pocket fill, hexagon sub-seats, the rhombic shape match, the turn
options and connection-face dials, undo, and the save format. The order of
every list and every floating-point operation follows the web engine, because
a saved step names its face by its position in `open_faces`: a port that put
one face in another place would replay a web save onto the wrong faces.
tests/test_engine.py holds it to the web engine's own answers.

Carve (subtractive placement, engine/carve.py) is ported too.

Not ported (yet): custom Factory modules (connectors, bores, pin joints),
face division, deleting a module from the middle. A web save that uses them
opens up to the first such step and says why it stopped.
"""

import math

from .vec import Vec3, Quat, Box3, box_from_points
from .geom import (cen3, nor3, face_key, get_edge, get_scale, cen_n, nor_n, build_tri_poly_map,
                   regular_polygon, is_rhomb_face, short_diagonal_first, is_regular_poly,
                   same_mouth_shape)
from .modules import MOD_DATA, MODULES, MODULE_BY_ID
from .snap import snap_transform, pick_conn_face
from .collision import check_intersection, ModuleWorldData
from .carve import (shell_crosses, coplanar_cover, cut_polygon, ring_around, face_on_polygon, valence_damage,
                    poly_normal, split_at_points)
from .jsnum import to_fixed
from .divide import subdivide_polygon, is_parallelogram, split_along_segments, DIVIDE_MAX, DIVIDE_MIN_FINE

DIVIDE_STEP = "__divide__"
# A module placed on a face modelled in Blender (an extruded face): not a face
# the web Form can have, so the step carries the face itself, under a module
# id the web refuses ("unknown module") rather than replays onto another face.
FREE_STEP = "__free__"

# Per-face connection dial for low-symmetry built-ins: one representative per
# proper-rotation orbit, by module id then by target face corner count.
ORBIT_DIAL = {
    "tridiminishedrhombicosidodecahedron": {3: [27, 28, 31], 4: [12, 15, 16, 21, 24], 5: [3, 6, 9]},
    "snubsquareantiprism": {3: [2, 10, 11]},
    "sphenocorona": {3: [2, 3, 4, 5, 6, 12]},
    "sphenomegacorona": {3: [2, 3, 4, 5, 7, 8, 9, 16]},
    "hebesphenomegacorona": {3: [3, 4, 5, 6, 8, 9, 10, 17, 19], 4: [0, 2]},
    "disphenocingulum": {3: [4, 5, 7, 9, 11]},
    "bilunabirotunda": {3: [6, 7]},
    "triangularhebesphenorotunda": {3: [7, 8, 13, 16, 19]},
    "bilinskidodecahedron": {4: [0, 2, 10]},
    "rhombicicosahedron": {4: [0, 2]},
}
# Modules that offer every face of the target's size.
ALL_FACE_DIAL = {"snubdisphenoid", "augmentedsphenocorona"}


def js_round(x):
    """Math.round: halves go up."""
    return math.floor(x + 0.5)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class AllFace:
    """One external triangle of the closed mesh, in world space."""
    __slots__ = ("va", "vb", "vc", "key", "color", "mod_idx", "face_idx", "conn_type", "conn_tag", "is_skirt",
                 "poly_idx", "sub_k", "split")

    def __init__(self, va, vb, vc, key, color, conn_type=None, conn_tag=None, face_idx=None, is_skirt=False):
        self.va = va
        self.vb = vb
        self.vc = vc
        self.key = key
        self.color = color
        self.mod_idx = None
        self.face_idx = face_idx
        self.conn_type = conn_type
        self.conn_tag = conn_tag
        self.is_skirt = is_skirt
        # Not in the web engine: which polygon of its module this triangle
        # is part of, and for a hexagon's centre triangles which of the six.
        # The Blender side uses them to build the mesh polygon by polygon and
        # to tell which open face a selected mesh face is. Nothing in the
        # engine reads them.
        self.poly_idx = None
        self.sub_k = None
        # Re-cut by a carve (a piece of a face, or a ring round a pocket's
        # mouth): it no longer matches its face by centroid.
        self.split = False

    def copy_with(self, va, vb, vc, key, split=None):
        """{...f, va, vb, vc, key}: the same face's tags on new corners."""
        g = AllFace(va, vb, vc, key, self.color, self.conn_type, self.conn_tag, self.face_idx, self.is_skirt)
        g.mod_idx, g.poly_idx, g.sub_k, g.split = self.mod_idx, self.poly_idx, self.sub_k, self.split
        if split is not None:
            g.split = split
        return g


class _Target:
    """A face to snap against: what snap_transform reads."""
    __slots__ = ("verts3", "center3", "normal3")

    def __init__(self, verts3, center3, normal3):
        self.verts3, self.center3, self.normal3 = verts3, center3, normal3


class OpenFace:
    """A face a module can be placed on."""
    __slots__ = ("verts3", "center3", "normal3", "face_type", "module_idx", "conn_type", "conn_tag",
                 "canonical_face_idx", "hex_sub", "fine", "sub_k", "free")

    def __init__(self, verts3, center3, normal3, face_type, module_idx, canonical_face_idx,
                 conn_type="open", conn_tag=None, hex_sub=False):
        self.verts3 = verts3
        self.center3 = center3
        self.normal3 = normal3
        self.face_type = face_type
        self.module_idx = module_idx
        self.conn_type = conn_type
        self.conn_tag = conn_tag
        self.canonical_face_idx = canonical_face_idx
        self.hex_sub = hex_sub
        self.fine = None
        self.sub_k = None  # a hexagon's sub-slot: which of its six (Blender side only)
        # A seat modelled in Blender (an extruded face), not one the engine
        # made: its record ({"v", "n", "m", "c", "t", "fine"}), else None.
        self.free = None

    @staticmethod
    def from_free(d):
        """The open face a FREE_STEP's record describes."""
        verts = [Vec3(*v) for v in d["v"]]
        f = OpenFace(verts, cen_n(verts), Vec3(*d["n"]), d.get("t", len(verts)), d.get("m", -1), d.get("c"))
        f.fine = d.get("fine")
        f.free = dict(d)
        return f


class DivideEntry:
    """A face division: made when `at` modules were placed, the face it
    divided `n` ways, and the lists as they stood before and after it."""
    __slots__ = ("at", "n", "used_idx", "used_face", "open_before", "all_before", "all_after", "step_x")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class HistoryEntry:
    __slots__ = ("new_of_count", "used_idx", "used_face", "mod_centroid", "snapshot_faces", "mod_id",
                 "hexa_sub", "orientation", "color", "conn_sel", "cfi", "Q", "tr", "scale",
                 "parent_mod_idx", "parent_face_idx", "step_x", "removed_open_faces", "open_before", "tex", "carve",
                 "free")

    def __init__(self, **kw):
        self.removed_open_faces = None
        self.tex = None
        self.carve = False
        self.free = None
        for k, v in kw.items():
            setattr(self, k, v)


class PlaceResult:
    __slots__ = ("ok", "reason")

    def __init__(self, ok, reason=None):
        self.ok = ok
        self.reason = reason

    def __repr__(self):
        return "PlaceResult(%r, %r)" % (self.ok, self.reason)


class CarvePreview:
    """previewCarve: the ghost's geometry and the verdict carve() would give."""
    __slots__ = ("md", "sv", "Q", "tr", "cfi", "ok", "reason", "w_verts")

    def __init__(self, md, sv, Q, tr, cfi, ok, reason, w_verts):
        self.md, self.sv, self.Q, self.tr, self.cfi = md, sv, Q, tr, cfi
        self.ok, self.reason, self.w_verts = ok, reason, w_verts


class Preview:
    __slots__ = ("md", "sv", "Q", "tr", "cfi", "intersects", "w_verts")

    def __init__(self, md, sv, Q, tr, cfi, intersects, w_verts):
        self.md = md
        self.sv = sv
        self.Q = Q
        self.tr = tr
        self.cfi = cfi
        self.intersects = intersects
        self.w_verts = w_verts


class _Hull:
    __slots__ = ("verts", "faces", "box", "cen", "void", "tol", "mwd")


class _Hulls:
    """The hull list plus its uniform grid (BuilderEngine._hullGrid)."""

    def __init__(self, hulls):
        self.hulls = hulls
        self._q = None

    def enclosed(self, p):
        if self._q is None:
            self._q = _build_grid(self.hulls)
        return self._q(p)


def _hull_planes(h):
    pl = h.mwd.planes if h.mwd is not None else None
    if pl is not None:
        return pl
    ns, os_ = [], []
    for fc in h.faces:
        fa = h.verts[fc[0]]
        nn = nor3(fa, h.verts[fc[1]], h.verts[fc[2]])
        if nn.dot(Vec3().sub_vectors(h.cen, fa)) > 0:
            nn.negate()
        ns.append((nn.x, nn.y, nn.z))
        os_.append((fa.x, fa.y, fa.z))
    pl = (ns, os_)
    if h.mwd is not None:
        h.mwd.planes = pl
    return pl


def _build_grid(hulls):
    planes = []
    for h in hulls:
        ns, os_ = _hull_planes(h)
        b = h.box
        planes.append(((b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z), ns, os_, h.void, h.tol))
    any_void = any(p[3] for p in planes)
    # Any cell size gives the same verdicts (a hull is filed in every cell its
    # box touches, so a point meets every hull whose box holds it); the web
    # uses the largest hull's size, a mean size keeps fewer hulls to a cell.
    size = 0.0
    for h in hulls:
        b = h.box
        size += max(b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z)
    size = size / len(hulls) if hulls else 0.0
    cs = size if size > 0 else 1

    def cell(v):
        return math.floor(v / cs)

    grid = {}
    for pl in planes:
        bx = pl[0]
        for x in range(cell(bx[0]), cell(bx[3]) + 1):
            for y in range(cell(bx[1]), cell(bx[4]) + 1):
                for z in range(cell(bx[2]), cell(bx[5]) + 1):
                    grid.setdefault((x, y, z), []).append(pl)

    def inside(pl, px, py, pz, lim):
        ns, os_ = pl[1], pl[2]
        for i in range(len(ns)):
            n = ns[i]
            o = os_[i]
            if n[0] * (px - o[0]) + n[1] * (py - o[1]) + n[2] * (pz - o[2]) > lim:
                return False
        return True

    def contains(bx, px, py, pz):
        return not (px < bx[0] or px > bx[3] or py < bx[1] or py > bx[4] or pz < bx[2] or pz > bx[5])

    def q(p):
        px, py, pz = p.x, p.y, p.z
        cands = grid.get((cell(px), cell(py), cell(pz)))
        if not cands:
            return False
        if not any_void:
            for pl in cands:
                if not contains(pl[0], px, py, pz):
                    continue
                if inside(pl, px, py, pz, 1e-9):
                    return True
            return False
        for pl in reversed(cands):
            if not contains(pl[0], px, py, pz):
                continue
            if pl[3]:
                if inside(pl, px, py, pz, -pl[4]):
                    return False
                continue
            if inside(pl, px, py, pz, 1e-9):
                return True
        return False

    return q


class BuilderEngine:
    def __init__(self, X=5, color="#c44444"):
        self.X = X
        self.mat_color = color
        self.all_faces = []
        self.open_faces = []
        self.history = []
        self.divides = []
        self.module_world_data = []
        self.seed_shape = 3
        # What a web save carries that the add-on keeps but does not use
        # (ground, texTile, uv, rig): written back as it came, so opening a
        # web formation here and saving it again loses none of it.
        self.extras = {}
        self.init()

    # ── Lookups ──────────────────────────────────────────────────────────────
    @property
    def module_count(self):
        return len(self.history)

    @property
    def can_undo(self):
        return len(self.history) + len(self.divides) > 0

    @property
    def has_steps(self):
        return len(self.history) > 0 or len(self.divides) > 0

    def mod_data(self, mid):
        return MOD_DATA.get(mid)

    def color_of(self, mid):
        m = MODULE_BY_ID.get(mid)
        return m.color if m else self.mat_color

    def _face(self, sel_idx):
        if sel_idx is None or not isinstance(sel_idx, int) or isinstance(sel_idx, bool):
            return None
        if 0 <= sel_idx < len(self.open_faces):
            return self.open_faces[sel_idx]
        return None

    # ── Which face of the module connects ────────────────────────────────────
    def pick_conn_face_for(self, mod_id, hexa_sub, tf, conn_sel):
        if tf.conn_type and tf.conn_type != "open":
            return -1
        if tf.hex_sub:
            hex_result = pick_conn_face(mod_id, hexa_sub, 6)
            if hex_result >= 0:
                parent = any(f.face_type == 6 and f.module_idx == tf.module_idx
                             and f.canonical_face_idx == tf.canonical_face_idx for f in self.open_faces)
                if parent:
                    return hex_result
        N = tf.face_type if tf.face_type is not None else 3
        md = self.mod_data(mod_id)
        cands = self._dial_candidates(mod_id, md, N) if md else None
        picked = -1
        if cands and len(cands) > 1:
            sel = (conn_sel or {}).get("face:%d" % N)
            if _is_num(sel):
                picked = cands[int(max(0, min(len(cands) - 1, sel)))]
        if picked < 0:
            picked = pick_conn_face(mod_id, hexa_sub, N)
        if picked < 0 or md is None:
            return picked

        def matches(pfi):
            mouth = [md.verts[i] for i in md.poly_faces[pfi]]
            if len(mouth) != len(tf.verts3):
                return False
            if is_regular_poly(tf.verts3) and is_regular_poly(mouth):
                return True
            return same_mouth_shape(tf.verts3, mouth)

        if tf.module_idx == -1 or matches(picked):
            return picked
        best, best_y = -1, math.inf
        for pfi, poly in enumerate(md.poly_faces):
            if not matches(pfi):
                continue
            y = nor_n([md.verts[i] for i in poly]).y
            if y < best_y:
                best_y, best = y, pfi
        return best

    def _resolve_target(self, tf, cfi, md):
        if tf.hex_sub and len(md.poly_faces[cfi]) == 6:
            for f in self.open_faces:
                if f.face_type == 6 and f.module_idx == tf.module_idx and f.canonical_face_idx == tf.canonical_face_idx:
                    return f
        return tf

    def can_connect_to(self, mod_id, sel_idx, hexa_sub="cap", conn_sel=None):
        if sel_idx is None:
            return True
        tf = self._face(sel_idx)
        if tf is None or self.mod_data(mod_id) is None:
            return False
        return self.pick_conn_face_for(mod_id, hexa_sub, tf, conn_sel or {}) >= 0

    def _dial_candidates(self, mod_id, md, N):
        reps = ORBIT_DIAL.get(mod_id, {}).get(N)
        if reps:
            return reps if len(reps) > 1 else None
        if mod_id in ALL_FACE_DIAL:
            arr = [i for i, p in enumerate(md.poly_faces) if len(p) == N]
            return arr if len(arr) > 1 else None
        return None

    def builtin_dial_info(self, mod_id, sel_idx):
        md = self.mod_data(mod_id)
        if md is None:
            return None
        tf = self._face(sel_idx)
        if tf is None or (tf.conn_type and tf.conn_type != "open"):
            return None
        N = tf.face_type if tf.face_type is not None else 3
        c = self._dial_candidates(mod_id, md, N)
        if not c or len(c) <= 1:
            return None
        return {"key": "face:%d" % N, "count": len(c)}

    def preview_placement(self, mod_id, sel_idx, hexa_sub="cap", orientation=0, conn_sel=None):
        tf = self._face(sel_idx)
        if tf is None:
            return None
        md = self.mod_data(mod_id)
        if md is None:
            return None
        cfi = self.pick_conn_face_for(mod_id, hexa_sub, tf, conn_sel or {})
        if cfi < 0:
            return None
        sc = get_scale(md, self.X)
        eff = self._resolve_target(tf, cfi, md)
        s = snap_transform(md, eff, orientation, sc, cfi)
        w = [v.clone().apply_quaternion(s.Q).add(s.tr) for v in s.sv]
        intersects = self._seat_problem(sel_idx, w, md, cfi) is not None
        return Preview(md, s.sv, s.Q, s.tr, cfi, intersects, w)

    # ── Turns ────────────────────────────────────────────────────────────────
    def turn_steps(self, face_idx):
        f = self._face(face_idx)
        if f is None:
            return 3
        vs = f.verts3
        n = len(vs)
        if n < 3:
            return 1
        c = cen_n(vs)
        nrm = f.normal3.clone().normalize()
        tol = vs[0].distance_to(vs[1]) * 1e-3
        order = 1
        for k in range(1, n):
            q = Quat().set_from_axis_angle(nrm, (k * 2 * math.pi) / n)
            onto = True
            for v in vs:
                p = v.clone().sub(c).apply_quaternion(q).add(c)
                if not any(w.distance_to(p) < tol for w in vs):
                    onto = False
                    break
            if onto:
                order += 1
        return order

    def turn_options(self, mod_id, sel_idx, hexa_sub="cap", conn_sel=None):
        n = self.turn_steps(sel_idx)
        # Whole degrees, as the web's Math.round: on a heptagon a turn sits up
        # to half a degree off a perfect fit. Saves hold these, so they stay;
        # the mesh closes the seams it leaves (meshplan.true_turns).
        all_ = [js_round((k * 360) / n) for k in range(n)]
        if sel_idx is None:
            return all_
        tf = self._face(sel_idx)
        md = self.mod_data(mod_id)
        if tf is None or md is None:
            return all_
        cfi = self.pick_conn_face_for(mod_id, hexa_sub if hexa_sub is not None else "cap", tf, conn_sel or {})
        if cfi < 0:
            return all_
        sc = get_scale(md, self.X)
        eff = self._resolve_target(tf, cfi, md)

        def seat(deg):
            s = snap_transform(md, eff, deg, sc, cfi)
            return [v.clone().apply_quaternion(s.Q).add(s.tr) for v in s.sv]

        tol = get_edge(self.X) * 1e-3

        def same(a, b):
            return len(a) == len(b) and all(any(q.distance_to(p) < tol for q in b) for p in a)

        kept = []
        for deg in all_:
            pts = seat(deg)
            if not any(same(k[1], pts) for k in kept):
                kept.append((deg, pts))
        return [k[0] for k in kept]

    def turn_matters(self, mod_id, sel_idx, hexa_sub="cap", conn_sel=None):
        return len(self.turn_options(mod_id, sel_idx, hexa_sub, conn_sel)) > 1

    # ── The seed ─────────────────────────────────────────────────────────────
    def init(self, shape=None):
        if shape is not None:
            self.seed_shape = shape
        self.all_faces = []
        self.open_faces = []
        self.history = []
        self.divides = []
        self.module_world_data = []
        self._purged_at = None
        verts = self._start_seed(self.seed_shape)
        self.all_faces = self._seed_tris(verts)
        self.open_faces = [OpenFace(verts, cen_n(verts), Vec3(0, 1, 0), len(verts), -1, None)]

    def _start_seed(self, n):
        return regular_polygon(max(3, js_round(n)), get_edge(self.X))

    def _seed_tris(self, verts):
        out = []
        for i in range(1, len(verts) - 1):
            a, b, c = verts[0], verts[i], verts[i + 1]
            out.append(AllFace(a.clone(), b.clone(), c.clone(), face_key(a, b, c), self.mat_color))
        return out

    # ── Placement ────────────────────────────────────────────────────────────
    def place(self, mod_id, sel_idx, hexa_sub=None, orientation=None, color=None, conn_sel=None):
        tf = self._face(sel_idx)
        if tf is None:
            return PlaceResult(False, "no such open face")
        md = self.mod_data(mod_id)
        if md is None:
            return PlaceResult(False, "unknown module " + str(mod_id))
        hexa_sub = "cap" if hexa_sub is None else hexa_sub
        orientation = 0 if orientation is None else orientation
        face_color = self.mat_color if color is None else color
        conn_sel = {} if conn_sel is None else conn_sel

        cfi = self.pick_conn_face_for(mod_id, hexa_sub, tf, conn_sel)
        if cfi < 0:
            return PlaceResult(False, "no connectable face")
        hex_sub_to_hex = tf.hex_sub is True and len(md.poly_faces[cfi]) == 6
        eff = self._resolve_target(tf, cfi, md)
        sc = get_scale(md, self.X)
        s = snap_transform(md, eff, orientation, sc, cfi)
        Q, tr, sv = s.Q, s.tr, s.sv

        def to_world(i):
            return sv[i].clone().apply_quaternion(Q).add(tr)

        w_verts = [to_world(i) for i in range(len(sv))]
        blocked = self._seat_problem(sel_idx, w_verts, md, cfi)
        if blocked:
            return PlaceResult(False, blocked)

        ctx = {
            "mod_id": mod_id, "md": md, "cfi": cfi, "sc": sc, "Q": Q, "tr": tr, "to_world": to_world,
            "w_verts": w_verts, "tf": eff, "fine": tf.fine if tf.fine is not None else 1, "sel_idx": sel_idx, "hexa_sub": hexa_sub,
            "orientation": orientation, "face_color": face_color, "conn_sel": conn_sel,
        }
        self._place_merge(ctx)

        if hex_sub_to_hex:
            hex_pfi = tf.canonical_face_idx
            hex_mod = tf.module_idx
            removed = []
            kept = []
            for f in self.open_faces:
                if f.module_idx == hex_mod and f.canonical_face_idx == hex_pfi and (f.face_type == 6 or f.hex_sub):
                    removed.append(f)
                else:
                    kept.append(f)
            self.open_faces = kept
            if removed and self.history:
                self.history[-1].removed_open_faces = removed
        return PlaceResult(True)

    def _seat_problem(self, sel_idx, w_verts, md, cfi):
        """Why the module cannot sit there ("intersection", "partial
        contact"), or None. The tuned hull test, unless a carve is involved:
        a seat on a carved wall, or a module overlapping a carved hull, is
        judged by the shell instead (the hull test sees the hull the cavity
        was cut from and would refuse every such seat)."""
        if not self.is_carved_wall(sel_idx):
            if not check_intersection(w_verts, md.faces, self.module_world_data):
                return None
            voids = [m for i, m in enumerate(self.module_world_data)
                     if i < len(self.history) and self.history[i].carve]
            if not voids or not check_intersection(w_verts, md.faces, voids):
                return "intersection"
        if shell_crosses(w_verts, md.faces, self.all_faces):
            return "intersection"
        # With no shell through it the module is all air or all material: any
        # one sample reading air settles it (a sample on a cavity's boundary
        # reads material, so the error only ever goes that way).
        in_material = self._hull_list().enclosed
        c = cen_n(w_verts)
        if all(in_material(p) for p in [c] + [c.clone().lerp(v, 0.5) for v in w_verts]):
            return "intersection"
        E = get_edge(self.X)
        for pfi, poly in enumerate(md.poly_faces):
            if pfi == cfi:
                continue
            if coplanar_cover([w_verts[i] for i in poly], self.all_faces, E)[1] == "mixed":
                return "partial contact"
        return None

    # ── Carve: a module cut out of the formation ─────────────────────────────
    # The subtractive counterpart of placement (pe-web builder.ts, "Carving").
    # The module is seated on the INNER side of the picked face and its volume
    # taken out, allowed only when it lies wholly inside the material. Its
    # history step is flagged `carve` and its hull is a VOID: a point is
    # material when the latest step whose hull holds it placed a module, air
    # when that step was a carve. A cavity's walls are ordinary open faces.
    @property
    def carve_count(self):
        return sum(1 for h in self.history if h.carve)

    def is_carved_wall(self, sel_idx):
        f = self._face(sel_idx)
        return f is not None and f.module_idx >= 0 and f.module_idx < len(self.history) \
            and self.history[f.module_idx].carve

    def can_carve_with(self, mod_id, sel_idx, hexa_sub="cap", conn_sel=None):
        """Can `mod_id` be carved through this face at all (the picker's
        question in carve mode)? Says nothing about fitting in the material."""
        if self.mod_data(mod_id) is None:
            return False
        if sel_idx is None:
            return True
        tf = self._face(sel_idx)
        if tf is None or tf.module_idx < 0 or tf.free or (tf.conn_type and tf.conn_type != "open"):
            return False
        return self.pick_conn_face_for(mod_id, hexa_sub, tf, conn_sel or {}) >= 0

    def preview_carve(self, mod_id, sel_idx, hexa_sub="cap", orientation=0, conn_sel=None):
        geom, reason, _ = self._carve_plan(mod_id, sel_idx, hexa_sub, orientation, None, conn_sel or {})
        if geom is None:
            return None
        return CarvePreview(geom["md"], geom["sv"], geom["Q"], geom["tr"], geom["cfi"], reason is None, reason,
                            geom["w_verts"])

    def carve(self, mod_id, sel_idx, hexa_sub=None, orientation=None, color=None, conn_sel=None):
        """Carve `mod_id` out of the formation through open face `sel_idx`.
        A refusal leaves the state untouched. Reasons: "nothing to carve", "no
        connectable face", "face mismatch", "thin ring", "face not solid",
        "outside", "partial contact", "pinch", "open"."""
        hexa_sub = "cap" if hexa_sub is None else hexa_sub
        orientation = 0 if orientation is None else orientation
        conn_sel = {} if conn_sel is None else conn_sel
        geom, reason, plan = self._carve_plan(mod_id, sel_idx, hexa_sub, orientation, color, conn_sel)
        if reason or plan is None or geom is None:
            return PlaceResult(False, reason or "no connectable face")
        md, cfi, sc, Q, tr, w_verts = geom["md"], geom["cfi"], geom["sc"], geom["Q"], geom["tr"], geom["w_verts"]
        nxt, removed, wall_open = plan
        E = get_edge(self.X)
        used = self.open_faces[sel_idx]
        open_before = list(self.open_faces)
        self.all_faces = nxt
        # An open face the cut took any part of is no longer a whole face.
        self.open_faces = [o for i, o in enumerate(self.open_faces)
                           if i != sel_idx and not any(face_on_polygon(f, o.verts3, E) for f in removed)]
        self.open_faces.extend(wall_open)
        self.history.append(HistoryEntry(
            new_of_count=len(wall_open), used_idx=sel_idx, used_face=used, mod_centroid=cen_n(w_verts),
            snapshot_faces=list(self.all_faces), mod_id=mod_id, hexa_sub=hexa_sub, orientation=orientation,
            color=self.mat_color if color is None else color, conn_sel=conn_sel, cfi=cfi,
            Q=(Q.x, Q.y, Q.z, Q.w), tr=(tr.x, tr.y, tr.z), scale=sc,
            parent_mod_idx=used.module_idx, parent_face_idx=used.canonical_face_idx,
            step_x=self.X, open_before=open_before, carve=True,
        ))
        self.module_world_data.append(ModuleWorldData(w_verts, [list(f) for f in md.faces]))
        # The cut re-cut faces the last purge never saw: the next one runs
        # over everything, as the web's always does.
        self._purged_at = None
        return PlaceResult(True)

    def carve_at_key(self, mod_id, key, **kw):
        idx = self.find_open_face(key)
        if idx < 0:
            return PlaceResult(False, "face no longer open")
        return self.carve(mod_id, idx, **kw)

    def _carve_plan(self, mod_id, sel_idx, hexa_sub, orientation, color, conn_sel):
        """(geom, refusal, plan): everything carve() and preview_carve() need,
        decided without touching state."""
        tf = self._face(sel_idx)
        if tf is None:
            return None, "no such open face", None
        if tf.free:
            # A face modelled in Blender: the engine has no material behind it.
            return None, "modelled face", None
        md = self.mod_data(mod_id)
        if md is None:
            return None, "unknown module " + str(mod_id), None
        if not self.history or tf.module_idx < 0:
            return None, "nothing to carve", None
        if tf.conn_type and tf.conn_type != "open":
            return None, "no connectable face", None
        cfi = self.pick_conn_face_for(mod_id, hexa_sub, tf, conn_sel)
        if cfi < 0:
            return None, "no connectable face", None

        E = get_edge(self.X)
        eff = self._resolve_target(tf, cfi, md)
        fine = tf.fine if tf.fine is not None else 1
        sc = get_scale(md, self.X)
        # Seated against the face from BEHIND: the flipped normal puts the
        # body in the material.
        s = snap_transform(md, _Target(eff.verts3, eff.center3, eff.normal3.clone().negate()), orientation, sc, cfi)
        w_verts = [v.clone().apply_quaternion(s.Q).add(s.tr) for v in s.sv]
        geom = {"md": md, "sv": s.sv, "Q": s.Q, "tr": s.tr, "cfi": cfi, "sc": sc, "w_verts": w_verts}

        # Wholly inside the material, its centre confirmed as material too.
        if shell_crosses(w_verts, md.faces, self.all_faces):
            return geom, "outside", None
        cen = cen_n(w_verts)
        if not self._hull_list().enclosed(cen):
            return geom, "outside", None

        # Face by face: on the shell (the shell goes there), in the material
        # (a wall), or on part of it (cut out of it). The mouth must lie on
        # the shell wholly; a smaller one inside the picked face leaves a ring.
        remove = set()
        walls, wall_pfi, kept, scraps, cut_pts, rings = [], [], [], [], [], []
        for pfi, poly in enumerate(md.poly_faces):
            pts = [w_verts[i] for i in poly]
            inside, status = coplanar_cover(pts, self.all_faces, E)
            if pfi == cfi and status == "none":
                return geom, "face not solid", None
            if pfi == cfi and status == "mixed":
                ring = ring_around(eff.verts3, pts, self.all_faces, E)
                if ring == "thin":
                    return geom, "thin ring", None
                if ring is not None:
                    remove.update(ring[0])
                    rings.append((self.all_faces[ring[0][0]], ring[1]))
                    continue
            if status == "mixed":
                cut = cut_polygon(pts, self.all_faces, E)
                if not cut.ok:
                    return geom, ("face not solid" if pfi == cfi else "partial contact"), None
                if pfi == cfi and cut.uncovered:
                    return geom, "face mismatch", None
                remove.update(cut.cut)
                for frm, kp in cut.keep:
                    kept.append((self.all_faces[frm], kp))
                scraps.extend(cut.uncovered)
                for q in cut.uncovered + [k[1] for k in cut.keep]:
                    cut_pts.extend(q)
            elif status == "full":
                remove.update(inside)
            else:
                walls.append(pts)
                wall_pfi.append(pfi)

        mod_idx = len(self.history)
        col = self.mat_color if color is None else color
        wall_faces, wall_open = [], []

        def tri(a, b, c, pfi=None, sub_k=None):
            # No face_idx: a wall belongs to no printed module. poly_idx and
            # sub_k are the Blender side's, so a wall is one mesh polygon.
            af = AllFace(a.clone(), b.clone(), c.clone(), face_key(a, b, c), col, conn_type="open")
            af.mod_idx, af.poly_idx, af.sub_k = mod_idx, pfi, sub_k
            return af

        for w, pts in enumerate(walls):
            pfi = wall_pfi[w]
            poly = [p.clone() for p in pts]
            c = cen_n(poly)
            if poly_normal(poly).dot(Vec3().sub_vectors(cen, c)) < 0:
                poly.reverse()
            n = poly_normal(poly)
            N = len(poly)

            def open_(verts3, face_type, hex_sub=False, sub_k=None):
                of = OpenFace(verts3, cen_n(verts3), n.clone(), face_type, mod_idx, pfi, hex_sub=hex_sub)
                if fine < 1:
                    of.fine = fine
                of.sub_k = sub_k
                return of

            if N == 6:
                for k in range(N):
                    wall_faces.append(tri(c, poly[k], poly[(k + 1) % N], pfi, k))
                wall_open.append(open_(poly, 6))
                for k in range(N):
                    wall_open.append(open_([c.clone(), poly[k].clone(), poly[(k + 1) % N].clone()], 3, True, k))
            elif N == 4 and is_rhomb_face(poly):
                q = short_diagonal_first(poly)
                wall_faces.extend((tri(q[0], q[1], q[2], pfi), tri(q[0], q[2], q[3], pfi)))
                wall_open.append(open_([v.clone() for v in q], 4))
            else:
                for i in range(1, N - 1):
                    wall_faces.append(tri(poly[0], poly[i], poly[i + 1], pfi))
                wall_open.append(open_(poly, N))

        # A ring keeps its face's tags and winding, like what is left of a cut face.
        ring_faces = []
        for frm, tris in rings:
            out = poly_normal([frm.va, frm.vb, frm.vc])
            for p, q, r in tris:
                if poly_normal([p, q, r]).dot(out) < 0:
                    a, b, c = p.clone(), r.clone(), q.clone()
                else:
                    a, b, c = p.clone(), q.clone(), r.clone()
                ring_faces.append(frm.copy_with(a, b, c, face_key(a, b, c), split=True))

        removed = [f for i, f in enumerate(self.all_faces) if i in remove]
        nxt = [f for i, f in enumerate(self.all_faces) if i not in remove] + wall_faces + ring_faces
        gone, added = removed, wall_faces + ring_faces
        if cut_pts:
            for frm, pts in kept:
                poly = list(reversed(pts)) if poly_normal(pts).dot(poly_normal([frm.va, frm.vb, frm.vc])) < 0 else pts
                for i in range(1, len(poly) - 1):
                    a, b, c = poly[0].clone(), poly[i].clone(), poly[i + 1].clone()
                    nxt.append(frm.copy_with(a, b, c, face_key(a, b, c), split=True))
            for pts in scraps:
                poly = list(reversed(pts)) if poly_normal(pts).dot(Vec3().sub_vectors(cen, cen_n(pts))) < 0 else pts
                for i in range(1, len(poly) - 1):
                    nxt.append(tri(poly[0], poly[i], poly[i + 1]))
            nxt = split_at_points(nxt, cut_pts + w_verts, E * 1e-3)
            was = {id(f) for f in self.all_faces}
            now = {id(f) for f in nxt}
            gone = [f for f in self.all_faces if id(f) not in now]
            added = [f for f in nxt if id(f) not in was]
        damage = valence_damage(self.all_faces, gone, added, E)
        if damage:
            return geom, damage, None
        return geom, None, (nxt, removed, wall_open)

    # ── Interior purge ───────────────────────────────────────────────────────
    def _hull_list(self, extra=None):
        E = get_edge(self.X)

        def entry(verts, faces, box=None, mwd=None, void=False):
            cen = Vec3()
            for v in verts:
                cen.add(v)
            cen.divide_scalar(len(verts) or 1)
            b = (box if box is not None else box_from_points(verts)).clone().expand_by_scalar(E * 0.01)
            h = _Hull()
            h.verts, h.faces, h.box, h.cen, h.void, h.tol, h.mwd = verts, faces, b, cen, void, E * 1e-4, mwd
            return h

        H = self.history
        lst = [entry(m.verts, m.faces, m.box, m, i < len(H) and H[i].carve)
               for i, m in enumerate(self.module_world_data)]
        if extra is not None:
            lst.append(entry(extra[0], extra[1]))
        return _Hulls(lst)

    def _face_is_interior(self, a, b, c, hulls):
        E = get_edge(self.X)
        cen = cen3(a, b, c)
        off = nor3(a, b, c).multiply_scalar(E * 0.005)
        if not hulls.enclosed(cen.clone().add(off)):
            return False
        for v in (a, b, c):
            if not hulls.enclosed(cen.clone().lerp(v, 0.75).add(off)):
                return False
        return True

    # The web re-tests every face of the shell after each placement. Only the
    # new module's hull is new, and a face is buried only when every sample of
    # it lies in some hull, so a face whose samples cannot reach the new
    # hull's box keeps the verdict it had: it survived the last purge, so it
    # stays. `_purged_at` is the edge length the last purge ran at; anything
    # that could change a verdict otherwise (another Height X, an undo, a
    # fresh start) clears it and the next purge runs over everything, as the
    # web's does.
    def _near(self, w_verts):
        E = get_edge(self.X)
        b = box_from_points(w_verts).expand_by_scalar(E * 0.01 + E * 0.005 + 1e-9)
        mnx, mny, mnz, mxx, mxy, mxz = b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z

        def near(a, bb, c):
            return not (max(a.x, bb.x, c.x) < mnx or min(a.x, bb.x, c.x) > mxx
                        or max(a.y, bb.y, c.y) < mny or min(a.y, bb.y, c.y) > mxy
                        or max(a.z, bb.z, c.z) < mnz or min(a.z, bb.z, c.z) > mxz)
        return near

    def _purge_interior_faces(self, extra=None):
        if len(self.all_faces) < 4:
            self._purged_at = None
            return
        hulls = self._hull_list(extra)
        if extra is not None and self._purged_at == get_edge(self.X):
            near = self._near(extra[0])
            self.all_faces = [f for f in self.all_faces
                              if f.is_skirt or not near(f.va, f.vb, f.vc)
                              or not self._face_is_interior(f.va, f.vb, f.vc, hulls)]
            return
        self.all_faces = [f for f in self.all_faces
                          if f.is_skirt or not self._face_is_interior(f.va, f.vb, f.vc, hulls)]

    def _purge_sealed_pockets(self):
        F = self.all_faces
        if len(F) < 4:
            return
        E = get_edge(self.X)
        vid = {}

        def id_of(p):
            k = (math.floor(p.x * 1e3 + 0.5), math.floor(p.y * 1e3 + 0.5), math.floor(p.z * 1e3 + 0.5))
            i = vid.get(k)
            if i is None:
                i = len(vid)
                vid[k] = i
            return i

        tri = [(id_of(f.va), id_of(f.vb), id_of(f.vc)) for f in F]

        def ek(i, j):
            return (i, j) if i < j else (j, i)

        def tri_edges(t):
            return ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))

        edges = {}
        for i, t in enumerate(tri):
            for x, y in tri_edges(t):
                edges.setdefault(ek(x, y), []).append(i)
        if not any(len(a) > 2 for a in edges.values()):
            return

        par = list(range(len(F)))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for arr in edges.values():
            if len(arr) != 2:
                continue
            ra, rb = find(arr[0]), find(arr[1])
            if ra != rb:
                par[ra] = rb
        comps = {}
        for i in range(len(F)):
            comps.setdefault(find(i), []).append(i)
        if len(comps) < 2:
            return

        doomed = set()
        for members in comps.values():
            if len(members) >= len(F):
                continue
            if any(F[i].is_skirt for i in members):
                continue
            local = {}
            pinched = False
            for i in members:
                for x, y in tri_edges(tri[i]):
                    k = ek(x, y)
                    local[k] = local.get(k, 0) + 1
                    if len(edges.get(k, ())) > 2:
                        pinched = True
            if not pinched:
                continue
            if any(n != 2 for n in local.values()):
                continue
            vol = 0
            for i in members:
                f = F[i]
                vol += f.va.dot(Vec3().cross_vectors(f.vb, f.vc)) / 6
            if vol >= -(E * E * E) * 1e-6:
                continue
            doomed.update(members)
        if doomed:
            self.all_faces = [f for i, f in enumerate(F) if i not in doomed]

    # ── Normal merge placement ───────────────────────────────────────────────
    def _place_merge(self, ctx):
        md, cfi, to_world, w_verts, tf = ctx["md"], ctx["cfi"], ctx["to_world"], ctx["w_verts"], ctx["tf"]
        E = get_edge(self.X)
        is_first = len(self.history) == 0
        mod_idx = len(self.history)
        face_color = ctx["face_color"]

        mod_cen = Vec3()
        for v in w_verts:
            mod_cen.add(v)
        mod_cen.divide_scalar(len(w_verts))

        skirt_faces = self._build_skirt(ctx)
        tri_poly_map = build_tri_poly_map(md.poly_faces)
        is_poly_module = any(len(p) > 3 for p in md.poly_faces)

        self._remove_target_face(tf, E)

        new_mod_faces = []
        hex_poly_set = set()
        hex_tri_set = set()
        for pfi, poly in enumerate(md.poly_faces):
            if len(poly) == 6:
                hex_poly_set.add(pfi)
                for i in range(len(tri_poly_map)):
                    if tri_poly_map[i] == pfi:
                        hex_tri_set.add(i)

        for fi, f in enumerate(md.faces):
            if fi in hex_tri_set:
                continue
            if tri_poly_map[fi] == cfi and not is_first:
                continue
            va, vb, vc = to_world(f[0]), to_world(f[1]), to_world(f[2])
            if nor3(va, vb, vc).dot(Vec3().sub_vectors(mod_cen, cen3(va, vb, vc))) > 0:
                vb, vc = vc, vb
            af = AllFace(va, vb, vc, face_key(va, vb, vc), face_color, conn_type="open", conn_tag=None, face_idx=fi)
            af.poly_idx = tri_poly_map[fi]
            new_mod_faces.append(af)

        if hex_poly_set:
            for pfi, poly in enumerate(md.poly_faces):
                if pfi not in hex_poly_set:
                    continue
                if pfi == cfi and not is_first:
                    continue
                poly_wv = [to_world(i) for i in poly]
                c_hex = cen_n(poly_wv)
                nor_hex = nor_n(poly_wv)
                if nor_hex.dot(Vec3().sub_vectors(mod_cen, c_hex)) > 0:
                    poly_wv.reverse()
                N = len(poly_wv)
                for k in range(N):
                    va, vb, vc = c_hex.clone(), poly_wv[k].clone(), poly_wv[(k + 1) % N].clone()
                    af = AllFace(va, vb, vc, face_key(va, vb, vc), face_color, conn_type="open")
                    af.poly_idx = pfi
                    af.sub_k = k
                    new_mod_faces.append(af)

        for sf in skirt_faces:
            sf.is_skirt = True
            new_mod_faces.append(sf)
        for nf in new_mod_faces:
            nf.mod_idx = mod_idx

        # Cancel shared interior walls (skirts never cancel). The web scans
        # the whole shell for the first face whose centroid sum lies within
        # E*0.06 (L1) of the new face's, splices it out or appends the new
        # face. Same answer here, found through a hash of centroid sums: a
        # match is within one cell, and the lowest position among the cells'
        # candidates is the web's first match. Removed faces are marked and
        # dropped at the end, which leaves the others in the web's order.
        faces = self.all_faces
        alive = [True] * len(faces)
        lim = E * 0.06
        buckets = {}

        def cell(sx, sy, sz):
            return (math.floor(sx / lim), math.floor(sy / lim), math.floor(sz / lim))

        def file(i, f):
            k = cell(f.va.x + f.vb.x + f.vc.x, f.va.y + f.vb.y + f.vc.y, f.va.z + f.vb.z + f.vc.z)
            buckets.setdefault(k, []).append(i)

        for i, f in enumerate(faces):
            if not f.is_skirt:
                file(i, f)
        for nf in new_mod_faces:
            if nf.is_skirt:
                faces.append(nf)
                alive.append(True)
                continue
            if nf.conn_type == "closed":
                continue
            if nf.conn_type == "visual":
                faces.append(nf)
                alive.append(True)
                file(len(faces) - 1, nf)
                continue
            nx = (nf.va.x, nf.vb.x, nf.vc.x)
            ny = (nf.va.y, nf.vb.y, nf.vc.y)
            nz = (nf.va.z, nf.vb.z, nf.vc.z)
            cx, cy, cz = cell(nx[0] + nx[1] + nx[2], ny[0] + ny[1] + ny[2], nz[0] + nz[1] + nz[2])
            dup = -1
            for ix in (cx - 1, cx, cx + 1):
                for iy in (cy - 1, cy, cy + 1):
                    for iz in (cz - 1, cz, cz + 1):
                        for i in buckets.get((ix, iy, iz), ()):
                            if not alive[i] or (dup != -1 and i >= dup):
                                continue
                            ef = faces[i]
                            dc = (abs(nx[0] + nx[1] + nx[2] - ef.va.x - ef.vb.x - ef.vc.x)
                                  + abs(ny[0] + ny[1] + ny[2] - ef.va.y - ef.vb.y - ef.vc.y)
                                  + abs(nz[0] + nz[1] + nz[2] - ef.va.z - ef.vb.z - ef.vc.z))
                            if dc < lim:
                                dup = i
            if dup != -1:
                alive[dup] = False
            else:
                faces.append(nf)
                alive.append(True)
                file(len(faces) - 1, nf)
        self.all_faces = [f for f, a in zip(faces, alive) if a]

        self._purge_interior_faces((w_verts, md.faces))
        self._purge_sealed_pockets()

        new_open = []
        if is_poly_module:
            keys = {af.key for af in self.all_faces}
            for pfi, poly in enumerate(md.poly_faces):
                if pfi == cfi and not is_first:
                    continue
                poly_wv = [to_world(i) for i in poly]
                cen = cen_n(poly_wv)
                nor = nor_n(poly_wv)
                if nor.dot(Vec3().sub_vectors(mod_cen, cen)) > 0:
                    nor = nor.clone().negate()
                    poly_wv.reverse()
                if pfi in hex_poly_set:
                    N = len(poly_wv)
                    sub_keys = [face_key(cen.clone(), poly_wv[k].clone(), poly_wv[(k + 1) % N].clone()) for k in range(N)]
                    if all(k in keys for k in sub_keys):
                        new_open.append(OpenFace([v.clone() for v in poly_wv], cen.clone(), nor.clone(), 6, mod_idx, pfi))
                    for k in range(N):
                        if sub_keys[k] not in keys:
                            continue
                        va, vb, vc = cen.clone(), poly_wv[k].clone(), poly_wv[(k + 1) % N].clone()
                        sub = OpenFace([va, vb, vc], cen3(va, vb, vc), nor.clone(), 3, mod_idx, pfi, hex_sub=True)
                        sub.sub_k = k
                        new_open.append(sub)
                    continue
                v0, v1, v2 = to_world(poly[0]), to_world(poly[1]), to_world(poly[2])
                if face_key(v0, v1, v2) not in keys and face_key(v0, v2, v1) not in keys:
                    continue
                verts3 = short_diagonal_first(poly_wv) if (len(poly) == 4 and is_rhomb_face(poly_wv)) else poly_wv
                new_open.append(OpenFace(verts3, cen, nor, len(poly), mod_idx, pfi))
        else:
            keys = {af.key for af in self.all_faces}
            for nf in new_mod_faces:
                if nf.is_skirt or nf.conn_type in ("closed", "visual"):
                    continue
                if nf.key in keys:
                    new_open.append(OpenFace([nf.va, nf.vb, nf.vc], cen3(nf.va, nf.vb, nf.vc), nor3(nf.va, nf.vb, nf.vc),
                                             3, mod_idx, nf.face_idx, conn_type=nf.conn_type or "open",
                                             conn_tag=nf.conn_tag or None))
        self._commit(ctx, new_open)

    def _build_skirt(self, ctx):
        md, cfi, to_world, tf, face_color = ctx["md"], ctx["cfi"], ctx["to_world"], ctx["tf"], ctx["face_color"]
        cf = md.poly_faces[cfi]
        M = len(cf)
        conn = [to_world(i) for i in cf]
        conn_edge = conn[0].distance_to(conn[1])
        target_edge = tf.verts3[0].distance_to(tf.verts3[1])
        E = get_edge(self.X)
        t_pts = tf.verts3
        N = len(t_pts)
        sized = abs(conn_edge - target_edge) <= target_edge * 0.02
        fitted = M == N and all(any(q.distance_to(p) <= E * 0.02 for q in t_pts) for p in conn)
        if fitted:
            return []
        if sized and len(self.history) == 0:
            return []
        if not sized:
            return self._skirt_band(ctx, conn, t_pts, M, N, E)

        def in_poly(p, poly):
            sign = 0
            eps = E * E * 1e-6
            for i in range(len(poly)):
                a, b = poly[i], poly[(i + 1) % len(poly)]
                c = Vec3().cross_vectors(Vec3().sub_vectors(b, a), Vec3().sub_vectors(p, a)).dot(tf.normal3)
                if abs(c) <= eps:
                    continue
                s = 1 if c > 0 else -1
                if sign == 0:
                    sign = s
                elif s != sign:
                    return False
            return True

        nested = all(in_poly(p, t_pts) for p in conn) or all(in_poly(p, conn) for p in t_pts)
        if nested:
            return self._skirt_band(ctx, conn, t_pts, M, N, E)

        def fan(pts, d):
            out = []
            for i in range(1, len(pts) - 1):
                a, b, c = pts[0], pts[i], pts[i + 1]
                keep = nor3(a, b, c).dot(d) >= 0
                p, q, r = a, (b if keep else c), (c if keep else b)
                out.append(AllFace(p.clone(), q.clone(), r.clone(), face_key(p, q, r), face_color, is_skirt=True))
            return out

        return fan(t_pts, tf.normal3) + fan(conn, tf.normal3.clone().negate())

    def _skirt_band(self, ctx, conn, t_pts, M, N, E):
        tf, face_color = ctx["tf"], ctx["face_color"]
        matched = [0] * M
        for ci in range(M):
            best_d, best_j = math.inf, 0
            for cj in range(N):
                d = conn[ci].distance_to(t_pts[cj])
                if d < best_d:
                    best_d, best_j = d, cj
            matched[ci] = best_j
        out = []
        outward = tf.normal3

        def tri(pA, pB, pC):
            if nor3(pA, pB, pC).dot(outward) >= 0:
                return AllFace(pA.clone(), pB.clone(), pC.clone(), face_key(pA, pB, pC), face_color, is_skirt=True)
            return AllFace(pA.clone(), pC.clone(), pB.clone(), face_key(pA, pC, pB), face_color, is_skirt=True)

        for ei in range(M):
            ej = (ei + 1) % M
            cA, cB = conn[ei], conn[ej]
            out.append(tri(cA, cB, t_pts[matched[ej]]))
            frm, to = matched[ei], matched[ej]
            fwd = (to - frm + N) % N
            step = 0 if fwd == 0 else (1 if fwd <= N - fwd else -1)
            k, guard = frm, 0
            while k != to and guard < N:
                kn = (k + step + N) % N
                if t_pts[k].distance_to(t_pts[kn]) > E * 0.01:
                    out.append(tri(cA, t_pts[kn], t_pts[k]))
                k = kn
                guard += 1
        return out

    def _remove_target_face(self, tf, E):
        N = len(tf.verts3)
        lim = E * 0.002
        if N == 3:
            a, b, c = tf.verts3
            cx = a.x + b.x + c.x
            cy = a.y + b.y + c.y
            cz = a.z + b.z + c.z
            tkey = face_key(a, b, c)

            def keep(f):
                if f.key == tkey:
                    return False
                if f.is_skirt:
                    return True
                d = (abs(f.va.x + f.vb.x + f.vc.x - cx) + abs(f.va.y + f.vb.y + f.vc.y - cy)
                     + abs(f.va.z + f.vb.z + f.vc.z - cz))
                return d >= lim
        else:
            refs = []
            if N == 6:
                c_hex = cen_n(tf.verts3)
                for k in range(6):
                    vb, vc = tf.verts3[k], tf.verts3[(k + 1) % 6]
                    refs.append((c_hex.x + vb.x + vc.x, c_hex.y + vb.y + vc.y, c_hex.z + vb.z + vc.z,
                                 face_key(c_hex, vb, vc), face_key(c_hex, vc, vb)))
            else:
                v0 = tf.verts3[0]
                for i in range(1, N - 1):
                    vi, vj = tf.verts3[i], tf.verts3[i + 1]
                    refs.append((v0.x + vi.x + vj.x, v0.y + vi.y + vj.y, v0.z + vi.z + vj.z,
                                 face_key(v0, vi, vj), face_key(v0, vj, vi)))

            def keep(f):
                if f.is_skirt:
                    return True
                for r in refs:
                    if f.key == r[3] or f.key == r[4]:
                        return False
                    d = (abs(f.va.x + f.vb.x + f.vc.x - r[0]) + abs(f.va.y + f.vb.y + f.vc.y - r[1])
                         + abs(f.va.z + f.vb.z + f.vc.z - r[2]))
                    if d < lim:
                        return False
                return True
        self.all_faces = [f for f in self.all_faces if keep(f)]
        # Triangles a carve re-cut match no centroid above: found by lying on
        # the face instead.
        if any(f.split for f in self.all_faces):
            self.all_faces = [f for f in self.all_faces
                              if not f.split or f.is_skirt or not face_on_polygon(f, tf.verts3, E)]

    def _commit(self, ctx, new_open):
        md, w_verts, sel_idx = ctx["md"], ctx["w_verts"], ctx["sel_idx"]
        E = get_edge(self.X)
        mc = Vec3()
        for v in w_verts:
            mc.add(v)
        mc.divide_scalar(len(w_verts))
        used = self.open_faces[sel_idx]
        Q, tr = ctx["Q"], ctx["tr"]
        # With divisions about, the target face's edges may carry grid
        # corners the new module's faces run straight past: re-cut there.
        if self.divides:
            T = ctx["tf"].verts3
            n = len(T)
            self.all_faces = split_along_segments(self.all_faces, [(T[k], T[(k + 1) % n]) for k in range(n)],
                                                  E * 1e-3)
        # A carve that cut part of a face left corners along the cut, so a
        # module placed back against it meets edges running past corners:
        # split both ways at every corner in reach.
        if self.carve_count:
            box = box_from_points(w_verts).expand_by_scalar(E * 1e-3)
            pts = list(w_verts)
            for f in self.all_faces:
                for p in (f.va, f.vb, f.vc):
                    if box.contains_point(p):
                        pts.append(p)
            self.all_faces = split_at_points(self.all_faces, pts, E * 1e-3)
        self.history.append(HistoryEntry(
            new_of_count=len(new_open), used_idx=sel_idx, used_face=used, mod_centroid=mc.clone(),
            snapshot_faces=list(self.all_faces), mod_id=ctx["mod_id"], hexa_sub=ctx["hexa_sub"],
            orientation=ctx["orientation"], color=ctx["face_color"], conn_sel=ctx["conn_sel"], cfi=ctx["cfi"],
            Q=(Q.x, Q.y, Q.z, Q.w), tr=(tr.x, tr.y, tr.z), scale=ctx["sc"],
            parent_mod_idx=used.module_idx, parent_face_idx=used.canonical_face_idx,
            step_x=self.X, open_before=list(self.open_faces),
        ))
        if used.free:
            self.history[-1].free = dict(used.free)
        del self.open_faces[sel_idx]
        # Faces of a module grown on a divided face are that small too.
        if ctx.get("fine", 1) < 1:
            for nof in new_open:
                nof.fine = ctx["fine"]
        lost, added = [], []
        lim = E * 0.02
        for nof in new_open:
            di = -1
            for i, ef in enumerate(self.open_faces):
                if ef.center3.distance_to(nof.center3) < lim:
                    di = i
                    break
            if di != -1:
                lost.append(self.open_faces[di])
                del self.open_faces[di]
            else:
                self.open_faces.append(nof)
                added.append(nof)
        buried = []
        if len(self.all_faces) >= 4:
            hulls = self._hull_list((w_verts, md.faces))
            near = self._near(w_verts) if self._purged_at == E else None
            kept = []
            for of in self.open_faces:
                v = of.verts3
                if (near is None or near(v[0], v[1], v[2])) and self._face_is_interior(v[0], v[1], v[2], hulls):
                    buried.append(of)
                else:
                    kept.append(of)
            self.open_faces = kept
            self._purged_at = E
        else:
            self._purged_at = None
        added_ids = {id(f) for f in added}
        buried_old = [f for f in buried if id(f) not in added_ids]
        step = self.history[-1]
        step.new_of_count = len(added) - (len(buried) - len(buried_old))
        taken = lost + buried_old
        if taken:
            step.removed_open_faces = (step.removed_open_faces or []) + taken
        self.module_world_data.append(ModuleWorldData(w_verts, [list(f) for f in md.faces]))
        if len(self.module_world_data) > len(self.history):
            del self.module_world_data[len(self.history):]

    # ── Keys, undo ───────────────────────────────────────────────────────────
    @staticmethod
    def face_key_of(f):
        c = f.center3
        fine = ("|" + to_fixed(f.fine, 6)) if f.fine is not None else ""
        return "%s,%s,%s|%d|%d%s" % (to_fixed(c.x, 4), to_fixed(c.y, 4), to_fixed(c.z, 4),
                                     f.face_type, 1 if f.hex_sub else 0, fine)

    def open_face_key(self, idx):
        f = self._face(idx)
        return BuilderEngine.face_key_of(f) if f else None

    def find_open_face(self, key):
        for i, f in enumerate(self.open_faces):
            if BuilderEngine.face_key_of(f) == key:
                return i
        return -1

    def undo(self):
        if not self.can_undo:
            return False
        # The newest step may be a division rather than a module.
        d = self.divides[-1] if self.divides else None
        if d is not None and d.at == len(self.history):
            self.divides.pop()
            self.open_faces = list(d.open_before)
            self.all_faces = list(d.all_before)
            self._purged_at = None
            return True
        if not self.history:
            return False
        last = self.history.pop()
        self._purged_at = None
        if len(self.module_world_data) > len(self.history):
            self.module_world_data.pop()
        if last.open_before is not None:
            self.open_faces = list(last.open_before)
        dv = self.divides[-1] if self.divides else None
        if dv is not None and dv.at == len(self.history):
            self.all_faces = list(dv.all_after)
        elif not self.history:
            self.all_faces = self._seed_tris(self._start_seed(self.seed_shape))
        else:
            snap = self.history[-1].snapshot_faces
            self.all_faces = list(snap) if snap is not None else []
        return True

    # ── Face division ────────────────────────────────────────────────────────
    # divide(sel_idx, n) splits one open face into n x n smaller copies of
    # itself, each 1/n of its edge: ordinary open faces carrying `fine`, the
    # same module's, the same face index. A division is not a module: it is
    # kept in `divides`, ordered among the modules by `at`, and undoes, saves
    # and replays as a step. The grid's corners land in the middle of the
    # face's edges, so the faces across them are re-cut there.
    def face_x(self, idx):
        """The Height X a module fits open face `idx` at exactly (unrounded)."""
        f = self._face(idx)
        if f is None or len(f.verts3) < 2:
            return None
        return f.verts3[0].distance_to(f.verts3[1]) * math.sqrt(3) / 2

    def divide_refusal(self, sel_idx, n):
        if sel_idx is None:
            return "no face"
        tf = self._face(sel_idx)
        if tf is None:
            return "no such open face"
        if not isinstance(n, int) or n < 2 or n > DIVIDE_MAX:
            return "bad count"
        if tf.conn_type and tf.conn_type != "open":
            return "connector face"
        if tf.free:
            return "modelled face"
        if tf.hex_sub or len(tf.verts3) != (tf.face_type if tf.face_type is not None else 3):
            return "shape"
        if (not is_parallelogram(tf.verts3, get_edge(self.X) * 0.01)) if len(tf.verts3) == 4 else len(tf.verts3) != 3:
            return "shape"
        if (tf.fine if tf.fine is not None else 1) / n < DIVIDE_MIN_FINE - 1e-9:
            return "too fine"
        return None

    def divide(self, sel_idx, n):
        why = self.divide_refusal(sel_idx, n)
        if why:
            return PlaceResult(False, why)
        tf = self.open_faces[sel_idx]
        E = get_edge(self.X)
        all_before = list(self.all_faces)
        open_before = list(self.open_faces)
        self._remove_target_face(tf, E)
        kept = {id(f) for f in self.all_faces}
        removed = [f for f in all_before if id(f) not in kept]
        proto = next((f for f in removed if not f.is_skirt), None)
        if proto is None or any(f.is_skirt for f in removed):
            self.all_faces = all_before
            return PlaceResult(False, "face not solid")
        fine = (tf.fine if tf.fine is not None else 1) / n
        flip = poly_normal(tf.verts3).dot(tf.normal3) < 0
        tris, opened = [], []
        for raw in subdivide_polygon(tf.verts3, n):
            p = list(reversed(raw)) if flip else raw
            for i in range(1, len(p) - 1):
                a, b, c = p[0].clone(), p[i].clone(), p[i + 1].clone()
                tris.append(proto.copy_with(a, b, c, face_key(a, b, c), split=False))
            of = OpenFace(p, cen_n(p), tf.normal3.clone(), len(p), tf.module_idx, tf.canonical_face_idx)
            of.fine = fine
            opened.append(of)
        N = len(tf.verts3)
        edges = [(tf.verts3[k], tf.verts3[(k + 1) % N]) for k in range(N)]
        self.all_faces = split_along_segments(self.all_faces + tris, edges, E * 1e-3)
        self.open_faces = [f for i, f in enumerate(self.open_faces) if i != sel_idx] + opened
        self._purged_at = None
        self.divides.append(DivideEntry(at=len(self.history), n=n, used_idx=sel_idx, used_face=tf,
                                        open_before=open_before, all_before=all_before,
                                        all_after=list(self.all_faces), step_x=self.X))
        return PlaceResult(True)

    def divide_at_key(self, key, n):
        idx = self.find_open_face(key)
        if idx < 0:
            return PlaceResult(False, "face no longer open")
        return self.divide(idx, n)

    # ── Save format (SerializedBuild, version 1) ─────────────────────────────
    def serialize(self):
        doc = {"v": 1, "X": self.X, "color": self.mat_color, "seedShape": self.seed_shape}
        # No custom module can be placed here, so no "mods" are ever written.
        for k in ("ground", "texTile", "uv", "rig"):
            if k not in self.extras:
                continue
            if k == "texTile" and (not _is_num(self.extras[k]) or self.extras[k] == 1):
                continue
            doc[k] = self.extras[k]
        steps = []
        H = len(self.history)
        for at in range(H + 1):
            # A division is a step of its own, slotted in where it was made.
            for dv in self.divides:
                if dv.at == at:
                    steps.append({"modId": DIVIDE_STEP, "usedIdx": dv.used_idx, "hexaSub": "", "orientation": 0,
                                  "color": "", "connSel": {}, "stepX": dv.step_x, "divide": dv.n})
            if at == H:
                break
            h = self.history[at]
            if h.free:
                fr = dict(h.free)
                fr["mod"] = h.mod_id
                steps.append({"modId": FREE_STEP, "usedIdx": h.used_idx, "hexaSub": h.hexa_sub,
                              "orientation": h.orientation, "color": h.color,
                              "connSel": dict(h.conn_sel or {}), "free": fr,
                              "stepX": h.step_x if h.step_x is not None else self.X})
                continue
            st = {
                "modId": h.mod_id, "usedIdx": h.used_idx, "hexaSub": h.hexa_sub,
                "orientation": h.orientation, "color": h.color, "connSel": dict(h.conn_sel or {}),
                "stepX": h.step_x if h.step_x is not None else self.X,
            }
            if h.carve:
                st["carve"] = True
            if h.tex:
                st["tex"] = dict(h.tex)
            steps.append(st)
        doc["steps"] = steps
        return doc

    def load(self, doc):
        """Replay a SerializedBuild. Returns (placed, failed) with failed None
        or {"index", "modId", "reason"}. Every failure stops the replay: a step
        names its face by position, so skipping one would land every later
        step on the wrong face."""
        d = doc if isinstance(doc, dict) else {}
        self.X = d["X"] if _is_num(d.get("X")) else 5
        self.mat_color = d["color"] if isinstance(d.get("color"), str) else "#c44444"
        steps = d["steps"] if isinstance(d.get("steps"), list) else []
        first = steps[0].get("stepX") if steps and isinstance(steps[0], dict) else None
        if _is_num(first):
            self.X = first
        self.init(d["seedShape"] if _is_num(d.get("seedShape")) else 3)
        self.extras = {k: d[k] for k in ("mods", "ground", "texTile", "uv", "rig") if k in d}
        placed = 0
        failed = None
        for i, s in enumerate(steps):
            if not isinstance(s, dict) or not isinstance(s.get("modId"), str) or not _is_num(s.get("usedIdx")):
                failed = {"index": i, "modId": str(s.get("modId", "?") if isinstance(s, dict) else "?"),
                          "reason": "malformed step"}
                break
            if _is_num(s.get("stepX")):
                self.X = s["stepX"]
            mid = s["modId"]
            if mid == FREE_STEP:
                fr = s.get("free")
                if not isinstance(fr, dict) or fr.get("mod") not in MOD_DATA:
                    failed = {"index": i, "modId": mid, "reason": "malformed step"}
                    break
                try:
                    of = OpenFace.from_free({k: v for k, v in fr.items() if k != "mod"})
                except (TypeError, KeyError, ValueError):
                    failed = {"index": i, "modId": mid, "reason": "malformed step"}
                    break
                self.open_faces = [o for o in self.open_faces if not o.free] + [of]
                cs = s.get("connSel")
                r = self.place(fr["mod"], len(self.open_faces) - 1, hexa_sub=s.get("hexaSub"),
                               orientation=s.get("orientation"), color=s.get("color"),
                               conn_sel=cs if isinstance(cs, dict) else None)
                self.open_faces = [o for o in self.open_faces if not o.free]
                if not r.ok:
                    failed = {"index": i, "modId": mid, "reason": r.reason}
                    break
                placed += 1
                continue
            if mid == DIVIDE_STEP:
                ui = s["usedIdx"]
                n = s.get("divide")
                r = (self.divide(int(ui) if float(ui).is_integer() else -1, int(n))
                     if _is_num(n) and float(n).is_integer() else PlaceResult(False, "malformed step"))
                if not r.ok:
                    failed = {"index": i, "modId": mid, "reason": r.reason}
                    break
                continue
            if mid not in MOD_DATA:
                why = "a Factory module (not in the add-on yet)" if any(
                    isinstance(m, dict) and m.get("id") == mid for m in self.extras.get("mods") or []) else "unknown module " + mid
                failed = {"index": i, "modId": mid, "reason": why}
                break
            ui = s["usedIdx"]
            idx = int(ui) if float(ui).is_integer() else -1
            cs = s.get("connSel")
            do = self.carve if s.get("carve") is True else self.place
            r = do(mid, idx, hexa_sub=s.get("hexaSub"), orientation=s.get("orientation"),
                   color=s.get("color"), conn_sel=cs if isinstance(cs, dict) else None)
            if not r.ok:
                failed = {"index": i, "modId": mid, "reason": r.reason}
                break
            tex = s.get("tex")
            if isinstance(tex, dict):
                clean = {k: v for k, v in tex.items() if isinstance(k, str) and isinstance(v, str)}
                if clean:
                    self.history[-1].tex = clean
            placed += 1
        self.X = d["X"] if _is_num(d.get("X")) else 5
        return placed, failed
