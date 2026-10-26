"""The formation as a Blender mesh object, edited in Edit Mode.

The engine works Y-up, as the web does; Blender is Z-up. A point (x, y, z) of
the engine is (x, -z, y) in the object's space, the turn Blender's OBJ
importer gives a web export. One engine unit is one Blender unit.

The mesh is the engine's closed shell, rebuilt after every change:
  - each open face (where a module can go) is one polygon, so a square is a
    quad and a pentagon an n-gon, and selecting it in Edit Mode picks it;
  - a hexagon stays its six centre triangles, as the engine keeps it: one of
    them selected is that sub-seat, all six the whole hexagon;
  - the rest (skirts, faces nothing can attach to) stays triangles.
Face attributes say what each polygon is: "pe_open" (the open face it is,
-1 for none) and "pe_hex" (the whole hexagon it is part of). Corner colour
"PE Color" is each module's colour.

Blender's own undo and the engine are kept in step by a write id: every
rebuild stamps a fresh random id on every face ("pe_wid") and files the
formation's save under it. Ctrl+Z in Edit Mode brings back an older mesh with
an older id, and `sync` reopens the formation filed under that id. The latest
save is also kept on the object ("pe_formation", the web's save format), so
the .blend carries it.

The object's origin is the web Form's base plate (engine/frame.py): the mesh
stands in local space with the plate's centre at the origin and up local +Z.
Every vertex keeps the point it was built from ("pe_base", which Blender
leaves alone when it moves the mesh as a whole), so whatever Set Origin,
Affect Only Origins or Apply did is read back off the mesh (`read_frame`) and
kept by the next rebuild. It is saved as the web save's `ground`.
"""

import json
import math
import os
import random

import bpy
import bmesh

from ..engine import frame as FR
from ..engine.geom import get_edge
from . import usergeo
from ..engine.form import FormState, Settings
from ..engine.meshplan import mesh_plan, to_bl  # noqa: F401

DOC_PROP = "pe_formation"
WID_PROP = "pe_wid"
FRAME_PROP = "pe_frame"
BASE_ATTR = "pe_base"
MARK_PROP = "pe_form"
COLOR_ATTR = "PE Color"
OPEN_ATTR = "pe_open"
HEX_ATTR = "pe_hex"
WID_ATTR = "pe_wid"
MAT_NAME = "PE Formation"
DOCS_KEPT = 400


def is_formation(obj):
    return obj is not None and obj.type == "MESH" and bool(obj.get(MARK_PROP))


def edit_formation(context):
    """The formation being edited, or None."""
    obj = getattr(context, "edit_object", None)
    if obj is not None and context.mode == "EDIT_MESH" and is_formation(obj):
        return obj
    return None


_settings = None


def settings_path():
    """The settings file, in the extension's own user directory (which an
    upgrade keeps; the add-on's folder may be read-only). Run as a plain
    module (the tests), Blender's config folder instead."""
    from .. import __package__ as base_package
    try:
        base = bpy.utils.extension_path_user(base_package, create=True)
    except (TypeError, ValueError, AttributeError):
        try:
            base = bpy.utils.user_resource("CONFIG", path="pe_form", create=True)
        except (TypeError, ValueError):
            base = os.path.join(bpy.utils.user_resource("CONFIG"), "pe_form")
    return os.path.join(base, "settings.json")


def settings():
    global _settings
    if _settings is None:
        try:
            _settings = Settings(settings_path())
        except Exception:
            _settings = Settings(None)
    return _settings


def material():
    mat = bpy.data.materials.get(MAT_NAME)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(MAT_NAME)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        attr = nt.nodes.new("ShaderNodeVertexColor")
        attr.layer_name = COLOR_ATTR
        attr.location = (bsdf.location.x - 260, bsdf.location.y)
        nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.55
    return mat


# ── the formation's engine ───────────────────────────────────────────────────
class Formation:
    """One formation object's engine, the saves filed by write id, and a memo."""

    _all = {}

    def __init__(self, obj):
        self.state = FormState(settings())
        self.wid = None
        self.docs = {}
        self._memo = {}
        self.frame = stored_frame(obj)
        # What the mesh was written from (usergeo): the plan, its index by
        # polygon, the open faces modelled away, the seats sealed by a module.
        self.plan_items = None
        self.mesh_plan = None
        self.taken_open = set()
        self.sealed = []
        # Pairs of the Form's corners the mesh keeps as separate copies
        # (usergeo.unjoined), found the last time its faces were sorted.
        self.unjoined = 0
        self.dirty = True

    @classmethod
    def of(cls, obj):
        key = obj.session_uid
        f = cls._all.get(key)
        if f is None:
            f = cls._all[key] = Formation(obj)
            raw = obj.get(DOC_PROP)
            if isinstance(raw, str):
                try:
                    doc = json.loads(raw)
                except ValueError:
                    doc = None
                if doc is not None:
                    f.state.load_doc(doc)
                    f.state.status = "Click a face to select"
                    f.wid = obj.get(WID_PROP)
                    if f.wid is not None:
                        f.docs[f.wid] = raw
        return f

    @classmethod
    def forget(cls, obj=None):
        if obj is None:
            cls._all.clear()
        else:
            cls._all.pop(obj.session_uid, None)

    def memo(self, name, fn):
        st = self.state
        k = (st.version, st.sel_idx, st.active_mod_id, st.hexa_sub, st.orientation,
             tuple(sorted(st.conn_sel.items())), st.engine.X, len(st.settings.pinned_mods),
             st.carve_mode, tuple(st.picks))
        hit = self._memo.get(name)
        if hit is not None and hit[0] == k:
            return hit[1]
        v = fn()
        self._memo[name] = (k, v)
        return v

    def file(self, wid, raw):
        self.docs[wid] = raw
        if len(self.docs) > DOCS_KEPT:
            for k in list(self.docs)[: len(self.docs) - DOCS_KEPT]:
                del self.docs[k]


# ── the frame: the origin as the base plate ─────────────────────────────────
def stored_frame(obj):
    return FR.unflat(obj.get(FRAME_PROP)) or [r[:] for r in FR.IDENT]


def _mesh_points(obj, only=None):
    """(built-from points, points now) of the mesh's vertices, all or those
    numbered in `only`; None when the mesh keeps no built-from points."""
    me = obj.data
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(me)
        lb = bm.verts.layers.float_vector.get(BASE_ATTR)
        if lb is None or not len(bm.verts):
            return None
        bm.verts.ensure_lookup_table()
        vs = bm.verts if only is None else [bm.verts[i] for i in only]
        return [tuple(v[lb]) for v in vs], [tuple(v.co) for v in vs]
    a = me.attributes.get(BASE_ATTR)
    n = len(me.vertices)
    if a is None or a.domain != "POINT" or a.data_type != "FLOAT_VECTOR" or not n:
        return None
    if only is None:
        b, c = [0.0] * (3 * n), [0.0] * (3 * n)
        a.data.foreach_get("vector", b)
        me.vertices.foreach_get("co", c)
        return ([(b[i], b[i + 1], b[i + 2]) for i in range(0, 3 * n, 3)],
                [(c[i], c[i + 1], c[i + 2]) for i in range(0, 3 * n, 3)])
    return [tuple(a.data[i].vector) for i in only], [tuple(me.vertices[i].co) for i in only]


def read_frame(obj):
    """The frame the mesh stands in now: the one stored, unless Blender has
    moved the mesh as a whole since (Set Origin...), when it is read off the
    mesh. Cheap when nothing moved: three vertices are compared."""
    M = stored_frame(obj)
    try:
        n = len(obj.data.vertices) if obj.mode != "EDIT" else len(bmesh.from_edit_mesh(obj.data).verts)
        if n < 3:
            return M
        got = _mesh_points(obj, (0, n // 2, n - 1))
        if got is None:
            return M
        size = max(1.0, FR.scale_of(M))
        if all(FR._len(FR._sub(FR.apply(M, b), c)) <= 1e-5 * size for b, c in zip(*got)):
            return M
        base, cur = _mesh_points(obj)
        return FR.fit(base, cur) or M
    except (ReferenceError, ValueError, IndexError):
        return M


def keep_frame(obj, M):
    """Remember a frame: on the object, and in the engine's save as the
    web's ground."""
    obj[FRAME_PROP] = FR.flat(M)
    f = Formation.of(obj)
    f.frame = M
    g = FR.to_ground(M)
    ex = f.state.engine.extras
    if g is None:
        ex.pop("ground", None)
    else:
        ex["ground"] = g


NAMES = (OPEN_ATTR, HEX_ATTR, WID_ATTR, BASE_ATTR, COLOR_ATTR)


def _tol(e):
    return get_edge(e.X) * 1e-4 + 1e-6


def _modelled(bm, L, M, old_plan, tol):
    """Has anything been modelled on the mesh since the Form wrote it?"""
    if old_plan is None:
        return False
    lr = L["role"]
    if len(bm.faces) != len(old_plan) or any(f[lr] != usergeo.ENGINE for f in bm.faces):
        return True
    lb = L["base"]
    for v in bm.verts:
        p = FR.apply(M, tuple(v[lb]))
        c = v.co
        if (p[0] - c.x) ** 2 + (p[1] - c.y) ** 2 + (p[2] - c.z) ** 2 > tol * tol:
            return True
    return False


def _write_fresh(bm, L, M, items, wid, select):
    """The whole mesh written from the plan: nothing modelled to keep."""
    # Delete the geometry, not bm.clear(): clear() also drops the mesh's
    # attribute layers, and a "PE Color" made again is not the active
    # colour, so Edit Mode showed the formation black.
    bmesh.ops.delete(bm, geom=list(bm.verts), context="VERTS")
    vmap = {}
    lb = L["base"]

    def vert(p):
        # The plan's corners are joined already: one point is one vertex.
        k = tuple(p)
        v = vmap.get(k)
        if v is None:
            v = bm.verts.new(FR.apply(M, p))
            v[lb] = p
            vmap[k] = v
        return v

    active = None
    for pts, col, oi, hx, k, src, m in items:
        try:
            face = bm.faces.new([vert(p) for p in pts])
        except ValueError:
            continue
        face[L["mod"]] = m + 1
        face[L["open"]] = oi
        face[L["hex"]] = hx
        face[L["wid"]] = wid
        face[L["role"]] = usergeo.ENGINE
        if src is not None:
            face[L["src"]] = src[0]
            face[L["srcn"]] = src[1]
        else:
            face[L["srcn"]] = 0
        for loop in face.loops:
            loop[L["color"]] = col
        if select is not None and (oi == select or hx == select):
            face.select_set(True)
            if oi == select or active is None:
                active = face
    bm.faces.active = active
    bm.select_flush_mode()
    bm.normal_update()   # new faces have no normals until asked: Solid shading drew them black


def write_mesh(obj, select=None, frame=None, fresh=False):
    """Write the object's mesh from its engine. `select` is an open face to
    leave selected (Edit Mode); `frame` puts it in another frame (else it
    stays in the one it stands in now). What was modelled on the mesh with
    Blender's tools is kept (usergeo), unless `fresh` (a new formation).
    Returns the new write id."""
    f = Formation.of(obj)
    st = f.state
    e = st.engine
    prev = read_frame(obj)          # the frame the mesh stands in now
    M = frame if frame is not None else prev
    keep_frame(obj, M)
    items = usergeo.build_plan(st)
    wid = random.randint(1, 2 ** 31 - 2)
    me = obj.data
    if not me.materials:
        me.materials.append(material())
    edit = obj.mode == "EDIT"
    if edit:
        bm = bmesh.from_edit_mesh(me)
    else:
        bm = bmesh.new()
        bm.from_mesh(me)
    L = usergeo.layers(bm, NAMES)
    tol = _tol(e)
    old_plan = None if fresh else f.mesh_plan
    if fresh:
        f.sealed = []
    if frame is None and _modelled(bm, L, M, old_plan, tol):
        usergeo.write_around(bm, L, M, st, old_plan, items, wid, select, f.sealed, tol)
    elif frame is not None and not fresh and old_plan is not None and _modelled(bm, L, prev, old_plan, tol):
        # A new frame (Origin to Plate) with modelled faces: what was modelled
        # moves with the formation.
        K = FR.mul(M, FR.inverse(prev))
        for v in bm.verts:
            v.co = FR.apply(K, tuple(v.co))
        usergeo.write_around(bm, L, M, st, old_plan, items, wid, select, f.sealed, tol)
    else:
        _write_fresh(bm, L, M, items, wid, select)
    if edit:
        bmesh.update_edit_mesh(me, loop_triangles=True, destructive=True)
    else:
        bm.to_mesh(me)
        bm.free()
        attr = me.color_attributes.get(COLOR_ATTR)
        if attr is not None:
            me.color_attributes.active_color = attr
        me.update()
    f.frame = M
    f.plan_items = items
    f.mesh_plan = usergeo.plan_index(items)
    f.dirty = True
    raw = json.dumps(e.serialize(), separators=(",", ":"))
    obj[DOC_PROP] = raw
    obj[WID_PROP] = wid
    f.wid = wid
    f.file(wid, raw)
    if edit:
        refresh(obj)
    return wid


def refresh(obj):
    """Sort the faces of the mesh (in Edit Mode) as usergeo says: which are
    the Form's, which modelled, which modelled seats."""
    f = Formation.of(obj)
    if obj.mode != "EDIT":
        return f
    st = f.state
    if f.mesh_plan is None:
        f.plan_items = usergeo.build_plan(st)
        f.mesh_plan = usergeo.plan_index(f.plan_items)
    bm = bmesh.from_edit_mesh(obj.data)
    L = usergeo.layers(bm, NAMES)
    srt = usergeo.classify(bm, L, f.frame, st, f.mesh_plan, _tol(st.engine))
    f.taken_open = usergeo.taken_open(f.plan_items, srt.intact)
    f.unjoined = usergeo.unjoined(bm, L, st.engine, srt.engine)
    f.dirty = False
    return f


def mesh_wid(obj):
    me = obj.data
    if obj.mode == "EDIT":
        bm = bmesh.from_edit_mesh(me)
        lw = bm.faces.layers.int.get(WID_ATTR)
        if lw is None or not len(bm.faces):
            return None
        bm.faces.ensure_lookup_table()
        return bm.faces[0][lw]
    a = me.attributes.get(WID_ATTR)
    if a is None or not len(a.data):
        return None
    return a.data[0].value


def sync(obj):
    """Bring the engine in line with the mesh after Blender's undo or redo
    (or a file load): the formation filed under the mesh's write id."""
    f = Formation.of(obj)
    f.frame = read_frame(obj)
    w = mesh_wid(obj)
    if w is None or w == f.wid:
        if obj.mode == "EDIT" and not f.dirty:
            # A cheap look for a change Blender did not report: the counts,
            # and where the vertices are in all.
            try:
                bm = bmesh.from_edit_mesh(obj.data)
                sx = sy = sz = 0.0
                for v in bm.verts:
                    c = v.co
                    sx += c.x
                    sy += c.y
                    sz += c.z
                sig = (len(bm.verts), len(bm.faces), round(sx, 4), round(sy, 4), round(sz, 4))
            except (ReferenceError, ValueError):
                sig = None
            if sig != getattr(f, "sig", None):
                f.sig = sig
                f.dirty = True
        if f.dirty or f.mesh_plan is None:
            try:
                refresh(obj)
            except (ReferenceError, ValueError):
                pass
        return f
    raw = f.docs.get(w)
    if raw is None and obj.get(WID_PROP) == w:
        raw = obj.get(DOC_PROP)
    if raw is None:
        return f   # a mesh the Form did not write: left as it is
    sel, mod = f.state.sel_idx, f.state.active_mod_id
    f.state.load_doc(json.loads(raw))
    f.state.active_mod_id = mod
    f.state.status = "Click a face to select"
    f.wid = w
    obj[DOC_PROP] = raw
    obj[WID_PROP] = w
    _ = sel
    # The mesh Blender brought back was written from this formation.
    f.mesh_plan = None
    f.dirty = True
    try:
        refresh(obj)
    except (ReferenceError, ValueError):
        pass
    return f


def picked(obj):
    """The open face the Edit Mode selection names: the active face's, or the
    whole hexagon when all six of its triangles are selected."""
    bm = bmesh.from_edit_mesh(obj.data)
    lo = bm.faces.layers.int.get(OPEN_ATTR)
    lh = bm.faces.layers.int.get(HEX_ATTR)
    if lo is None or lh is None:
        return None
    act = bm.faces.active
    if act is None or not act.select:
        act = next((x for x in bm.faces if x.select), None)
        if act is None:
            return None
    oi, hx = act[lo], act[lh]
    if hx >= 0:
        n = 0
        for x in bm.faces:
            if x[lh] == hx and x.select:
                n += 1
        if n >= 6:
            return hx
    return oi if oi >= 0 else None


def picked_all(obj):
    """Every open face the Edit Mode selection names, in mesh order: each
    selected polygon's open face, and a hexagon whole where all six of its
    triangles are selected (its sub-seats otherwise)."""
    bm = bmesh.from_edit_mesh(obj.data)
    lo = bm.faces.layers.int.get(OPEN_ATTR)
    lh = bm.faces.layers.int.get(HEX_ATTR)
    if lo is None or lh is None:
        return []
    sel = [x for x in bm.faces if x.select]
    per_hex = {}
    for x in sel:
        if x[lh] >= 0:
            per_hex[x[lh]] = per_hex.get(x[lh], 0) + 1
    out, seen = [], set()
    for x in sel:
        hx, oi = x[lh], x[lo]
        i = hx if hx >= 0 and per_hex[hx] >= 6 else oi
        if i >= 0 and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def follow_selection(obj):
    """Sync, then point the store at the faces Edit Mode has selected: the
    active one, and every other one picked with it."""
    f = sync(obj)
    st = f.state
    n = len(st.engine.open_faces)
    try:
        idx = picked(obj)
        picks = picked_all(obj)
    except (ReferenceError, ValueError):
        idx, picks = None, []
    if idx is not None and idx >= n:
        idx = None
    picks = [i for i in picks if i < n]
    if idx is None:
        picks = []
    if idx != st.sel_idx or picks != st.picks:
        st.select(idx, picks)
    return f


def create(context, seed=3, name="PE Formation"):
    me = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, me)
    obj[MARK_PROP] = True
    (context.collection or context.scene.collection).objects.link(obj)
    obj.location = context.scene.cursor.location
    me.materials.append(material())
    f = Formation.of(obj)
    f.state.set_seed_shape(seed)
    f.state.status = "Click a face to select"
    write_mesh(obj, fresh=True)
    return obj
