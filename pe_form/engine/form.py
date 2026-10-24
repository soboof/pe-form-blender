"""The Form's store: the state the UI shows and the actions it calls.

A port of pe-web's store/builder.ts (the Form's state and actions),
store/placement.ts (what fits, the side, the turn, Height X, would Place go
through) and the parts of store/settings.ts the Form reads. Pure Python, so
the UI's logic is tested without Blender; the viewport only draws this and
calls into it.
"""

import json
import math
import os

from .builder import BuilderEngine
from .geom import get_edge, split_kind_of, is_rhomb_face
from .modules import MODULES, MODULE_BY_ID

X_MIN = 0.1
X_MAX = 50

# Built-in modules with a connection-face dial, and the face groups each offers.
SUB_OPTIONS = {
    "anticupola": [("cap", "Cap"), ("side", "Side")],
    "gyroelongdipyr": [("cap", "Cap"), ("side", "Side")],
    "snubdisphenoid": [("cap", "Cap"), ("side", "Side")],
    "rhombicprism": [("cap", "Cap ◆"), ("side", "Side □")],
    "goldenrhombicprism": [("cap", "Cap ◆"), ("side", "Side □")],
    "snubdodecahedron": [("edge", "Edge"), ("vertex", "Vertex")],
    "snubcube": [("edge", "Edge"), ("vertex", "Vertex")],
    "triaugprism": [("cap", "Cap"), ("cap2", "Cap 2"), ("side", "Side")],
    "tridiminishedicosahedron": [("cap", "Cap Top"), ("cap2", "Cap Bottom"), ("side", "Side")],
    "triangularcupola": [("side", "Side △"), ("cap", "Cap △")],
    "squarecupola": [("side", "Side □"), ("cap", "Cap □")],
    "pentagonalcupola": [("side", "Side"), ("cap", "Cap")],
    "pentagonalrotunda": [("side", "Side"), ("cap", "Cap")],
}

FACE_NAME = {3: "Triangle", 4: "Square", 5: "Pentagon", 6: "Hexagon", 7: "Heptagon",
             8: "Octagon", 9: "Nonagon", 10: "Decagon"}

CAT_ORDER = ["Triangular", "Square", "Rhombic", "Pentagonal", "Hexagonal", "Heptagonal",
             "Octagonal", "Nonagonal", "Decagonal"]
CAT_ICON = {"Triangular": "△", "Square": "□", "Rhombic": "◆", "Pentagonal": "⬠",
            "Hexagonal": "⬡", "Heptagonal": "7", "Octagonal": "⎔", "Nonagonal": "9", "Decagonal": "⏣"}
# The face-shape families' polygon badge: corner count, and a rhombus or not.
CAT_SHAPE = {"Triangular": (3, False), "Square": (4, False), "Rhombic": (4, True), "Pentagonal": (5, False),
             "Hexagonal": (6, False), "Heptagonal": (7, False), "Octagonal": (8, False),
             "Nonagonal": (9, False), "Decagonal": (10, False)}
MATH_ORDER = ["Platonic", "Archimedean", "Catalan", "Zonohedron", "Johnson", "Prism", "Antiprism"]
MATH_ICON = {"Platonic": "Pl", "Archimedean": "Ar", "Catalan": "Ca", "Zonohedron": "Zo", "Johnson": "Jn",
             "Prism": "Pr", "Antiprism": "Ap"}

SOLID_COLORS = ["#b89a7a", "#d4aa5a", "#4aaa6a"]
VIEW_MODES = ("shaded", "solid", "wire")
FITS_SHOWN = 8

# The idle tips (Viewport.tsx TIPS_EMPTY / TIPS_BUILDING), in Edit Mode's words.
TIPS_EMPTY = [
    "Select the start face, then choose a module",
    "Place (or Enter) puts it on the face",
]
TIPS_BUILDING = [
    "Select an open face to keep building",
    "Enter places the module on the selected face",
    "Undo takes the last module off; Ctrl+Z works too",
    "Save the formation as .json: it opens in the web Form",
]


def face_label(face_type, is_rhomb=False):
    if face_type == 4 and is_rhomb:
        return "Rhombus"
    return FACE_NAME.get(face_type, "%d-gon" % face_type)


def formation_status(e):
    c = e.carve_count
    n = e.module_count - c
    return ("%d module(s) placed · %d carved" % (n, c)) if c else "%d module(s) placed" % n


# BuilderEngine.carve's refusals, in the words the web's viewport hint shows.
CARVE_REFUSAL = {
    "outside": "The module sticks out of the formation",
    "partial contact": "A face of the module would only partly meet a wall",
    "pinch": "The cut would leave material joined along a bare edge",
    "open": "The cut would leave the shell open",
    "face mismatch": "The module's face runs off this face: lower Height X",
    "thin ring": "Too close to the face's own size: match its Height X, or go lower",
    "face not solid": "This face is not a plain wall",
    "nothing to carve": "Place a module first, there is nothing to carve yet",
    "modelled face": "A face you modelled has no formation behind it to carve",
}


DIVIDE_REFUSAL = {
    "no face": "Pick a face to divide",
    "shape": "Only triangles and squares divide into smaller copies of themselves",
    "connector face": "Connector faces keep their size",
    "too fine": "That would be smaller than 1/16 of a module edge",
    "bad count": "Pick 2 to 8 divisions",
    "face not solid": "This face has a hole in it",
    "modelled face": "A face you modelled keeps its own shape",
}


def divide_refusal(reason):
    return DIVIDE_REFUSAL.get(reason, reason or "divide blocked")


def carve_refusal(reason):
    return CARVE_REFUSAL.get(reason, reason or "carve blocked")


def place_refusal(reason):
    if reason == "intersection":
        return "Intersection: placement blocked"
    if reason == "partial contact":
        return "A face of the module would only partly meet a wall: placement blocked"
    return str(reason)


def nearest_turn(options, deg):
    def d(a):
        x = abs((((a - deg) % 360) + 360) % 360)
        return min(x, 360 - x)
    best = 0
    for i, o in enumerate(options):
        if d(o) < d(options[best]):
            best = i
    return best


class ModGroup:
    __slots__ = ("key", "icon", "name", "mods")

    def __init__(self, key, icon, name, mods):
        self.key, self.icon, self.name, self.mods = key, icon, name, mods


def module_groups(category_type, hidden, pinned):
    out = []
    fav = [MODULE_BY_ID[i] for i in pinned if i in MODULE_BY_ID]
    if fav:
        out.append(ModGroup("pinned", "★", "Favorites", fav))
    if category_type == "math":
        for c in MATH_ORDER:
            mods = [m for m in MODULES if m.math_category == c]
            if mods and c not in hidden:
                out.append(ModGroup(c, MATH_ICON[c], c, mods))
    else:
        for c in CAT_ORDER:
            mods = [m for m in MODULES if m.category == c]
            if mods and c not in hidden:
                out.append(ModGroup(c, CAT_ICON.get(c, c[0]), c, mods))
    return out


class Settings:
    """store/settings.ts, the parts the Form reads. Kept in a JSON file the
    Blender layer names (the add-on's own config folder), like localStorage."""

    FIELDS = ("category_type", "hidden_categories", "pinned_mods", "solid_color", "show_view_snap",
              "module_view", "spin")

    def __init__(self, path=None):
        self.path = path
        self.category_type = "tile"
        self.hidden_categories = []
        self.pinned_mods = []
        self.solid_color = SOLID_COLORS[0]
        self.show_view_snap = False
        self.module_view = "list"   # the module list: "list" (rows) or "compact" (a grid)
        self.spin = True            # the current module's icon turns, as in the headset
        self.fav_pick = False  # not kept: a mode, not a setting
        self.load()

    def load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                s = json.load(fh)
        except (OSError, ValueError):
            return
        if s.get("category_type") in ("tile", "math"):
            self.category_type = s["category_type"]
        if isinstance(s.get("hidden_categories"), list):
            self.hidden_categories = [c for c in s["hidden_categories"] if isinstance(c, str)]
        if isinstance(s.get("pinned_mods"), list):
            self.pinned_mods = [m for m in s["pinned_mods"] if m in MODULE_BY_ID]
        if isinstance(s.get("solid_color"), str):
            self.solid_color = s["solid_color"]
        if isinstance(s.get("show_view_snap"), bool):
            self.show_view_snap = s["show_view_snap"]
        if s.get("module_view") in ("list", "compact"):
            self.module_view = s["module_view"]
        if isinstance(s.get("spin"), bool):
            self.spin = s["spin"]

    def set(self, field, value):
        setattr(self, field, value)
        self.save()

    def save(self):
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump({k: getattr(self, k) for k in self.FIELDS}, fh)
        except OSError:
            pass

    def toggle_fav(self, mid):
        if mid in self.pinned_mods:
            self.pinned_mods.remove(mid)
        else:
            self.pinned_mods.append(mid)
        self.save()

    def set_category_type(self, t):
        self.category_type = t
        self.save()

    def toggle_category(self, c):
        if c in self.hidden_categories:
            self.hidden_categories.remove(c)
        else:
            self.hidden_categories.append(c)
        self.save()


class FormState:
    def __init__(self, settings=None):
        self.engine = BuilderEngine()
        self.settings = settings or Settings()
        self.version = 0
        self.active_mod_id = MODULES[0].id
        self.sel_idx = None
        self.orientation = 0
        self.hexa_sub = "cap"
        self.sub_face_mode = "whole"
        self.conn_sel = {}
        self.status = "Click a face to select"
        self.view_mode = "shaded"
        self.recent = []
        self._turn_key = None
        # Carve mode (the − of the + / − pill): Place cuts the module out.
        self.carve_mode = False
        # Every open face picked (Edit Mode allows a multiple selection);
        # sel_idx is the active one, which the parameters and checks describe.
        self.picks = []
        # Face division: how many ways (2 to 8), and whether Height X was
        # last snapped to a small face (picking a full face then puts it back).
        self.divide_n = 2
        self.x_from_face = False
        # Seats modelled in Blender (extruded faces): open faces the engine
        # did not make. They sit at the END of engine.open_faces while the
        # Form looks at them, and are taken out around every change to the
        # engine, so the engine's own faces keep the numbers the web gives
        # them (a save names faces by position).
        self.free_seats = []

    # ── store/builder.ts actions ─────────────────────────────────────────────
    def bump(self):
        self.version += 1

    # ── seats modelled in Blender ────────────────────────────────────────────
    def strip_free(self):
        e = self.engine
        if any(f.free for f in e.open_faces):
            e.open_faces = [f for f in e.open_faces if not f.free]

    def inject_free(self):
        self.strip_free()
        if self.free_seats:
            self.engine.open_faces = self.engine.open_faces + list(self.free_seats)

    def set_free_seats(self, seats):
        """The seats the mesh has now (bl/usergeo.py reads them off it)."""
        key = BuilderEngine.face_key_of
        old = [key(f) for f in self.free_seats]
        sel_key = self.engine.open_face_key(self.sel_idx) if self.sel_idx is not None else None
        self.free_seats = list(seats)
        self.inject_free()
        if [key(f) for f in seats] != old:
            if sel_key is not None:
                i = self.engine.find_open_face(sel_key)
                self.sel_idx = i if i >= 0 else None
                self.picks = [self.sel_idx] if self.sel_idx is not None else []
            self.bump()

    def _engine_op(self, fn):
        """Run fn(index of the picked face, with the modelled seats out of the
        list) and put the seats back. Index -1 for a modelled seat."""
        e = self.engine
        k = e.open_face_key(self.sel_idx) if self.sel_idx is not None else None
        self.strip_free()
        try:
            return fn(e.find_open_face(k) if k is not None else None)
        finally:
            self.inject_free()

    def set_active_mod(self, mid):
        subs = SUB_OPTIONS.get(mid)
        self.active_mod_id = mid
        self.hexa_sub = subs[0][0] if subs else "cap"
        self.conn_sel = {}
        self.sync()

    def set_orientation(self, deg):
        self.orientation = deg

    def set_hexa_sub(self, s):
        self.hexa_sub = s
        self.sync()

    def set_sub_face_mode(self, m):
        self.sub_face_mode = m

    def set_conn_sel_idx(self, gk, idx):
        self.conn_sel = dict(self.conn_sel)
        self.conn_sel[gk] = idx
        self.sync()

    def set_x(self, x):
        e = self.engine
        try:
            v = max(X_MIN, min(X_MAX, float(x)))
        except (TypeError, ValueError):
            return
        if v != v or v == e.X:
            return
        e.X = v
        if not e.has_steps:
            e.init()
            self.sel_idx = None
            self.status = "Click a face to select"
        self.bump()
        self.sync()

    def set_seed_shape(self, n):
        e = self.engine
        if e.has_steps or n not in range(3, 11):
            return
        e.init(n)
        self.sel_idx = None
        self.status = "Click a face to select"
        self.bump()

    # ── Face division ────────────────────────────────────────────────────────
    def set_divide_n(self, n):
        self.divide_n = max(2, min(8, int(round(n))))
        self.sync()

    def divide_state(self):
        """(shown, can, why): Divide is offered on a face the engine would
        divide, or one already at its finest (so the reason shows), never in
        carve mode (formActs.ts divideState)."""
        why = self.engine.divide_refusal(self.sel_idx, self.divide_n)
        face = self.engine._face(self.sel_idx) if self.sel_idx is not None else None
        show = face is not None and not self.carve_mode and (why is None or why == "too fine")
        return show, show and why is None, why

    def divide(self):
        e = self.engine
        if self.sel_idx is None:
            return False
        n = self.divide_n
        r = self._engine_op(lambda i: e.divide(i, n) if i is not None and i >= 0 else
                            type("R", (), {"ok": False, "reason": "modelled face"})())
        if r.ok:
            self.sel_idx = None
            self.picks = []
            self.status = "Face divided %d × %d · pick a small face to grow on" % (n, n)
            self.bump()
            self.sync()
            return True
        self.status = "⚠ " + divide_refusal(r.reason)
        self.sync()
        return False

    def set_carve_mode(self, on):
        on = bool(on)
        if on == self.carve_mode:
            return
        self.carve_mode = on
        self.status = "Carve mode · pick a face to cut into" if on else "Click a face to select"
        if on and self.sel_idx is not None:
            fits = self.fits_for_pick()
            if fits and not any(m.id == self.active_mod_id for m in fits):
                self.set_active_mod(self.order_fits(fits)[0].id)
        self.bump()
        self.sync()

    def targets(self):
        """The faces Place acts on: every face picked, the active one first."""
        if self.sel_idx is None:
            return []
        rest = [i for i in self.picks if i != self.sel_idx and self.engine._face(i) is not None]
        return [self.sel_idx] + rest

    def select(self, idx, picks=None):
        # Picking a small face snaps Height X to the face's own height, so the
        # module shown fits it; picking a full face after that puts X back to
        # that face's height (store/builder.ts select).
        e = self.engine
        f = e._face(idx) if idx is not None else None
        fx = e.face_x(idx) if f is not None else None
        if f is not None and fx is not None and (f.fine is not None or self.x_from_face):
            e.X = max(X_MIN, min(X_MAX, fx))
            self.x_from_face = f.fine is not None
            self.bump()
        self.sel_idx = idx
        self.picks = list(picks) if picks else ([] if idx is None else [idx])
        n = len(self.targets())
        self.status = ("Click a face to select" if idx is None else
                       "Face %d selected" % (idx + 1) if n < 2 else "%d faces selected" % n)
        if idx is not None:
            # FaceCard: picking a face the current module cannot take switches
            # to the best one that can, so Place is ready straight away.
            fits = self.fits_for_pick()
            if fits and not any(m.id == self.active_mod_id for m in fits):
                self.set_active_mod(self.order_fits(fits)[0].id)
        self.sync()

    def _act(self, idx):
        e = self.engine
        kw = dict(hexa_sub=self.hexa_sub, orientation=self.orientation, color=e.color_of(self.active_mod_id),
                  conn_sel=self.conn_sel)
        if self.carve_mode:
            return e.carve(self.active_mod_id, idx, **kw)
        return e.place(self.active_mod_id, idx, **kw)

    def _refused(self, reason):
        return ("⚠ %s: carve blocked" % carve_refusal(reason)) if self.carve_mode else "⚠ " + place_refusal(reason)

    def place(self):
        """Place the module (in carve mode: cut it out) on every face picked.
        Several faces are named by key, not position: each placement reorders
        open_faces, and one may take a face another was going to use."""
        e = self.engine
        targets = self.targets()
        if not targets:
            return False
        keys = [e.open_face_key(i) for i in targets]
        seats = {BuilderEngine.face_key_of(f): f for f in self.free_seats}
        done = refused = gone = 0
        last = None
        used = set()
        for k in keys:
            # The engine's faces alone, and the one modelled seat placed on.
            self.strip_free()
            if k in seats:
                e.open_faces = e.open_faces + [seats[k]]
                i = len(e.open_faces) - 1
            else:
                i = e.find_open_face(k)
            if i < 0:
                gone += 1
                continue
            r = self._act(i)
            if r.ok:
                done += 1
                if k in seats:
                    used.add(k)
            else:
                refused += 1
                last = r.reason
        self.strip_free()
        self.free_seats = [f for f in self.free_seats if BuilderEngine.face_key_of(f) not in used]
        self.inject_free()
        if not done:
            self.status = self._refused(last) if last else "⚠ The picked faces are gone"
            return False
        self.note_recent(self.active_mod_id)
        self.sel_idx = None
        self.picks = []
        st = formation_status(e)
        if len(keys) > 1:
            verb = "Carved through" if self.carve_mode else "Placed on"
            st += " · %s %d of %d faces" % (verb, done, len(keys))
            if refused or gone:
                st += " (%d blocked)" % (refused + gone)
        self.status = st
        self.bump()
        self.sync()
        return True

    def undo(self):
        e = self.engine
        self.strip_free()
        ok = e.can_undo and e.undo()
        self.inject_free()
        if not ok:
            return False
        self.sel_idx = None
        self.status = formation_status(e)
        self.bump()
        self.sync()
        return True

    def reset(self):
        e = self.engine
        self.free_seats = []
        e.init()
        e.extras = {}
        self.sel_idx = None
        self.status = "Click a face to select"
        self.bump()
        self.sync()

    def load_doc(self, doc):
        e = self.engine
        placed, failed = e.load(doc)
        self.inject_free()
        total = len(doc.get("steps", [])) if isinstance(doc, dict) and isinstance(doc.get("steps"), list) else placed
        if failed:
            self.status = "⚠ Stopped at module %d of %d: %s, %d loaded" % (
                failed["index"] + 1, total, failed.get("reason") or failed["modId"], placed)
        else:
            self.status = "Loaded %d module(s)" % placed
        self.sel_idx = None
        self.bump()
        self.sync()
        return placed, failed

    # ── store/placement.ts ───────────────────────────────────────────────────
    def fits_for_pick(self):
        e = self.engine
        if self.sel_idx is None:
            return []
        if self.carve_mode:
            return [m for m in MODULES if e.can_carve_with(m.id, self.sel_idx, self.hexa_sub)]
        return [m for m in MODULES if e.can_connect_to(m.id, self.sel_idx, self.hexa_sub)]

    def side_state(self):
        e = self.engine
        face = e._face(self.sel_idx)
        subs = SUB_OPTIONS.get(self.active_mod_id)
        builtin = e.builtin_dial_info(self.active_mod_id, self.sel_idx) if self.sel_idx is not None else None
        generic = (builtin["key"], builtin["count"]) if builtin else None
        count = generic[1] if generic else (len(subs) if subs else 1)
        if generic:
            idx = min(self.conn_sel.get(generic[0], 0), generic[1] - 1)
        elif subs:
            ids = [s[0] for s in subs]
            idx = max(0, ids.index(self.hexa_sub) if self.hexa_sub in ids else -1)
        else:
            idx = 0
        label = ("%d of %d" % (idx + 1, count)) if generic else (subs[idx][1] if subs else "")
        return {
            "has": count > 1, "count": count, "idx": idx, "label": label,
            "split_kind": split_kind_of(face) if face else None,
            "has_split": bool(face) and split_kind_of(face) is not None,
            "is_rhomb": bool(face) and len(face.verts3) == 4 and is_rhomb_face(face.verts3),
            "generic": generic, "subs": subs,
        }

    def step_side(self, d):
        s = self.side_state()
        if not s["has"]:
            return
        self.set_side((s["idx"] + (1 if d == 1 else s["count"] - 1)) % s["count"])

    def set_side(self, i):
        s = self.side_state()
        if not s["has"]:
            return
        nxt = ((i % s["count"]) + s["count"]) % s["count"]
        if s["generic"]:
            self.set_conn_sel_idx(s["generic"][0], nxt)
        elif s["subs"]:
            self.set_hexa_sub(s["subs"][nxt][0])

    def turn_state(self):
        e = self.engine
        f = e._face(self.sel_idx)
        face_type = f.face_type if f is not None else 3
        options = e.turn_options(self.active_mod_id, self.sel_idx, self.hexa_sub, self.conn_sel)
        idx = nearest_turn(options, self.orientation)
        return {"face_type": face_type, "options": options, "steps": len(options), "idx": idx,
                "deg": options[idx], "on": len(options) > 1}

    def step_turn(self, d):
        t = self.turn_state()
        if not t["on"]:
            return
        nxt = (t["idx"] + (1 if d == 1 else t["steps"] - 1)) % t["steps"]
        self.set_orientation(t["options"][nxt])

    def set_turn(self, i):
        t = self.turn_state()
        if not t["on"]:
            return
        self.set_orientation(t["options"][((i % t["steps"]) + t["steps"]) % t["steps"]])

    def step_x(self, d):
        self.set_x(max(X_MIN, min(X_MAX, self.engine.X + d)))

    def place_check(self, idx=None):
        """(blocked, why) for the active face, or face `idx`."""
        e = self.engine
        idx = self.sel_idx if idx is None else idx
        if idx is None:
            return True, None
        if self.carve_mode:
            f = e._face(idx)
            if not e.history or (f is not None and f.module_idx < 0):
                return True, carve_refusal("nothing to carve")
            p = e.preview_carve(self.active_mod_id, idx, self.hexa_sub, self.orientation, self.conn_sel)
            if p is None:
                return True, "This module has no face that matches this one."
            if not p.ok:
                return True, carve_refusal(p.reason) + "."
            return False, None
        p = e.preview_placement(self.active_mod_id, idx, self.hexa_sub, self.orientation, self.conn_sel)
        if p is None:
            return True, "This module has no side that fits here."
        if p.intersects:
            return True, "It would collide with the formation. Try another side or turn."
        return False, None

    def note_recent(self, mid):
        if mid in self.recent:
            self.recent.remove(mid)
        self.recent.insert(0, mid)
        del self.recent[8:]

    def order_fits(self, fits):
        pinned = self.settings.pinned_mods

        def rank(mid):
            if mid == self.active_mod_id:
                return 0
            if mid in self.recent:
                return 1 + self.recent.index(mid)
            if mid in pinned:
                return 20
            return 30
        return sorted(fits, key=lambda m: rank(m.id))

    def groups(self):
        s = self.settings
        return module_groups(s.category_type, s.hidden_categories, s.pinned_mods)

    def sync(self):
        """FormRail: the turn goes back to 0 whenever the picked face's shape
        or the turns on offer change (a turn means something per face shape)."""
        e = self.engine
        f = e._face(self.sel_idx)
        ft = f.face_type if f is not None else None
        key = (ft, tuple(e.turn_options(self.active_mod_id, self.sel_idx, self.hexa_sub, self.conn_sel)))
        if key != self._turn_key:
            self._turn_key = key
            self.orientation = 0

    # ── HUD words ────────────────────────────────────────────────────────────
    def hud_label(self, blocked=False):
        err = self.status.startswith("⚠")
        if self.carve_mode and self.sel_idx is not None and not err:
            return ("blocked", "err") if blocked else ("carve", "ready")
        if err:
            if "ntersect" in self.status:
                return "blocked", "err"
            if "connectable" in self.status:
                return "no face", "err"
            return "error", "err"
        if self.sel_idx is not None:
            return "face %d" % (self.sel_idx + 1), "ready"
        return "select", ""

    def hint(self, tip, check=None):
        """The hint line. `check` is place_check()'s answer, when the caller has it."""
        if self.carve_mode and not self.status.startswith("⚠"):
            if self.sel_idx is not None:
                blocked, why = check if check is not None else self.place_check()
                if blocked:
                    return (why or "Carve blocked").rstrip("."), True
                return "Carve (or Enter) cuts it out of the formation", False
            if self.engine.module_count == 0:
                return "Place a module first, then carve into it", False
            return "Carve mode · select a face to cut into", False
        if self.status.startswith("⚠"):
            if "ntersect" in self.status:
                return "Blocked by existing geometry: try a different face", True
            if "connectable" in self.status:
                return "No matching connection: pick a different module", True
            return self.status.replace("⚠ ", ""), True
        if self.sel_idx is not None:
            n = len(self.targets())
            if n > 1:
                return "Place (or Enter) puts the module on all %d selected faces" % n, False
            return "Place (or Enter) puts the module on the selected face", False
        return tip, False

    def edge_cm(self):
        return get_edge(self.engine.X)
