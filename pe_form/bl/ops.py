"""Operators: Add > PE Formation, and everything the Form's panels press.

Everything that changes the formation rebuilds the mesh and is an undo step
of its own (bl_options UNDO), so Ctrl+Z in Edit Mode takes it back and
`formation.sync` reopens the formation that mesh was written from. Choosing a
module, a side or a turn changes nothing in the scene and is not a step.
"""

import json
import os

import bpy
import bmesh
from bpy.props import BoolProperty, IntProperty, StringProperty, EnumProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..engine import frame as FR
from ..engine.geom import is_regular_poly, same_mouth_shape
from ..engine.modules import MODULE_BY_ID
from .formation import (Formation, create, edit_formation, follow_selection, is_formation, settings, sync,
                        write_mesh, keep_frame, read_frame, DOC_PROP, WID_PROP, MARK_PROP, OPEN_ATTR, HEX_ATTR,
                        WID_ATTR, FRAME_PROP, BASE_ATTR)


def _redraw(context):
    for area in context.screen.areas if context.screen else ():
        if area.type == "VIEW_3D":
            area.tag_redraw()


class _EditOp:
    @classmethod
    def poll(cls, context):
        return edit_formation(context) is not None


def _picked_state(context):
    obj = edit_formation(context)
    f = follow_selection(obj)
    return obj, f, f.state


# ── Add > PE Formation ───────────────────────────────────────────────────────
class PE_OT_add_formation(bpy.types.Operator):
    """Add a formation of polyhedral modules, starting from this face. It opens in Edit Mode with the face picked"""
    bl_idname = "pe.add_formation"
    bl_label = "Add PE Formation"
    bl_options = {"REGISTER", "UNDO"}

    seed: IntProperty(name="Start Face", default=3, min=3, max=10,
                      description="Corners of the first face: 3 (a triangle) to 10 (a decagon)")

    def execute(self, context):
        if context.mode != "OBJECT" and context.active_object is not None:
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in context.selected_objects:
            o.select_set(False)
        obj = create(context, seed=self.seed)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (False, False, True)
        write_mesh(obj, select=0, fresh=True)        # the start face, picked
        follow_selection(obj)
        sp = context.space_data
        if sp is not None and sp.type == "VIEW_3D":
            sp.show_region_ui = True
            # Solid shading by the modules' colours (the colour attribute),
            # the way the web Form shows a formation.
            if sp.shading.type == "SOLID" and sp.shading.color_type == "MATERIAL":
                sp.shading.color_type = "VERTEX"
        self.report({"INFO"}, "PE Formation: its panels are in the sidebar's PE tab (N)")
        return {"FINISHED"}


# ── placing ──────────────────────────────────────────────────────────────────
# The lists Place's properties choose from, as they were for the face it was
# done on (the redo panel draws after the face is taken, so they cannot be
# worked out then). Kept alive here: Blender keeps no copy of a dynamic
# enum's strings.
_CHOICES = {"mods": [], "faces": [], "turns": []}
_MOD_NUM = {mid: i + 1 for i, mid in enumerate(sorted(MODULE_BY_ID))}
_NONE = [("0", "None", "", 0)]


def _mod_items(self, context):
    return _CHOICES["mods"] or [("", "None", "", 0, 0)]


def _face_items(self, context):
    return _CHOICES["faces"] or _NONE


def _turn_items(self, context):
    return _CHOICES["turns"] or _NONE


def _remember_choices(st):
    """The modules that fit the picked face (the one in hand first), its
    connection faces and its turns, for Place's redo panel."""
    from . import previews
    fits = st.fits_for_pick()
    ids = [m.id for m in st.order_fits(fits)] if fits else []
    if st.active_mod_id not in ids:
        ids.insert(0, st.active_mod_id)
    mods = []
    for mid in ids:
        m = MODULE_BY_ID.get(mid)
        name = m.name if m else mid
        tip = ("%s (%s): %s" % (m.name, m.short_code, m.sub)) if m and m.short_code else name
        # A dynamic enum keeps its choice by number: each module its own,
        # whatever order the list is in, or a re-sorted list moves the choice.
        mods.append((mid, name, tip, previews.icon(mid), _MOD_NUM.get(mid, 0)))
    _CHOICES["mods"] = mods
    side = st.side_state()
    if side["generic"]:
        faces = [(str(i), "Face %d of %d" % (i + 1, side["count"]), "") for i in range(side["count"])]
    elif side["subs"]:
        faces = [(str(i), lab, "") for i, (_, lab) in enumerate(side["subs"])]
    else:
        faces = [("0", "The one that fits", "")]
    _CHOICES["faces"] = faces
    turn = st.turn_state()
    _CHOICES["turns"] = [(str(i), "%g°" % d, "") for i, d in enumerate(turn["options"])] or [("0", "0°", "")]
    return side, turn


class PE_OT_place(_EditOp, bpy.types.Operator):
    """Place the module on every selected face (in Carve mode: cut it out of the formation there). Enter, or double-click a face. After it, the Adjust Last Operation panel changes the module, Height X, connection face and turn"""
    bl_idname = "pe.place"
    bl_label = "Place"
    bl_options = {"REGISTER", "UNDO"}

    module: EnumProperty(name="Module", items=_mod_items, options={"SKIP_SAVE"},
                         description="The module placed: any that fits this face")
    height: bpy.props.FloatProperty(name="Height X", min=0.1, max=50.0, precision=2, step=100,
                                    options={"SKIP_SAVE"}, description="Sizes the module (edge = X · 2/√3 cm)")
    face: EnumProperty(name="Conn. Face", items=_face_items, options={"SKIP_SAVE"},
                       description="Which face of the module attaches")
    turn: EnumProperty(name="Turn", items=_turn_items, options={"SKIP_SAVE"},
                       description="How it is turned on the face")
    carve: BoolProperty(name="Carve", default=False, options={"SKIP_SAVE"},
                        description="Cut the module out of the formation instead of adding it")
    ready: BoolProperty(default=False, options={"HIDDEN", "SKIP_SAVE"})

    # Any formation in Edit Mode: Blender hides the Adjust Last Operation
    # panel when the operator's poll fails afterwards, and the placement has
    # just used the picked face. With no face picked it cancels, and Enter or
    # the double click goes on to Blender.
    @classmethod
    def poll(cls, context):
        return edit_formation(context) is not None

    @classmethod
    def description(cls, context, properties):
        obj = edit_formation(context)
        st = follow_selection(obj).state if obj is not None else None
        if st is not None and st.carve_mode:
            return "Carve: cut the module out of the formation through every selected face (Enter, or double-click)"
        return "Place the module on every selected face (Enter, or double-click a face)"

    def _apply(self, st):
        """The properties onto the Form, in the order each depends on the
        last: carve, module, connection face, Height X, turn. The first time
        they are the Form's own; in the redo panel they are the user's."""
        e = st.engine
        if not self.ready:
            self.carve = st.carve_mode
            self.height = e.X
            _remember_choices(st)
            self.module = st.active_mod_id
            side, turn = _remember_choices(st)
            self.face = str(side["idx"])
            self.turn = str(turn["idx"])
            self.ready = True
            return
        st.set_carve_mode(self.carve)
        _remember_choices(st)
        if self.module and self.module != st.active_mod_id and self.module in MODULE_BY_ID:
            st.set_active_mod(self.module)
        side, _ = _remember_choices(st)
        try:
            st.set_side(int(self.face))
        except ValueError:
            pass
        e.X = max(0.1, min(50.0, self.height))
        side, turn = _remember_choices(st)
        try:
            st.set_turn(int(self.turn))
        except ValueError:
            pass
        side, turn = _remember_choices(st)
        # Shown as they now are (a new module may have fewer faces or turns).
        self.face = str(side["idx"])
        self.turn = str(turn["idx"])

    def draw(self, context):
        lay = self.layout
        lay.use_property_split = True
        lay.use_property_decorate = False
        lay.prop(self, "module")
        lay.prop(self, "height")
        if len(_CHOICES["faces"]) > 1:
            lay.prop(self, "face")
        if len(_CHOICES["turns"]) > 1:
            lay.row(align=True).prop(self, "turn", expand=True)
        lay.prop(self, "carve")

    def execute(self, context):
        obj, f, st = _picked_state(context)
        if st.sel_idx is None:
            return {"CANCELLED"}
        self._apply(st)
        if st.place():
            write_mesh(obj, select=None)
            follow_selection(obj)
            _redraw(context)
            self.report({"INFO"}, st.status)
            return {"FINISHED"}
        self.report({"WARNING"}, st.status.replace("⚠ ", ""))
        return {"CANCELLED"}


def same_kind(e, a, b):
    """Are open faces a and b the same kind of seat: the same shape (corner
    count, rhombus or not, its proportions) at the same size, and a
    hexagon's slot only with a slot?"""
    if a.face_type != b.face_type or bool(a.hex_sub) != bool(b.hex_sub) or len(a.verts3) != len(b.verts3):
        return False
    ea, eb = a.verts3[0].distance_to(a.verts3[1]), b.verts3[0].distance_to(b.verts3[1])
    if abs(ea - eb) > max(ea, eb) * 0.02:
        return False
    if is_regular_poly(a.verts3) and is_regular_poly(b.verts3):
        return True
    return same_mouth_shape(a.verts3, b.verts3)


class PE_OT_select_same(_EditOp, bpy.types.Operator):
    """Select every open face of the same kind as the active one (the same shape and size), so Place puts the module on all of them"""
    bl_idname = "pe.select_same"
    bl_label = "Select Same Kind"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = edit_formation(context)
        if obj is None:
            return False
        st = follow_selection(obj).state
        f = st.engine._face(st.sel_idx)
        return f is not None and not f.hex_sub

    def execute(self, context):
        obj, f, st = _picked_state(context)
        e = st.engine
        ref = e._face(st.sel_idx)
        want = {i for i, o in enumerate(e.open_faces) if o is ref or (not o.hex_sub and same_kind(e, ref, o))}
        bm = bmesh.from_edit_mesh(obj.data)
        lo, lh = bm.faces.layers.int.get(OPEN_ATTR), bm.faces.layers.int.get(HEX_ATTR)
        act = bm.faces.active
        for x in bm.faces:
            x.select_set(x[lo] in want or (x[lh] >= 0 and x[lh] in want))
        bm.faces.active = act
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        follow_selection(obj)
        self.report({"INFO"}, "%d faces of this kind selected" % len(want))
        _redraw(context)
        return {"FINISHED"}


class PE_OT_undo(_EditOp, bpy.types.Operator):
    """Take the last module off"""
    bl_idname = "pe.undo"
    bl_label = "Undo Module"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj, f, st = _picked_state(context)
        if not st.undo():
            return {"CANCELLED"}
        write_mesh(obj, select=None)
        _redraw(context)
        return {"FINISHED"}


class PE_OT_restart(_EditOp, bpy.types.Operator):
    """Restart: empty the formation and start again from its start face"""
    bl_idname = "pe.restart"
    bl_label = "Restart the formation?"
    bl_options = {"REGISTER", "UNDO"}

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        obj, f, st = _picked_state(context)
        st.reset()
        write_mesh(obj, select=0, fresh=True)
        follow_selection(obj)
        _redraw(context)
        return {"FINISHED"}


class PE_OT_build_mode(_EditOp, bpy.types.Operator):
    """Add puts modules on the formation; Carve cuts them out of it, leaving a cavity in their shape"""
    bl_idname = "pe.build_mode"
    bl_label = "Build Mode"
    bl_options = {"INTERNAL"}

    mode: StringProperty(default="add")

    @classmethod
    def description(cls, context, properties):
        if properties.mode == "carve":
            return ("Carve: Place cuts the module out of the formation through the picked face, "
                    "leaving a cavity in its shape (only where it lies wholly inside the material)")
        return "Add: Place puts the module on the picked face"

    def execute(self, context):
        obj, f, st = _picked_state(context)
        st.set_carve_mode(self.mode == "carve")
        _redraw(context)
        return {"FINISHED"}


class PE_OT_setting(_EditOp, bpy.types.Operator):
    """Change how the Form's panels show the modules"""
    bl_idname = "pe.setting"
    bl_label = "Setting"
    bl_options = {"INTERNAL"}

    field: StringProperty()
    value: StringProperty()

    @classmethod
    def description(cls, context, properties):
        return {
            ("module_view", "list"): "The modules as a list: each one's name and counts",
            ("module_view", "compact"): "The modules compact: a grid of icons and codes, the name on hover",
            ("spin", "1"): "The module in hand turns, as in the headset",
            ("spin", "0"): "The module in hand stands still",
        }.get((properties.field, properties.value), cls.__doc__)

    def execute(self, context):
        if self.field == "module_view" and self.value in ("list", "compact"):
            settings().set("module_view", self.value)
        elif self.field == "spin":
            settings().set("spin", self.value == "1")
        else:
            return {"CANCELLED"}
        _redraw(context)
        return {"FINISHED"}


# ── the face card's choices (no undo step: nothing in the scene changes) ─────
class PE_OT_set_module(_EditOp, bpy.types.Operator):
    """Use this module (while picking favorites: make it one, or take it out)"""
    bl_idname = "pe.set_module"
    bl_label = "Module"
    bl_options = {"INTERNAL"}

    mod: StringProperty()
    clear_search: BoolProperty(default=False, options={"SKIP_SAVE"})

    @classmethod
    def description(cls, context, properties):
        m = MODULE_BY_ID.get(properties.mod)
        if m is None:
            return cls.__doc__
        what = "%s (%s): %s" % (m.name, m.short_code, m.sub) if m.short_code else "%s: %s" % (m.name, m.sub)
        if context.window_manager.pe_fav_pick:
            return what + ". Click to make it a favorite, or take it out"
        return what + ". Click to use it"

    def execute(self, context):
        obj, f, st = _picked_state(context)
        wm = context.window_manager
        if wm.pe_fav_pick:
            settings().toggle_fav(self.mod)
        else:
            st.set_active_mod(self.mod)
            st.note_recent(self.mod)
        if self.clear_search:
            wm.pe_search = ""
        _redraw(context)
        return {"FINISHED"}


class PE_OT_set_side(_EditOp, bpy.types.Operator):
    """Which face of the module attaches"""
    bl_idname = "pe.set_side"
    bl_label = "Conn. Face"
    bl_options = {"INTERNAL"}

    index: IntProperty()

    def execute(self, context):
        obj, f, st = _picked_state(context)
        st.set_side(self.index)
        _redraw(context)
        return {"FINISHED"}


class PE_OT_step_side(_EditOp, bpy.types.Operator):
    """The next or previous face of the module (Alt+Left / Alt+Right; Tab in the web Form)"""
    bl_idname = "pe.step_side"
    bl_label = "Step Conn. Face"
    bl_options = {"INTERNAL"}

    step: IntProperty(default=1)

    @classmethod
    def poll(cls, context):
        # Nothing to step: the key goes on to Blender.
        obj = edit_formation(context)
        st = follow_selection(obj).state if obj is not None else None
        return st is not None and st.sel_idx is not None and st.side_state()["has"]

    def execute(self, context):
        obj, f, st = _picked_state(context)
        st.step_side(1 if self.step > 0 else -1)
        _redraw(context)
        return {"FINISHED"}


class PE_OT_set_turn(_EditOp, bpy.types.Operator):
    """Turn the module to this seat"""
    bl_idname = "pe.set_turn"
    bl_label = "Turn"
    bl_options = {"INTERNAL"}

    index: IntProperty()

    def execute(self, context):
        obj, f, st = _picked_state(context)
        st.set_turn(self.index)
        _redraw(context)
        return {"FINISHED"}


class PE_OT_step_turn(_EditOp, bpy.types.Operator):
    """The next or previous turn (Left / Right; R, Shift+R in the web Form)"""
    bl_idname = "pe.step_turn"
    bl_label = "Step Turn"
    bl_options = {"INTERNAL"}

    step: IntProperty(default=1)

    @classmethod
    def poll(cls, context):
        obj = edit_formation(context)
        st = follow_selection(obj).state if obj is not None else None
        return st is not None and st.sel_idx is not None and st.turn_state()["on"]

    def execute(self, context):
        obj, f, st = _picked_state(context)
        st.step_turn(1 if self.step > 0 else -1)
        _redraw(context)
        return {"FINISHED"}


class PE_OT_seed(_EditOp, bpy.types.Operator):
    """Start from this face (before anything is built): 3 corners (a triangle) to 10 (a decagon)"""
    bl_idname = "pe.seed"
    bl_label = "Start Face"
    bl_options = {"REGISTER", "UNDO"}

    n: IntProperty(default=3, min=3, max=10)

    @classmethod
    def poll(cls, context):
        obj = edit_formation(context)
        return obj is not None and not Formation.of(obj).state.engine.has_steps

    def execute(self, context):
        obj, f, st = _picked_state(context)
        st.set_seed_shape(self.n)
        write_mesh(obj, select=0, fresh=True)
        follow_selection(obj)
        _redraw(context)
        return {"FINISHED"}


class PE_OT_divide(_EditOp, bpy.types.Operator):
    """Divide the picked face (a triangle or a square) into N × N smaller copies of itself: a module grown on one comes out 1/N the size"""
    bl_idname = "pe.divide"
    bl_label = "Divide Face"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = edit_formation(context)
        return obj is not None and follow_selection(obj).state.divide_state()[1]

    def execute(self, context):
        obj, f, st = _picked_state(context)
        if st.divide():
            write_mesh(obj, select=None)
            follow_selection(obj)
            _redraw(context)
            self.report({"INFO"}, st.status)
            return {"FINISHED"}
        self.report({"WARNING"}, st.status.replace("⚠ ", ""))
        return {"CANCELLED"}


class PE_OT_step_divide(_EditOp, bpy.types.Operator):
    """How many ways to divide the face: 2 to 8"""
    bl_idname = "pe.step_divide"
    bl_label = "Divisions"

    step: IntProperty(default=1)

    def execute(self, context):
        obj, f, st = _picked_state(context)
        st.set_divide_n(st.divide_n + (1 if self.step > 0 else -1))
        _redraw(context)
        return {"FINISHED"}


class PE_OT_step_x(_EditOp, bpy.types.Operator):
    """Height X down or up by 1 (Down / Up; sizes the next module; with nothing built, the start face too)"""
    bl_idname = "pe.step_x"
    bl_label = "Height X"
    bl_options = {"UNDO", "INTERNAL"}

    step: IntProperty(default=1)

    def execute(self, context):
        obj, f, st = _picked_state(context)
        built = st.engine.has_steps
        st.step_x(1 if self.step > 0 else -1)
        if not built:
            write_mesh(obj, select=0, fresh=True)
            follow_selection(obj)
        _redraw(context)
        return {"FINISHED"}


class PE_OT_face_seat(_EditOp, bpy.types.Operator):
    """A hexagon whole, or one of its six triangle slots: selects the six, or only the one"""
    bl_idname = "pe.face_seat"
    bl_label = "Seat"
    bl_options = {"REGISTER", "UNDO"}

    mode: StringProperty(default="whole")

    def execute(self, context):
        obj = edit_formation(context)
        bm = bmesh.from_edit_mesh(obj.data)
        lh = bm.faces.layers.int.get(HEX_ATTR)
        act = bm.faces.active
        if lh is None or act is None or act[lh] < 0:
            return {"CANCELLED"}
        hx = act[lh]
        for x in bm.faces:
            if x[lh] == hx:
                x.select_set(self.mode == "whole" or x == act)
        if self.mode == "tri":
            act.select_set(True)
        bm.select_flush_mode()
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        follow_selection(obj)
        _redraw(context)
        return {"FINISHED"}


# ── the dock and settings ────────────────────────────────────────────────────
class PE_OT_category(_EditOp, bpy.types.Operator):
    """Group the modules by face shape (Tile) or by kind of solid (Mathematical)"""
    bl_idname = "pe.category"
    bl_label = "Category"
    bl_options = {"INTERNAL"}

    kind: StringProperty(default="tile")

    def execute(self, context):
        settings().set_category_type(self.kind)
        return {"FINISHED"}


class PE_OT_toggle_family(_EditOp, bpy.types.Operator):
    """Show or hide this family in the module list"""
    bl_idname = "pe.toggle_family"
    bl_label = "Show Family"
    bl_options = {"INTERNAL"}

    key: StringProperty()

    def execute(self, context):
        settings().toggle_category(self.key)
        return {"FINISHED"}


# ── files ────────────────────────────────────────────────────────────────────
def _target(context):
    obj = context.edit_object or context.active_object
    return obj if is_formation(obj) else None


class PE_OT_export(bpy.types.Operator):
    """Export the formation with Blender's own exporter, as it stands in the scene (its origin is its base plate)"""
    bl_idname = "pe.export"
    bl_label = "Export Formation"

    fmt: EnumProperty(items=[("obj", "OBJ", ""), ("stl", "STL", ""), ("gltf", "glTF", "")])

    @classmethod
    def poll(cls, context):
        return _target(context) is not None

    def execute(self, context):
        obj = _target(context)
        if obj.mode == "EDIT":
            obj.update_from_editmode()
        f = Formation.of(obj)
        base = "pythagoras_m%d" % f.state.engine.module_count
        folder = bpy.path.abspath("//") or os.path.expanduser("~")
        for o in context.view_layer.objects:
            if o != obj:
                o.select_set(False)
        obj.select_set(True)
        try:
            if self.fmt == "obj":
                bpy.ops.wm.obj_export("INVOKE_DEFAULT", filepath=os.path.join(folder, base + ".obj"),
                                      export_selected_objects=True, forward_axis="NEGATIVE_Z", up_axis="Y")
            elif self.fmt == "stl":
                bpy.ops.wm.stl_export("INVOKE_DEFAULT", filepath=os.path.join(folder, base + ".stl"),
                                      export_selected_objects=True)
            else:
                bpy.ops.export_scene.gltf("INVOKE_DEFAULT", filepath=os.path.join(folder, base + ".gltf"),
                                          use_selection=True, export_format="GLTF_SEPARATE")
        except (AttributeError, RuntimeError) as e:
            self.report({"ERROR"}, "%s export is not available: %s" % (self.fmt.upper(), e))
            return {"CANCELLED"}
        return {"FINISHED"}


class PE_OT_save_formation(bpy.types.Operator, ExportHelper):
    """Save the formation in the web Form's format (.json): it opens at pythagorasengine.com"""
    bl_idname = "pe.save_formation"
    bl_label = "Save Formation"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return _target(context) is not None

    def invoke(self, context, event):
        f = Formation.of(_target(context))
        self.filepath = "pythagoras_m%d.json" % f.state.engine.module_count
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        obj = _target(context)
        if obj is None:
            return {"CANCELLED"}
        f = sync(obj)
        keep_frame(obj, read_frame(obj))     # the origin as it is now, as the save's ground
        doc = f.state.engine.serialize()
        with open(self.filepath, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, separators=(",", ":"))
        self.report({"INFO"}, "Saved %d module(s)" % len(doc["steps"]))
        return {"FINISHED"}


class PE_OT_open_formation(bpy.types.Operator, ImportHelper):
    """Open a formation saved by the web Form or the add-on (.json)"""
    bl_idname = "pe.open_formation"
    bl_label = "Open Formation"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        try:
            with open(self.filepath, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as e:
            self.report({"ERROR"}, "Not a formation: %s" % e)
            return {"CANCELLED"}
        # A record from the web's account list wraps the formation in "data".
        if isinstance(doc, dict) and "steps" not in doc and isinstance(doc.get("data"), dict):
            doc = doc["data"]
        if not isinstance(doc, dict) or not isinstance(doc.get("steps"), list):
            self.report({"ERROR"}, "Not a formation: no steps")
            return {"CANCELLED"}
        obj = _target(context)
        if obj is None:
            if context.mode != "OBJECT" and context.active_object is not None:
                bpy.ops.object.mode_set(mode="OBJECT")
            obj = create(context, seed=doc.get("seedShape", 3) if isinstance(doc.get("seedShape"), int) else 3,
                         name=os.path.splitext(os.path.basename(self.filepath))[0] or "PE Formation")
            for o in context.selected_objects:
                o.select_set(False)
            obj.select_set(True)
            context.view_layer.objects.active = obj
        f = Formation.of(obj)
        placed, failed = f.state.load_doc(doc)
        # Standing on the base plate it was saved with: its centre the origin.
        write_mesh(obj, select=None, fresh=True, frame=FR.from_ground(doc.get("ground")))
        if failed:
            self.report({"WARNING"}, "Opened %d module(s); stopped at step %d: %s" % (
                placed, failed["index"] + 1, failed.get("reason") or failed["modId"]))
        else:
            self.report({"INFO"}, "Opened %d module(s)" % placed)
        _redraw(context)
        return {"FINISHED"}


class PE_OT_convert_to_mesh(bpy.types.Operator):
    """Make the formation a plain mesh: it keeps its shape and colours, and the Form lets go of it"""
    bl_idname = "pe.convert_to_mesh"
    bl_label = "Convert Formation to Mesh"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _target(context) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        obj = _target(context)
        was_edit = obj.mode == "EDIT"
        if was_edit:
            bpy.ops.object.mode_set(mode="OBJECT")
        me = obj.data
        for name in (OPEN_ATTR, HEX_ATTR, WID_ATTR, "pe_role", "pe_src", "pe_srcn", "pe_mod"):
            a = me.attributes.get(name)
            if a is not None:
                me.attributes.remove(a)
        a = me.attributes.get(BASE_ATTR)
        if a is not None:
            me.attributes.remove(a)
        for p in (MARK_PROP, DOC_PROP, WID_PROP, FRAME_PROP):
            if p in obj:
                del obj[p]
        Formation.forget(obj)
        if was_edit:
            bpy.ops.object.mode_set(mode="EDIT")
        return {"FINISHED"}


# ── the origin is the base plate ─────────────────────────────────────────────
def _move_frame(obj, M):
    """Stand the mesh in frame M. The object's own transform is left as it
    is, so the formation goes to stand on its plate at the object's origin,
    as the web Form's view stands it there. (Keeping it in place would move
    the object as well, and Edit Mode's undo, which keeps the mesh, would
    then bring the mesh back and leave the object moved.)"""
    st = Formation.of(obj).state
    write_mesh(obj, select=st.sel_idx, frame=M)
    follow_selection(obj)


class PE_OT_origin_plate(_EditOp, bpy.types.Operator):
    """Set the origin as the base plate the formation stands on: on the selected face (its centre the origin, its outside facing down, X along one of its edges), or on the plane through 3 or more selected vertices. The formation then stands on it at the object's origin"""
    bl_idname = "pe.origin_plate"
    bl_label = "Origin to Plate"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = edit_formation(context)
        follow_selection(obj)
        bm = bmesh.from_edit_mesh(obj.data)
        faces = [x for x in bm.faces if x.select]
        if len(faces) == 1:
            pts, face = [tuple(v.co) for v in faces[0].verts], True
        else:
            pts, face = [tuple(v.co) for v in bm.verts if v.select], False
        if len(pts) < 3:
            self.report({"WARNING"}, "Select a face, or 3 or more vertices, to stand it on")
            return {"CANCELLED"}
        n = len(bm.verts)
        inside = tuple(sum(v.co[k] for v in bm.verts) / n for k in range(3))
        K = FR.plate(pts, inside, face)
        if K is None:
            self.report({"WARNING"}, "Those points are on one line: they name no plane")
            return {"CANCELLED"}
        _move_frame(obj, FR.mul(K, read_frame(obj)))
        self.report({"INFO"}, "Origin on the plate: the formation stands on it")
        return {"FINISHED"}


class PE_OT_origin_seed(_EditOp, bpy.types.Operator):
    """Set the origin back on the seed plane: the formation stands as it was built, on its first face's plane"""
    bl_idname = "pe.origin_seed"
    bl_label = "Origin to Seed Plane"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = edit_formation(context)
        follow_selection(obj)
        s = FR.scale_of(read_frame(obj))
        _move_frame(obj, [[s if i == j and i < 3 else (1.0 if i == j else 0.0) for j in range(4)] for i in range(4)])
        return {"FINISHED"}


CLASSES = (PE_OT_add_formation, PE_OT_origin_plate, PE_OT_origin_seed, PE_OT_place, PE_OT_undo, PE_OT_restart, PE_OT_build_mode, PE_OT_set_module,
           PE_OT_set_side, PE_OT_step_side, PE_OT_divide, PE_OT_step_divide, PE_OT_seed, PE_OT_set_turn, PE_OT_step_turn, PE_OT_step_x, PE_OT_face_seat,
           PE_OT_select_same, PE_OT_setting, PE_OT_category, PE_OT_toggle_family, PE_OT_export, PE_OT_save_formation,
           PE_OT_open_formation, PE_OT_convert_to_mesh)
