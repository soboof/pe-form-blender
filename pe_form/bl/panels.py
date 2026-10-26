"""The Form's bars as Blender panels, in the 3D View's sidebar (tab "PE"),
shown while a formation is in Edit Mode.

  Formation   the web's left rail: the module in hand (turning, as in the
              headset), + Add / - Carve, Undo, Edit (Restart), Export, and the
              HUD's count, Height X and hint, and the origin, which is the
              web's base plate (on the selected face or vertices, or back
              on the seed plane), and, for a mesh whose modules' corners
              are not joined (written before the Form joined them), a
              warning with Join Modules; Settings under it
  Parameters  the web's face card: the face (or faces) picked in Edit Mode,
              Select Same Kind, Height X, Conn. Face (whole or sub-seat, and
              which face of the module), Turn, Divide face (a triangle or
              square into N x N smaller faces to grow on at 1/N size), and
              Place (Carve in carve mode), lit in the active colour when it
              can go
  Modules     every module, as a list (name and counts) or compact (a grid of
              icons and codes): a search, then while a face is picked what
              fits it, then Favorites and every family, each a collapsible
              section; what cannot attach is dimmed

Plus Add > PE Formation, a submenu of start faces, and Set Origin in Edit
Mode's right-click menu for a formation.
"""

import bpy

from ..engine.form import (divide_refusal, MODULES, FITS_SHOWN, CAT_ORDER, MATH_ORDER, MATH_ICON, CAT_ICON, face_label,
                           TIPS_EMPTY, TIPS_BUILDING)
from ..engine.geom import get_edge
from ..engine.modules import MODULE_BY_ID
from . import previews, spin
from .formation import edit_formation, follow_selection, settings

CATEGORY = "PE"
COMPACT_COLUMNS = 4


def _wrap(layout, text, width=36, alert=False):
    col = layout.column(align=True)
    col.alert = alert
    line = ""
    for word in text.split(" "):
        t = word if not line else line + " " + word
        if len(t) <= width or not line:
            line = t
        else:
            col.label(text=line)
            line = word
    if line:
        col.label(text=line)


class _FormPanel:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY

    @classmethod
    def poll(cls, context):
        return edit_formation(context) is not None


def _state(context):
    obj = edit_formation(context)
    f = follow_selection(obj)
    return obj, f, f.state


def _mod_icon(st, mid):
    """The module's icon; the module in hand turns."""
    return spin.icon(mid) if mid == st.active_mod_id else previews.icon(mid)


# ── the rail ─────────────────────────────────────────────────────────────────
class PE_PT_form(_FormPanel, bpy.types.Panel):
    bl_label = "Formation"
    bl_idname = "PE_PT_form"
    bl_order = 0

    def draw_header(self, context):
        self.layout.label(text="", icon_value=pe_icon())

    def draw(self, context):
        lay = self.layout
        obj, f, st = _state(context)
        e = st.engine
        m = MODULE_BY_ID.get(st.active_mod_id)
        # The module in hand (the rail's model), turning.
        row = lay.row()
        row.template_icon(icon_value=_mod_icon(st, st.active_mod_id), scale=3.2)
        col = row.column(align=True)
        col.label(text=m.name if m else st.active_mod_id)
        col.label(text=(m.short_code or "") + "  " + (m.sub if m else ""))
        col.label(text="%s · %s" % (m.category, m.math_category) if m else "")
        # + Add / - Carve.
        row = lay.row(align=True)
        row.scale_y = 1.2
        row.operator("pe.build_mode", text="Add", icon="ADD", depress=not st.carve_mode).mode = "add"
        sub = row.row(align=True)
        sub.enabled = st.carve_mode or e.module_count > 0
        sub.operator("pe.build_mode", text="Carve", icon="REMOVE", depress=st.carve_mode).mode = "carve"
        row = lay.row(align=True)
        row.operator("pe.undo", text="Undo", icon="LOOP_BACK")
        row.menu("PE_MT_edit", text="Edit", icon="GREASEPENCIL")
        row.menu("PE_MT_export", text="Export", icon="EXPORT")
        # The HUD: modules, Height X, the hint.
        box = lay.box()
        c = e.carve_count
        count = ("%d mod · %d carved" % (e.module_count - c, c)) if c else "%d mod" % e.module_count
        x = "X %g · %.2f cm" % (round(e.X * 100) / 100, get_edge(e.X))
        if c:
            col = box.column(align=True)
            col.label(text=count, icon_value=pe_icon())
            col.label(text=x, icon="BLANK1")
        else:
            r = box.row()
            r.label(text=count, icon_value=pe_icon())
            r.label(text=x)
        check = f.memo("check", st.place_check)
        tips = TIPS_EMPTY if e.module_count == 0 else TIPS_BUILDING
        hint, err = st.hint(tips[(e.module_count // 3) % len(tips)], check)
        _wrap(box, hint, alert=err)
        if f.unjoined:
            # A mesh written before the Form joined corners: its modules are loose from each other.
            box = lay.box()
            _wrap(box, "The modules are not joined: each keeps its own copy of the corners it meets "
                       "(%d). Subdivision Surface opens them up." % f.unjoined, alert=True)
            box.operator("pe.join_modules", icon="AUTOMERGE_ON")
        # The origin is the base plate it stands on.
        row = lay.row(align=True)
        row.label(text="Origin", icon="OBJECT_ORIGIN")
        row.operator("pe.origin_plate", text="To Plate")
        row.operator("pe.origin_seed", text="Seed Plane")


class PE_PT_settings(_FormPanel, bpy.types.Panel):
    bl_label = "Settings"
    bl_idname = "PE_PT_settings"
    bl_parent_id = "PE_PT_form"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        lay = self.layout
        s = settings()
        wm = context.window_manager
        lay.label(text="Category")
        row = lay.row(align=True)
        for t, label in (("tile", "Tile"), ("math", "Mathematical")):
            row.operator("pe.category", text=label, depress=s.category_type == t).kind = t
        lay.label(text="Start face" + ("" if not _state(context)[2].engine.has_steps else " (before anything is built)"))
        grid = lay.grid_flow(row_major=True, columns=8, even_columns=True, align=True)
        grid.enabled = not _state(context)[2].engine.has_steps
        seed = _state(context)[2].engine.seed_shape
        for n in range(3, 11):
            grid.operator("pe.seed", text="", icon_value=previews.icon("seed_%d" % n), depress=n == seed).n = n
        lay.label(text="Favorites")
        row = lay.row(align=True)
        n = len(s.pinned_mods)
        row.prop(wm, "pe_fav_pick", text="Pick favorites" + (" (%d now)" % n if n else " (none yet)"),
                 icon="SOLO_ON", toggle=True)
        lay.label(text="Modules")
        cats = MATH_ORDER if s.category_type == "math" else CAT_ORDER
        key = "math_category" if s.category_type == "math" else "category"
        col = lay.column(align=True)
        for c in cats:
            if not any(getattr(m, key) == c for m in MODULES):
                continue
            vis = c not in s.hidden_categories
            op = col.operator("pe.toggle_family", text="%s  %s" % (MATH_ICON.get(c) or CAT_ICON.get(c, ""), c),
                              icon="CHECKBOX_HLT" if vis else "CHECKBOX_DEHLT", emboss=False)
            op.key = c
        lay.label(text="Module menu")
        row = lay.row(align=True)
        for v, label, ic in (("list", "List", "LONGDISPLAY"), ("compact", "Compact", "IMGDISPLAY")):
            op = row.operator("pe.setting", text=label, icon=ic, depress=s.module_view == v)
            op.field, op.value = "module_view", v
        lay.label(text="View")
        lay.prop(wm, "pe_show_open", text="Show open faces")
        op = lay.operator("pe.setting", text="Module in hand turns", depress=s.spin,
                          icon="CHECKBOX_HLT" if s.spin else "CHECKBOX_DEHLT", emboss=False)
        op.field, op.value = "spin", "0" if s.spin else "1"


# ── the parameters: the web's face card, less its module picks ──────────────
class PE_PT_face(_FormPanel, bpy.types.Panel):
    bl_label = "Parameters"
    bl_idname = "PE_PT_face"
    bl_order = 1

    def draw(self, context):
        lay = self.layout
        obj, f, st = _state(context)
        e = st.engine
        wm = context.window_manager
        face = e._face(st.sel_idx)
        side = turn = None
        blocked, why = True, ""
        targets = st.targets()
        if face is None:
            _wrap(lay, ("Carve mode: select" if st.carve_mode else "No face picked: select")
                  + " an open face (face select, click). Shift adds more.")
        else:
            side = f.memo("side", st.side_state)
            turn = f.memo("turn", st.turn_state)
            blocked, why = f.memo("check", st.place_check)
            edge = face.verts3[0].distance_to(face.verts3[1])
            m = MODULE_BY_ID.get(st.active_mod_id)
            box = lay.box()
            box.label(text="%s · %.2f cm" % (face_label(turn["face_type"], side["is_rhomb"]).upper(), edge),
                      icon="FACESEL")
            if len(targets) > 1:
                box.label(text="%d faces selected, Place acts on each" % len(targets), icon="STICKY_UVS_LOC")
            box.label(text=(m.name if m else st.active_mod_id) + (" cut out" if st.carve_mode else " on it"),
                      icon_value=_mod_icon(st, st.active_mod_id))
            row = box.row()
            row.operator("pe.select_same", text="Select Same Kind", icon="RESTRICT_SELECT_OFF")

        # Height X: always, since with nothing built it sizes the start face.
        col = lay.column(align=True)
        col.label(text="Height X  (↑ ↓)")
        row = col.row(align=True)
        row.operator("pe.step_x", text="", icon="REMOVE").step = -1
        row.prop(wm, "pe_x", text="X · %.2f cm" % get_edge(e.X))
        row.operator("pe.step_x", text="", icon="ADD").step = 1

        # Conn. Face: which seat of this face, and which face of the module.
        if side is not None and (side["has_split"] or side["has"]):
            col = lay.column(align=True)
            col.label(text="Conn. Face" + ((" · " + side["label"] + "  (Alt ← →)") if side["has"] else ""))
            if side["has_split"]:
                row = col.row(align=True)
                hexwhole = face.face_type == 6
                row.operator("pe.face_seat", text="⬡ Whole", depress=hexwhole).mode = "whole"
                row.operator("pe.face_seat", text="△ Slot", depress=not hexwhole).mode = "tri"
            if side["has"]:
                row = col.row(align=True)
                row.operator("pe.step_side", text="", icon="TRIA_LEFT").step = -1
                grid = row.grid_flow(row_major=True, columns=min(side["count"], 8), even_columns=True, align=True)
                for i in range(side["count"]):
                    op = grid.operator("pe.set_side", text=str(i + 1), depress=i == side["idx"])
                    op.index = i
                row.operator("pe.step_side", text="", icon="TRIA_RIGHT").step = 1

        # Turn.
        if turn is not None:
            col = lay.column(align=True)
            col.label(text="Turn  (← →)")
            row = col.row(align=True)
            row.enabled = turn["on"]
            row.operator("pe.step_turn", text="", icon="TRIA_LEFT").step = -1
            for i, deg in enumerate(turn["options"]):
                op = row.operator("pe.set_turn", text="%d°" % deg, depress=i == turn["idx"])
                op.index = i
            row.operator("pe.step_turn", text="", icon="TRIA_RIGHT").step = 1
            if not turn["on"]:
                col.label(text="Every turn seats it the same here")

        # Face division: the picked triangle or square into N x N.
        show, can, dwhy = st.divide_state()
        if show:
            col = lay.column(align=True)
            col.label(text="Divide face")
            row = col.row(align=True)
            row.operator("pe.step_divide", text="", icon="TRIA_LEFT").step = -1
            n = st.divide_n
            sub = row.row(align=True)
            sub.enabled = can
            sub.operator("pe.divide", text="%d × %d  ·  %.2f cm" % (n, n, face.verts3[0].distance_to(face.verts3[1]) / n),
                         icon="MESH_GRID")
            row.operator("pe.step_divide", text="", icon="TRIA_RIGHT").step = 1
            if dwhy:
                col.label(text=divide_refusal(dwhy), icon="INFO")

        if why:
            _wrap(lay, why, alert=True)
        # Place: lit in the active colour when it can go (Enter does the same).
        verb = "Carve" if st.carve_mode else "Place"
        ready = face is not None and not blocked
        if face is not None and blocked:
            text = verb + " (blocked)"
        elif len(targets) > 1:
            text = "%s on %d faces" % (verb, len(targets))
        else:
            text = verb
        row = lay.row()
        row.scale_y = 1.8
        row.enabled = ready
        row.operator("pe.place", text=text, icon="REMOVE" if st.carve_mode else "CHECKMARK", depress=ready)


# ── the modules: one list, a section per family ─────────────────────────────
def _panel_id(key):
    return "PE_mods_" + "".join(c if c.isalnum() else "_" for c in key)


def _module_rows(body, mods, st, fits, fav, pinned):
    for m in mods:
        row = body.row(align=True)
        row.active = fav or fits is None or m.id in fits
        sp = row.split(factor=0.6, align=True)
        name = (("★ " if m.id in pinned else "☆ ") + m.name) if fav else m.name
        act = (not fav) and m.id == st.active_mod_id
        op = sp.operator("pe.set_module", text=name, icon_value=_mod_icon(st, m.id), depress=act, emboss=act)
        op.mod = m.id
        sp.label(text=m.sub)


def _module_grid(body, mods, st, fits, fav, pinned):
    # Rows of COMPACT_COLUMNS, the last padded, so every tile is one size
    # (a grid_flow stretches a family of one module across the whole row).
    for r in range(0, len(mods), COMPACT_COLUMNS):
        row = body.row(align=True)
        chunk = mods[r:r + COMPACT_COLUMNS]
        for m in chunk:
            cell = row.row(align=True)
            cell.active = fav or fits is None or m.id in fits
            code = m.short_code or m.name[:3]
            if fav:
                code = ("★" if m.id in pinned else "☆") + code
            act = (not fav) and m.id == st.active_mod_id
            op = cell.operator("pe.set_module", text=code, icon_value=_mod_icon(st, m.id), depress=act)
            op.mod = m.id
        for _ in range(COMPACT_COLUMNS - len(chunk)):
            row.label(text="")


def module_sections(st, fit_list, fav, q):
    """The list's sections: (key, icon, title, modules). While a face is
    picked the first is what fits it, in the face card's order (all of them
    when searching, the first FITS_SHOWN otherwise)."""

    def match(m):
        return not q or q in m.name.lower() or q in (m.short_code or "").lower() or q in m.id

    out = []
    if st.sel_idx is not None and not fav:
        shown = [m for m in st.order_fits(fit_list) if match(m)]
        what = "carves here" if st.carve_mode else "fits this face"
        title = ("%s · %d" % (what[0].upper() + what[1:], len(fit_list))) if fit_list else "Nothing " + what
        out.append(("fits", "◎", title, shown if q else shown[:FITS_SHOWN]))
    for g in st.groups():
        out.append((g.key, g.icon, "%s · %d" % (g.name, len(g.mods)), [m for m in g.mods if match(m)]))
    return out


class PE_PT_modules(_FormPanel, bpy.types.Panel):
    bl_label = "Modules"
    bl_idname = "PE_PT_modules"
    bl_order = 2   # under Parameters

    def draw_header_preset(self, context):
        s = settings()
        row = self.layout.row(align=True)
        for v, ic in (("list", "LONGDISPLAY"), ("compact", "IMGDISPLAY")):
            op = row.operator("pe.setting", text="", icon=ic, depress=s.module_view == v)
            op.field, op.value = "module_view", v

    def draw(self, context):
        lay = self.layout
        obj, f, st = _state(context)
        wm = context.window_manager
        fav = wm.pe_fav_pick
        if fav:
            lay.label(text="Click modules to make them favorites", icon="SOLO_ON")
        picked = st.sel_idx is not None
        fit_list = f.memo("fits", st.fits_for_pick) if picked else []
        fits = {m.id for m in fit_list} if picked else None
        lay.prop(wm, "pe_search", text="", icon="VIEWZOOM", placeholder="Search all %d modules" % len(MODULES))
        q = wm.pe_search.strip().lower()
        sections = module_sections(st, fit_list, fav, q)
        if not st.groups():
            lay.label(text="Every family is hidden (Formation > Settings).")
        pinned = settings().pinned_mods
        draw = _module_grid if settings().module_view == "compact" else _module_rows
        shown = False
        for key, icon, title, mods in sections:
            if q and not mods:
                continue
            shown = True
            header, body = lay.panel(_panel_id(key), default_closed=False)
            header.label(text="%s  %s" % (icon, title))
            if body is not None and mods:
                draw(body.column(align=True), mods, st, fits, fav, pinned)
        if q and not shown:
            lay.label(text="No module matches “%s”." % q)


# ── menus ────────────────────────────────────────────────────────────────────
class PE_MT_edit(bpy.types.Menu):
    bl_label = "Edit"
    bl_idname = "PE_MT_edit"

    def draw(self, context):
        lay = self.layout
        for glyph, label in (("⬡", "Fill"), ("◧", "Texture")):
            r = lay.row()
            r.enabled = False
            r.label(text="%s %s (not in the add-on yet)" % (glyph, label))
        # The plate is the origin here (Origin to Plate, or Blender's Set Origin).
        lay.operator("pe.origin_plate", text="⊥ Plate: Origin to Plate")
        lay.operator("pe.join_modules", icon="AUTOMERGE_ON")
        lay.separator()
        lay.operator("pe.restart", text="Restart", icon="FILE_REFRESH")


class PE_MT_export(bpy.types.Menu):
    bl_label = "Export"
    bl_idname = "PE_MT_export"

    def draw(self, context):
        lay = self.layout
        lay.operator("pe.export", text="OBJ").fmt = "obj"
        lay.operator("pe.export", text="STL").fmt = "stl"
        lay.operator("pe.export", text="glTF").fmt = "gltf"
        lay.separator()
        lay.operator("pe.save_formation", text="Save formation (.json)", icon="FILE_TICK")
        lay.operator("pe.open_formation", text="Open formation (.json)", icon="FILE_FOLDER")
        lay.separator()
        lay.operator("pe.convert_to_mesh", text="Convert to plain mesh", icon="MESH_DATA")


class VIEW3D_MT_pe_formation_add(bpy.types.Menu):
    """Add > PE Formation: a new formation, from the start face chosen"""
    bl_label = "PE Formation"
    bl_idname = "VIEW3D_MT_pe_formation_add"

    def draw(self, context):
        lay = self.layout
        lay.operator_context = "EXEC_DEFAULT"
        for n in range(3, 11):
            lay.operator("pe.add_formation", text=SEED_NAMES[n], icon_value=previews.icon("seed_%d" % n)).seed = n
        lay.separator()
        lay.operator_context = "INVOKE_DEFAULT"
        lay.operator("pe.add_samples", text="Sample Formations to Asset Library...", icon="ASSET_MANAGER")


SEED_NAMES = {3: "Triangle", 4: "Square", 5: "Pentagon", 6: "Hexagon", 7: "Heptagon", 8: "Octagon",
              9: "Nonagon", 10: "Decagon"}


def pe_icon():
    """The site's icon (pythagorasengine.com): the add-on's and its formations'."""
    return previews.icon("pe")


def menu_add(self, context):
    self.layout.menu("VIEW3D_MT_pe_formation_add", icon_value=pe_icon())


class PE_MT_origin(bpy.types.Menu):
    """Set Origin, for a formation in Edit Mode: its origin is the base plate it stands on"""
    bl_label = "Set Origin"
    bl_idname = "PE_MT_origin"

    def draw(self, context):
        lay = self.layout
        lay.operator("pe.origin_plate", text="Origin to Plate (selected face or vertices)")
        lay.operator("pe.origin_seed", text="Origin to Seed Plane")


def menu_context(self, context):
    if edit_formation(context) is not None:
        self.layout.menu("PE_MT_origin", icon_value=pe_icon())
        self.layout.separator()


CLASSES = (PE_MT_edit, PE_MT_export, VIEW3D_MT_pe_formation_add, PE_MT_origin,
           PE_PT_form, PE_PT_settings, PE_PT_face, PE_PT_modules)
