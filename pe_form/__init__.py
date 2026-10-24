"""Pythagoras Engine: the Form, inside Blender's Edit Mode.

The web Form's engine (pe-web/src/engine/builder.ts), ported to Python and
held to the web engine's own answers by tests/. Add > PE Formation creates a
formation from the start face chosen and opens it in Edit Mode: select an
open face (or several), choose a module in the sidebar's PE tab, Place (or
Enter); in Carve mode the module is cut out instead. Its origin is the web
Form's base plate: Set Origin (Blender's, or Origin to Plate on a face) says
what it stands on. Skeleton from Modules makes the Theater's skeleton: a joint
on every module, a bone to every module built on it, to pose it; its poses are saved by name, and Save to Asset
Library puts the formation and its poses in the user's asset library. A
formation saves in the web's format, so it goes
back and forth between the add-on and pythagorasengine.com.
"""

try:
    import bpy
except ImportError:  # the engine and its tests run without Blender
    bpy = None

if bpy is not None:
    from bpy.props import BoolProperty, FloatProperty, StringProperty

    from .bl import assets, handlers, ops, overlay, panels, previews, rigging, spin
    from .bl.formation import Formation, edit_formation, write_mesh, follow_selection
    from .engine.form import X_MIN, X_MAX

    _keymaps = []

    def _x_get(self):
        obj = edit_formation(bpy.context)
        return float(Formation.of(obj).state.engine.X) if obj is not None else 5.0

    def _x_set(self, value):
        obj = edit_formation(bpy.context)
        if obj is None:
            return
        st = Formation.of(obj).state
        built = st.engine.has_steps
        st.set_x(value)
        if not built:
            # Nothing built yet: the start face is remade at the new size.
            write_mesh(obj, select=0, fresh=True)
            follow_selection(obj)

    PROPS = {
        "pe_search": StringProperty(name="Search", default="", options={"SKIP_SAVE", "TEXTEDIT_UPDATE"},
                                    description="Search every module by name or code"),
        "pe_fav_pick": BoolProperty(name="Pick Favorites", default=False, options={"SKIP_SAVE"},
                                    description="Click modules in the module list to make them favorites, or take them out"),
        "pe_show_open": BoolProperty(name="Show Open Faces", default=True,
                                     description="Tint every face a module can go on by its corner count"),
        "pe_x": FloatProperty(name="Height X", get=_x_get, set=_x_set, min=X_MIN, max=X_MAX, step=100,
                              precision=2, description="Height X: sizes the next module (edge = X · 2/√3 cm)"),
    }

    def register():
        for c in ops.CLASSES + panels.CLASSES + rigging.CLASSES + assets.CLASSES:
            bpy.utils.register_class(c)
        for k, p in {**PROPS, **rigging.PROPS, **assets.PROPS}.items():
            setattr(bpy.types.WindowManager, k, p)
        bpy.types.VIEW3D_MT_add.append(panels.menu_add)
        bpy.types.VIEW3D_MT_edit_mesh_context_menu.prepend(panels.menu_context)
        previews.register()
        spin.register()
        overlay.register()
        rigging.register()
        handlers.register()
        # Enter and a double click are Place (Carve in carve mode): over the viewport ("Mesh",
        # Edit Mode's own keymap) and over the rest of the 3D View, the
        # sidebar included ("3D View Generic"). The operator only runs while
        # a formation has a face picked; otherwise Enter passes on.
        kc = bpy.context.window_manager.keyconfigs.addon
        if kc is not None:
            for name, space in (("Mesh", "EMPTY"), ("3D View Generic", "VIEW_3D")):
                km = kc.keymaps.new(name=name, space_type=space)
                for key in ("RET", "NUMPAD_ENTER"):
                    _keymaps.append((km, km.keymap_items.new("pe.place", key, "PRESS")))
            # A double click (a double tap on a tablet) on a face: the first
            # click picks it, the second places (carves) there.
            km = kc.keymaps.new(name="Mesh", space_type="EMPTY")
            _keymaps.append((km, km.keymap_items.new("pe.place", "LEFTMOUSE", "DOUBLE_CLICK")))
            # The arrows: Left / Right turn the module, Up / Down change
            # Height X, Alt+Left / Alt+Right the connection face. Each passes
            # the key on (frames, keyframes) when there is nothing to change.
            for key, op, step, alt in (("LEFT_ARROW", "pe.step_turn", -1, False),
                                       ("RIGHT_ARROW", "pe.step_turn", 1, False),
                                       ("UP_ARROW", "pe.step_x", 1, False),
                                       ("DOWN_ARROW", "pe.step_x", -1, False),
                                       ("LEFT_ARROW", "pe.step_side", -1, True),
                                       ("RIGHT_ARROW", "pe.step_side", 1, True)):
                kmi = km.keymap_items.new(op, key, "PRESS", alt=alt, repeat=True)
                kmi.properties.step = step
                _keymaps.append((km, kmi))

    def unregister():
        for km, kmi in _keymaps:
            try:
                km.keymap_items.remove(kmi)
            except (ReferenceError, RuntimeError):
                pass
        _keymaps.clear()
        handlers.unregister()
        rigging.unregister()
        overlay.unregister()
        spin.unregister()
        previews.unregister()
        bpy.types.VIEW3D_MT_add.remove(panels.menu_add)
        bpy.types.VIEW3D_MT_edit_mesh_context_menu.remove(panels.menu_context)
        for k in list(PROPS) + list(rigging.PROPS) + list(assets.PROPS):
            if hasattr(bpy.types.WindowManager, k):
                delattr(bpy.types.WindowManager, k)
        for c in reversed(ops.CLASSES + panels.CLASSES + rigging.CLASSES + assets.CLASSES):
            bpy.utils.unregister_class(c)
        Formation.forget()
