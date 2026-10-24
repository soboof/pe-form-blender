"""Blender's undo, redo and file loading, kept in step with the engines."""

import bpy
from bpy.app.handlers import persistent

from .formation import Formation, is_formation, sync


@persistent
def _after_undo(*_):
    # The mesh Blender brought back carries the write id of the formation it
    # was written from; sync reopens that formation.
    for obj in bpy.data.objects:
        if is_formation(obj):
            try:
                sync(obj)
            except (ReferenceError, RuntimeError, ValueError):
                pass


@persistent
def _before_load(*_):
    Formation.forget()


@persistent
def _after_depsgraph(scene, depsgraph):
    # Something was modelled (or moved) on a formation in Edit Mode: its faces
    # are sorted again (bl/usergeo.py) before they are next looked at.
    for u in depsgraph.updates:
        if not u.is_updated_geometry:
            continue
        idb = u.id
        obj = idb if isinstance(idb, bpy.types.Object) else None
        if obj is None:
            continue
        if is_formation(obj) and obj.mode == "EDIT":
            try:
                Formation.of(obj).dirty = True
            except ReferenceError:
                pass


HANDLERS = (("undo_post", _after_undo), ("redo_post", _after_undo), ("load_pre", _before_load),
            ("depsgraph_update_post", _after_depsgraph))


def register():
    for name, fn in HANDLERS:
        lst = getattr(bpy.app.handlers, name)
        if fn not in lst:
            lst.append(fn)


def unregister():
    for name, fn in HANDLERS:
        lst = getattr(bpy.app.handlers, name)
        if fn in lst:
            lst.remove(fn)
