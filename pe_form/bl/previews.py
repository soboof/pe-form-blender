"""The module icons (pe_form/icons/*.png, scripts/make_icons.py) as Blender
preview icons, for the panels' buttons."""

import os

import bpy
import bpy.utils.previews

ICON_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons")
_pcoll = None


def register():
    global _pcoll
    if _pcoll is not None:
        return
    _pcoll = bpy.utils.previews.new()
    if os.path.isdir(ICON_DIR):
        for name in os.listdir(ICON_DIR):
            if name.endswith(".png"):
                _pcoll.load(name[:-4], os.path.join(ICON_DIR, name), "IMAGE")


def unregister():
    global _pcoll
    if _pcoll is not None:
        bpy.utils.previews.remove(_pcoll)
        _pcoll = None


def icon(mid):
    if _pcoll is None or mid not in _pcoll:
        return 0
    return _pcoll[mid].icon_id
