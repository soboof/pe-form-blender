"""The module in hand turns, as the current module does in the headset's dock
(pe-web xr/canvasPanel.ts: a turn in seven seconds).

Its frames are drawn by engine/iconraster.py into preview icons of their own,
one per frame, and the panels show whichever frame the clock is at. Drawing a
frame of a heavy module costs tens of milliseconds, so frames are drawn a few
at a time in a timer, never inside a panel's draw; until a frame exists the
nearest one drawn stands in for it (the still icon before any). The timer
also redraws the sidebar while its PE tab is showing a formation in Edit Mode,
and does nothing else.
"""

import math
import time

import bpy
import bpy.utils.previews
import numpy as np

from ..engine.iconraster import frame, SPIN_FRAMES, SPIN_FPS
from ..engine.modules import MODULE_BY_ID
from . import previews
from .formation import is_formation, settings

SIZE = 64
BUDGET = 0.012   # seconds of frame drawing per tick
KEEP = 4         # modules whose frames are kept

_pcoll = None
_frames = {}      # module id -> {frame index: icon id}
_order = []       # modules, most recently turned last
_t0 = time.monotonic()


def _index():
    return int((time.monotonic() - _t0) * SPIN_FPS) % SPIN_FRAMES


def _draw(mid, k):
    m = MODULE_BY_ID.get(mid)
    img = frame(mid, m.color if m else "#c44444", k * 2 * math.pi / SPIN_FRAMES, size=SIZE, ss=2)
    pv = _pcoll.new("%s#%d" % (mid, k))
    pv.image_size = (SIZE, SIZE)
    pv.image_pixels_float = img[::-1].ravel().tolist()      # bottom row first
    a = img[..., 3:4]
    pre = np.concatenate([img[..., :3] * a, a], axis=2).reshape(SIZE // 2, 2, SIZE // 2, 2, 4).mean(axis=(1, 3))
    sa = pre[..., 3:4]
    small = np.concatenate([np.where(sa > 0, pre[..., :3] / np.maximum(sa, 1e-9), 0), sa], axis=2)
    pv.icon_size = (SIZE // 2, SIZE // 2)
    pv.icon_pixels_float = small[::-1].ravel().tolist()
    _frames.setdefault(mid, {})[k] = pv.icon_id


def _forget(mid):
    for k in _frames.pop(mid, {}):
        try:
            del _pcoll["%s#%d" % (mid, k)]
        except (KeyError, AttributeError):
            pass


def icon(mid):
    """The icon to show for the module in hand: its frame now, while turning."""
    if _pcoll is None or not settings().spin or mid not in MODULE_BY_ID:
        return previews.icon(mid)
    if mid in _order:
        _order.remove(mid)
    _order.append(mid)
    got = _frames.get(mid)
    if not got:
        return previews.icon(mid)
    k = _index()
    if k in got:
        return got[k]
    best = min(got, key=lambda j: (k - j) % SPIN_FRAMES)
    return got[best]


def _showing():
    """UI regions showing the PE tab over a formation in Edit Mode."""
    out = []
    wm = bpy.context.window_manager
    for win in wm.windows if wm else ():
        obj = win.view_layer.objects.active if win.view_layer else None
        if obj is None or obj.mode != "EDIT" or not is_formation(obj):
            continue
        for area in win.screen.areas:
            if area.type != "VIEW_3D" or not area.spaces.active.show_region_ui:
                continue
            for r in area.regions:
                if r.type == "UI" and r.active_panel_category == "PE":
                    out.append(r)
    return out


def _tick():
    try:
        regions = _showing() if settings().spin else []
        if not regions or not _order:
            return 0.5
        mid = _order[-1]
        while len(_order) > KEEP:
            _forget(_order.pop(0))
        got = _frames.get(mid, {})
        if len(got) < SPIN_FRAMES:
            start = time.monotonic()
            k = _index()
            for d in range(SPIN_FRAMES):   # the frames just ahead first
                j = (k + 1 + d) % SPIN_FRAMES
                if j in got:
                    continue
                _draw(mid, j)
                if time.monotonic() - start > BUDGET:
                    break
        for r in regions:
            r.tag_redraw()
    except Exception as ex:  # a timer that raises is dropped for good
        print("PE Form spin:", repr(ex))
    return 1 / SPIN_FPS


def register():
    global _pcoll
    if _pcoll is None:
        _pcoll = bpy.utils.previews.new()
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.5, persistent=True)


def unregister():
    global _pcoll
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    _frames.clear()
    _order.clear()
    if _pcoll is not None:
        bpy.utils.previews.remove(_pcoll)
        _pcoll = None
