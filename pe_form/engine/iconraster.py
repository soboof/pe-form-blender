"""Module icons drawn the way the headset's dock draws its models
(pe-web xr/canvasPanel.ts `model`, xr/moduleModel.ts `unitPositions`): the
solid centred on its bounding box and scaled to radius one, flat shaded in its
colour, its polygon edges in black at 0.35, turned (0.5, 0.6) rad about X and
Y to show three faces. The current module turns about its own Y, a turn in
seven seconds (ICON_SPIN), so `frame(mid, color, spin)` draws it at any turn.

Pure numpy, no bpy: scripts/make_icons.py writes the still icons with it, and
bl/spin.py the turning one. Each polygon and edge is rasterized only over its
own bounding box, so a frame costs milliseconds, not the whole canvas per
triangle.
"""

import math

import numpy as np

from .modules import MOD_DATA

ICON_TILT = (0.5, 0.6)
ICON_SPIN = 0.0009  # rad / ms
SPIN_FPS = 10
SPIN_FRAMES = int(round(2 * math.pi / (ICON_SPIN * 1000) * SPIN_FPS))  # one turn

LIGHT = np.array([0.35, 0.75, 0.56])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
EDGE_ALPHA = 0.35

_geo = {}


def _geometry(mid):
    """Unit-radius vertices, polygons, and each polygon edge with the
    polygons on either side of it."""
    g = _geo.get(mid)
    if g is not None:
        return g
    md = MOD_DATA[mid]
    v = np.array([[p.x, p.y, p.z] for p in md.verts], dtype=float)
    c = (v.min(axis=0) + v.max(axis=0)) / 2
    v = v - c
    r = np.sqrt((v * v).sum(axis=1)).max()
    v = v / (r if r > 1e-9 else 1)
    polys = [list(p) for p in md.poly_faces]
    owners = {}
    for pi, p in enumerate(polys):
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]
            owners.setdefault((a, b) if a < b else (b, a), []).append(pi)
    g = _geo[mid] = (v, polys, list(owners.items()))
    return g


def _rot(spin):
    rx, ry = ICON_TILT[0], ICON_TILT[1] + spin
    cx, sx, cy, sy = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rx @ Ry  # three's Euler XYZ: Rx · Ry · Rz


def _hex(color):
    h = color.lstrip("#") if isinstance(color, str) else "c44444"
    try:
        return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    except ValueError:
        return np.array([0.77, 0.27, 0.27])


def frame(mid, color, spin=0.0, size=64, ss=3, dim=False):
    """RGBA float image (size, size, 4), top row first, straight alpha."""
    v, polys, edges = _geometry(mid)
    p = v @ _rot(spin).T
    N = size * ss
    k = 0.46 * N
    sx = N / 2 + p[:, 0] * k
    sy = N / 2 - p[:, 1] * k
    base = _hex(color)
    rgb = np.zeros((N, N, 3))
    alpha = np.zeros((N, N))

    normals, front, wound = [], [], []
    for poly in polys:
        q = p[poly]
        n = np.zeros(3)
        for i in range(len(q)):
            a, b = q[i], q[(i + 1) % len(q)]
            n += ((a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1]))
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-12 else n
        if n @ q.mean(axis=0) < 0:  # authored inward: wind it outward
            n = -n
            poly = poly[::-1]
        normals.append(n)
        front.append(n[2] > 1e-6)
        wound.append(poly)
    polys = wound

    order = sorted((i for i in range(len(polys)) if front[i]), key=lambda i: p[polys[i]][:, 2].mean())
    for pi in order:
        n = normals[pi]
        shade = 0.38 + 0.52 * max(0.0, float(n @ LIGHT)) + 0.12 * (0.5 + 0.5 * n[1])
        col = np.clip(base * shade, 0, 1)
        poly = polys[pi]
        x0, x1 = int(max(0, math.floor(sx[poly].min()))), int(min(N, math.ceil(sx[poly].max()) + 1))
        y0, y1 = int(max(0, math.floor(sy[poly].min()))), int(min(N, math.ceil(sy[poly].max()) + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        px, py = xs + 0.5, ys + 0.5
        inside = np.ones(px.shape, dtype=bool)
        m = len(poly)
        # Convex polygon, front-facing (counter-clockwise on screen, y down
        # makes it clockwise in pixel space): inside every edge.
        for i in range(m):
            ax, ay = sx[poly[i]], sy[poly[i]]
            bx, by = sx[poly[(i + 1) % m]], sy[poly[(i + 1) % m]]
            inside &= ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) <= 1e-9
        rgb[y0:y1, x0:x1][inside] = col
        alpha[y0:y1, x0:x1][inside] = 1.0

    ink = np.zeros((N, N))
    w = ss * 0.55
    for (a, b), owners in edges:
        if not any(front[o] for o in owners):
            continue
        ax, ay, bx, by = sx[a], sy[a], sx[b], sy[b]
        x0, x1 = int(max(0, math.floor(min(ax, bx) - w))), int(min(N, math.ceil(max(ax, bx) + w) + 1))
        y0, y1 = int(max(0, math.floor(min(ay, by) - w))), int(min(N, math.ceil(max(ay, by) + w) + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        px, py = xs + 0.5, ys + 0.5
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy or 1e-12
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
        d = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
        ink[y0:y1, x0:x1] = np.maximum(ink[y0:y1, x0:x1], (d <= w).astype(float))
    a_ink = EDGE_ALPHA * ink
    rgb = rgb * (1 - a_ink[..., None])
    alpha = alpha + a_ink * (1 - alpha)
    if dim:
        alpha = alpha * 0.3
    pre = np.concatenate([rgb * alpha[..., None], alpha[..., None]], axis=2)
    small = pre.reshape(size, ss, size, ss, 4).mean(axis=(1, 3))
    a = small[..., 3:4]
    out = np.where(a > 0, small[..., :3] / np.maximum(a, 1e-9), 0)
    return np.clip(np.concatenate([out, a], axis=2), 0, 1)


def spin_at(ms):
    """The turn the current module is at, `ms` milliseconds in."""
    return (ms * ICON_SPIN) % (2 * math.pi)
