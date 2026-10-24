"""A picture of a formation (or of its skin, posed) for an asset's preview:
its triangles flat shaded in their colours, seen from the three-quarter view
Blender's own viewport opens on, framed to fill the square.

Pure numpy, no bpy and no GPU, so a preview is made the same way in a window
and in background Blender, at once (Blender's own preview render runs later,
in a job, and an asset written straight after it would have none).
"""

import math

import numpy as np

# Blender's default view: from front right (+X, -Y), a little above.
_EYE = np.array([1.0, -1.0, 0.75])
_LIGHT_VIEW = np.array([0.35, 0.6, 0.72])   # in view space: up, right, toward the eye


def _basis():
    b = -_EYE / np.linalg.norm(_EYE)              # looking along
    r = np.cross(b, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, b)
    return r, u, -b                               # right, up, toward the eye


def snapshot(tris, colors, size=128, ss=2, margin=0.08):
    """RGBA float image (size, size, 4), top row first, straight alpha.
    `tris` is (n, 3, 3) points (Z up), `colors` (n, 3) linear-ish RGB 0..1."""
    T = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    C = np.asarray(colors, dtype=float).reshape(-1, 3)
    N = size * ss
    rgb = np.zeros((N, N, 3))
    alpha = np.zeros((N, N))
    if not len(T):
        return np.zeros((size, size, 4))
    r, u, e = _basis()
    X, Y, Z = T @ r, T @ u, T @ e                  # (n, 3) each
    x0, x1, y0, y1 = X.min(), X.max(), Y.min(), Y.max()
    span = max(x1 - x0, y1 - y0, 1e-9)
    k = N * (1 - 2 * margin) / span
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    SX = N / 2 + (X - cx) * k
    SY = N / 2 - (Y - cy) * k
    zbuf = np.full((N, N), -np.inf)
    light = _LIGHT_VIEW / np.linalg.norm(_LIGHT_VIEW)
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(nrm, axis=1)
    keep = ln > 1e-12
    nrm[keep] /= ln[keep, None]
    nv = np.stack([nrm @ r, nrm @ u, nrm @ e], axis=1)       # normal in view space
    nv[nv[:, 2] < 0] *= -1                                    # seen from either side
    shade = 0.32 + 0.58 * np.clip(nv @ light, 0, 1) + 0.12 * (0.5 + 0.5 * nv[:, 1])
    cols = np.clip(C * shade[:, None], 0, 1)
    for i in np.nonzero(keep)[0]:
        ax, bx, cx_ = SX[i]
        ay, by, cy_ = SY[i]
        px0, px1 = int(max(0, math.floor(min(ax, bx, cx_)))), int(min(N, math.ceil(max(ax, bx, cx_)) + 1))
        py0, py1 = int(max(0, math.floor(min(ay, by, cy_)))), int(min(N, math.ceil(max(ay, by, cy_)) + 1))
        if px0 >= px1 or py0 >= py1:
            continue
        area = (bx - ax) * (cy_ - ay) - (by - ay) * (cx_ - ax)
        if abs(area) < 1e-12:
            continue
        ys, xs = np.mgrid[py0:py1, px0:px1]
        px, py = xs + 0.5, ys + 0.5
        w0 = ((bx - px) * (cy_ - py) - (by - py) * (cx_ - px)) / area
        w1 = ((cx_ - px) * (ay - py) - (cy_ - py) * (ax - px)) / area
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        z = w0 * Z[i, 0] + w1 * Z[i, 1] + w2 * Z[i, 2]
        zb = zbuf[py0:py1, px0:px1]
        win = inside & (z > zb)
        zb[win] = z[win]
        rgb[py0:py1, px0:px1][win] = cols[i]
        alpha[py0:py1, px0:px1][win] = 1.0
    pre = np.concatenate([rgb * alpha[..., None], alpha[..., None]], axis=2)
    small = pre.reshape(size, ss, size, ss, 4).mean(axis=(1, 3))
    a = small[..., 3:4]
    out = np.where(a > 0, small[..., :3] / np.maximum(a, 1e-9), 0)
    return np.clip(np.concatenate([out, a], axis=2), 0, 1)
