"""Port of pe-web/src/engine/geom.ts (the parts the Form uses)."""

import math

from . import jsmath

from .vec import Vec3, Quat
from .jsnum import to_fixed


def cen3(a, b, c):
    return Vec3((a.x + b.x + c.x) / 3, (a.y + b.y + c.y) / 3, (a.z + b.z + c.z) / 3)


def nor3(a, b, c):
    return Vec3().cross_vectors(Vec3().sub_vectors(b, a), Vec3().sub_vectors(c, a)).normalize()


def get_edge(X):
    """Design parameter X -> world edge length. THE core formula: X*2/sqrt(3)."""
    return (X * 2) / math.sqrt(3)


def get_scale(md, X):
    return get_edge(X) / md.base_edge


def face_key(a, b, c):
    cx = (a.x + b.x + c.x) / 3
    cy = (a.y + b.y + c.y) / 3
    cz = (a.z + b.z + c.z) / 3
    return to_fixed(cx, 4) + "," + to_fixed(cy, 4) + "," + to_fixed(cz, 4)


def cen_n(pts):
    out = Vec3()
    for p in pts:
        out.add(p)
    return out.divide_scalar(len(pts))


def nor_n(pts):
    return nor3(pts[0], pts[1], pts[2])


def face_key_n(pts):
    c = cen_n(pts)
    return to_fixed(c.x, 4) + "," + to_fixed(c.y, 4) + "," + to_fixed(c.z, 4)


def is_rhomb_face(pts):
    if len(pts) != 4:
        return False
    d1 = pts[0].distance_to(pts[2])
    d2 = pts[1].distance_to(pts[3])
    return abs(d1 - d2) > max(d1, d2) * 0.02


def is_regular_poly(pts, tol=0.02):
    n = len(pts)
    if n < 3:
        return False
    c = cen_n(pts)
    sides = [pts[i].distance_to(pts[(i + 1) % n]) for i in range(n)]
    radii = [p.distance_to(c) for p in pts]

    def spread(a):
        return (max(a) - min(a)) / max(a)

    return spread(sides) <= tol and spread(radii) <= tol


def mouth_shape(pts):
    n = len(pts)
    per = 0
    for i in range(n):
        per = per + pts[i].distance_to(pts[(i + 1) % n])
    per = per or 1
    sides = sorted(pts[i].distance_to(pts[(i + 1) % n]) / per for i in range(n))
    diags = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            diags.append(pts[i].distance_to(pts[j]) / per)
    return sides + sorted(diags)


def same_mouth_shape(a, b, tol=0.02):
    if len(a) != len(b):
        return False
    sa, sb = mouth_shape(a), mouth_shape(b)
    return all(abs(v - sb[i]) <= tol * max(v, sb[i], 1e-9) for i, v in enumerate(sa))


def split_kind_of(f):
    return "hex" if (f.face_type == 6 or f.hex_sub) else None


def short_diagonal_first(pts):
    if len(pts) != 4:
        return pts
    if pts[0].distance_to(pts[2]) <= pts[1].distance_to(pts[3]):
        return pts
    return [pts[1], pts[2], pts[3], pts[0]]


def regular_polygon(n, edge):
    R = edge / (2 * jsmath.sin(math.pi / n))
    out = []
    for k in range(n):
        th = (-float(k) * 2 * math.pi) / n  # float: -0.0 as in JS, so the seed's first z is -0
        out.append(Vec3(R * jsmath.cos(th), 0, R * jsmath.sin(th)))
    return out


def build_tri_poly_map(poly_faces):
    m = []
    for pfi, poly in enumerate(poly_faces):
        for _ in range(len(poly) - 2):
            m.append(pfi)
    return m


def axis_angle(axis, angle):
    return Quat().set_from_axis_angle(axis, angle)
