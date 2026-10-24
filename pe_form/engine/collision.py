"""Port of pe-web/src/engine/collision.ts: SAT on 6%-shrunk hulls.

The verdict is the web's: the same shrink, the same box pre-test, the same
axes in the same order (both hulls' face normals, then every edge pair's
cross product). Two things differ, for Python's sake, and neither changes a
verdict: axes are tested as they are made, stopping at the first that
separates the hulls (the web builds them all first, then stops at the same
one), and each hull's axes are worked out once and kept, with directions
that repeat (the fan triangles of one polygon, a polygon's parallel edges)
tested once. The golden tests hold the port to the web's answers.
"""

from .vec import Vec3, box_from_points

ISECT_SHRINK = 0.06  # tuned: smaller = false blocks, larger = missed collisions


def shrink_verts(verts, f):
    cx = cy = cz = 0
    n = len(verts)
    for v in verts:
        cx += v.x
        cy += v.y
        cz += v.z
    cx /= n
    cy /= n
    cz /= n
    return [Vec3(v.x + (cx - v.x) * f, v.y + (cy - v.y) * f, v.z + (cz - v.z) * f) for v in verts]


def _dedupe(dirs):
    seen = set()
    out = []
    for d in dirs:
        k = (round(d[0], 9), round(d[1], 9), round(d[2], 9))
        if k[0] < 0 or (k[0] == 0 and (k[1] < 0 or (k[1] == 0 and k[2] < 0))):
            k = (-k[0], -k[1], -k[2])
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
    return out


class Hull:
    """A convex hull ready for SAT: its points, face normals and edge directions."""
    __slots__ = ("xs", "ys", "zs", "axes", "edges")

    def __init__(self, verts, faces):
        self.xs = [v.x for v in verts]
        self.ys = [v.y for v in verts]
        self.zs = [v.z for v in verts]
        axes = []
        for f in faces:
            a = verts[f[0]]
            e1 = Vec3().sub_vectors(verts[f[1]], a)
            e2 = Vec3().sub_vectors(verts[f[2]], a)
            n = Vec3().cross_vectors(e1, e2)
            if n.length_sq() > 1e-10:
                n.normalize()
                axes.append((n.x, n.y, n.z))
        self.axes = _dedupe(axes)
        seen = set()
        edges = []
        for fc in faces:
            for k in range(3):
                a, b = fc[k], fc[(k + 1) % 3]
                key = (a, b) if a < b else (b, a)
                if key not in seen:
                    seen.add(key)
                    d = Vec3().sub_vectors(verts[b], verts[a]).normalize()
                    edges.append((d.x, d.y, d.z))
        self.edges = _dedupe(edges)

    def span(self, ax, ay, az):
        ds = [x * ax + y * ay + z * az for x, y, z in zip(self.xs, self.ys, self.zs)]
        return min(ds), max(ds)


def _separates(A, B, ax, ay, az):
    amn, amx = A.span(ax, ay, az)
    bmn, bmx = B.span(ax, ay, az)
    return amx < bmn or bmx < amn


def _centre(H):
    n = len(H.xs)
    return sum(H.xs) / n, sum(H.ys) / n, sum(H.zs) / n


def sat_hulls(A, B):
    """True when the two convex hulls overlap on every axis (intersect)."""
    # First the line between their centres: not one of the web's axes, but
    # an axis that separates two hulls proves them apart, and between two
    # modules on a shared face it is nearly the face's normal, so it settles
    # most seats at once. Overlapping hulls have no separating axis at all.
    ca, cb = _centre(A), _centre(B)
    dx, dy, dz = cb[0] - ca[0], cb[1] - ca[1], cb[2] - ca[2]
    l2 = dx * dx + dy * dy + dz * dz
    if l2 > 1e-20:
        inv = 1 / (l2 ** 0.5)
        if _separates(A, B, dx * inv, dy * inv, dz * inv):
            return False
    for ax in A.axes:
        if _separates(A, B, *ax):
            return False
    for ax in B.axes:
        if _separates(A, B, *ax):
            return False
    for ex, ey, ez in A.edges:
        for fx, fy, fz in B.edges:
            cx = ey * fz - ez * fy
            cy = ez * fx - ex * fz
            cz = ex * fy - ey * fx
            l2 = cx * cx + cy * cy + cz * cz
            if l2 > 1e-10:
                inv = 1 / (l2 ** 0.5)
                if _separates(A, B, cx * inv, cy * inv, cz * inv):
                    return False
    return True


def sat_convex(va, faces_a, vb, faces_b):
    return sat_hulls(Hull(va, faces_a), Hull(vb, faces_b))


class ModuleWorldData:
    __slots__ = ("verts", "shrunk_verts", "faces", "box", "shrunk_box", "planes", "_hull")

    def __init__(self, verts, faces):
        self.verts = verts
        self.shrunk_verts = shrink_verts(verts, ISECT_SHRINK)
        self.faces = faces
        self.box = box_from_points(verts)
        self.shrunk_box = box_from_points(self.shrunk_verts)
        self.planes = None  # outward hull planes, filled by the interior purge
        self._hull = None

    @property
    def hull(self):
        if self._hull is None:
            self._hull = Hull(self.shrunk_verts, self.faces)
        return self._hull


def check_intersection(w_verts, faces, existing):
    if not existing:
        return False
    shr = shrink_verts(w_verts, ISECT_SHRINK)
    box = box_from_points(shr)
    mine = None
    for ex in existing:
        if not box.intersects_box(ex.shrunk_box):
            continue
        if mine is None:
            mine = Hull(shr, faces)
        if sat_hulls(mine, ex.hull):
            return True
    return False
