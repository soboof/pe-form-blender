"""Port of pe-web/src/engine/carve.ts, and splitAtPoints from divide.ts: the
geometry Carve is built from (BuilderEngine.carve holds the state).

A carved module is seated on the INNER side of an open face and its volume is
taken out of the formation. The result is the boundary of (formation minus
module): the shell less every part of it the module's faces lie on, plus the
module's faces that lie inside the material, wound into the cavity.
Coincidence is decided by area in the polygon's plane, not by matching
triangle centroids, because the shell and the module rarely fan a shared
polygon from the same corner.

Line for line with the web, in its order of operations, with V8's own
Math.hypot (`js_hypot`) and Math.atan2: the clipped corners become the
formation's vertices, their centroids become face keys, and a save names
faces by position.
"""

import math

from . import jsmath
from .vec import Vec3, Box3, box_from_points
from .collision import Hull, sat_hulls, shrink_verts
from .geom import face_key

# How far the module is pulled in toward its centre before the containment
# test: it legitimately touches the shell (its mouth lies on the picked face).
CARVE_SHRINK = 0.01
# The thinnest ring, as a fraction of the module edge, a smaller carve may
# leave standing round its mouth.
RING_MIN = 0.01


def js_round(x):
    return math.floor(x + 0.5)


def js_hypot(a, b):
    """V8's Math.hypot for two arguments: scaled by the larger, the squares
    summed with Kahan's compensation, then the square root."""
    a, b = abs(a), abs(b)
    if a != a or b != b:
        return math.inf if (a == math.inf or b == math.inf) else math.nan
    m = a if a > b else b
    if m == math.inf:
        return math.inf
    if m == 0:
        return 0.0
    s = 0.0
    comp = 0.0
    for v in (a, b):
        n = v / m
        summand = n * n - comp
        pre = s + summand
        comp = (pre - s) - summand
        s = pre
    return math.sqrt(s) * m


def _shell_ann(f):
    return getattr(f, "ann_key", None) or getattr(f, "ring_key", None)


def shell_crosses(w_verts, faces, shell):
    """Does any shell triangle pass through the (slightly shrunk) module? The
    whole containment argument: the module is convex and its mouth lies on a
    face whose back is material, so it can only leave the material by
    crossing the shell."""
    sh = shrink_verts(w_verts, CARVE_SHRINK)
    box = box_from_points(sh)
    mn, mx = box.min, box.max
    mine = None
    tri = [[0, 1, 2]]
    for f in shell:
        a, b, c = f.va, f.vb, f.vc
        if max(a.x, b.x, c.x) < mn.x or min(a.x, b.x, c.x) > mx.x:
            continue
        if max(a.y, b.y, c.y) < mn.y or min(a.y, b.y, c.y) > mx.y:
            continue
        if max(a.z, b.z, c.z) < mn.z or min(a.z, b.z, c.z) > mx.z:
            continue
        if mine is None:
            mine = Hull(sh, faces)
        if sat_hulls(mine, Hull([a, b, c], tri)):
            return True
    return False


# ── planar helpers ───────────────────────────────────────────────────────────
def poly_normal(pts):
    """Unit normal of a planar polygon, by Newell's method."""
    n = Vec3()
    L = len(pts)
    for i in range(L):
        a, b = pts[i], pts[(i + 1) % L]
        n.x += (a.y - b.y) * (a.z + b.z)
        n.y += (a.z - b.z) * (a.x + b.x)
        n.z += (a.x - b.x) * (a.y + b.y)
    return n.normalize()


def poly_area(pts):
    s = Vec3()
    L = len(pts)
    for i in range(L):
        s.add(Vec3().cross_vectors(pts[i], pts[(i + 1) % L]))
    return s.length() / 2


def _basis(n):
    ref = Vec3(1, 0, 0) if abs(n.x) < 0.9 else Vec3(0, 1, 0)
    u = Vec3().cross_vectors(n, ref).normalize()
    v = Vec3().cross_vectors(n, u)
    return u, v


def _ccw(p):
    a = 0
    L = len(p)
    for i in range(L):
        q, r = p[i], p[(i + 1) % L]
        a += q[0] * r[1] - r[0] * q[1]
    return list(reversed(p)) if a < 0 else p


def _inside_by(p, poly):
    """Signed distance of `p` inside every edge of CCW convex `poly`."""
    m = math.inf
    L = len(poly)
    for i in range(L):
        a, b = poly[i], poly[(i + 1) % L]
        ex, ey = b[0] - a[0], b[1] - a[1]
        ln = js_hypot(ex, ey) or 1
        d = (ex * (p[1] - a[1]) - ey * (p[0] - a[0])) / ln
        if d < m:
            m = d
    return m


def _overlaps(a, b, eps):
    """Do two convex polygons' interiors overlap by more than `eps`? (2D SAT.)"""
    for poly in (a, b):
        L = len(poly)
        for i in range(L):
            p, q = poly[i], poly[(i + 1) % L]
            nx, ny = q[1] - p[1], p[0] - q[0]
            ln = js_hypot(nx, ny)
            if ln < 1e-12:
                continue
            a_min, a_max, b_min, b_max = math.inf, -math.inf, math.inf, -math.inf
            for s in a:
                d = (s[0] * nx + s[1] * ny) / ln
                a_min = min(a_min, d)
                a_max = max(a_max, d)
            for s in b:
                d = (s[0] * nx + s[1] * ny) / ln
                b_min = min(b_min, d)
                b_max = max(b_max, d)
            if a_max <= b_min + eps or b_max <= a_min + eps:
                return False
    return True


def _off_box(f, mn, mx):
    a, b, c = f.va, f.vb, f.vc
    return (max(a.x, b.x, c.x) < mn.x or min(a.x, b.x, c.x) > mx.x
            or max(a.y, b.y, c.y) < mn.y or min(a.y, b.y, c.y) > mx.y
            or max(a.z, b.z, c.z) < mn.z or min(a.z, b.z, c.z) > mx.z)


def _plane2(n, o):
    u, v = _basis(n)

    def to2(p):
        d = p.clone().sub(o)
        return (d.dot(u), d.dot(v))
    return u, v, to2


def coplanar_cover(poly, faces, E):
    """How the shell `faces` cover convex polygon `poly`: (indices lying
    wholly inside it, "none" | "full" | "mixed")."""
    tol_plane, tol_in = E * 1e-3, E * 1e-3
    n = poly_normal(poly)
    o = poly[0]
    _, _, to2 = _plane2(n, o)
    P = _ccw([to2(p) for p in poly])
    area = poly_area(poly)
    box = box_from_points(poly).expand_by_scalar(tol_plane)
    mn, mx = box.min, box.max
    inside = []
    covered = 0
    mixed = False
    d = Vec3()
    for i, f in enumerate(faces):
        if _off_box(f, mn, mx):
            continue
        if any(abs(d.sub_vectors(p, o).dot(n)) > tol_plane for p in (f.va, f.vb, f.vc)):
            continue
        a = poly_area([f.va, f.vb, f.vc])
        if a < E * E * 1e-9:
            continue
        T = _ccw([to2(f.va), to2(f.vb), to2(f.vc)])
        if all(_inside_by(p, P) >= -tol_in for p in T):
            inside.append(i)
            covered += a
            continue
        if _overlaps(T, P, tol_in):
            mixed = True
            break
    if mixed:
        return inside, "mixed"
    if covered < area * 1e-3:
        return inside, ("mixed" if inside else "none")
    return inside, ("full" if abs(covered - area) <= area * 1e-3 else "mixed")


# ── cutting a module face out of the shell it partly lies on ────────────────
def _clip_half(poly, a, b, keep_right, eps):
    """Sutherland-Hodgman: the part of convex `poly` left of a->b, or right of it."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    ln = js_hypot(ex, ey) or 1

    def side(p):
        s = (ex * (p[1] - a[1]) - ey * (p[0] - a[0])) / ln
        return -s if keep_right else s

    out = []
    L = len(poly)
    for i in range(L):
        cur, prev = poly[i], poly[(i + L - 1) % L]
        sc, sp = side(cur), side(prev)
        cin, pin = sc >= -eps, sp >= -eps
        if cin:
            if not pin:
                t = sp / (sp - sc)
                out.append((prev[0] + (cur[0] - prev[0]) * t, prev[1] + (cur[1] - prev[1]) * t))
            out.append(cur)
        elif pin:
            t = sp / (sp - sc)
            out.append((prev[0] + (cur[0] - prev[0]) * t, prev[1] + (cur[1] - prev[1]) * t))
    return out


def _area2(p):
    a = 0
    L = len(p)
    for i in range(L):
        q, r = p[i], p[(i + 1) % L]
        a += q[0] * r[1] - r[0] * q[1]
    return abs(a) / 2


def _tidy(p, tol, min_area):
    """Drop repeated and collinear corners; None when nothing with area is left."""
    L = len(p)
    q = [pt for i, pt in enumerate(p)
         if js_hypot(pt[0] - p[(i + 1) % L][0], pt[1] - p[(i + 1) % L][1]) > tol]
    changed = True
    while changed and len(q) >= 3:
        changed = False
        n = len(q)
        for i in range(n):
            a, b, c = q[(i + n - 1) % n], q[i], q[(i + 1) % n]
            ln = js_hypot(c[0] - a[0], c[1] - a[1]) or 1
            if abs((c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])) / ln <= tol:
                del q[i]
                changed = True
                break
    return q if len(q) >= 3 and _area2(q) > min_area else None


def _subtract(a, b, eps, tol, min_area):
    """Convex `a` minus convex CCW `b`, as convex pieces."""
    out = []
    rest = a
    for i in range(len(b)):
        if rest is None:
            break
        p, q = b[i], b[(i + 1) % len(b)]
        piece = _tidy(_clip_half(rest, p, q, True, -eps), tol, min_area)
        if piece is not None:
            out.append(piece)
        rest = _tidy(_clip_half(rest, p, q, False, eps), tol, min_area)
    return out


class Cut:
    __slots__ = ("ok", "cut", "keep", "uncovered")


def cut_polygon(poly, faces, E):
    """Cut convex polygon `poly` out of the shell `faces` that partly lie on
    it. `keep` is what is left of each cut triangle (as (index, points)),
    `uncovered` the parts of the polygon the shell does not lie on."""
    tol_plane, tol_in, eps, tol, min_area = E * 1e-3, E * 1e-3, E * 1e-7, E * 1e-5, E * E * 1e-9
    n = poly_normal(poly)
    o = poly[0]
    u, v, to2 = _plane2(n, o)
    P = _ccw([to2(p) for p in poly])
    box = box_from_points(poly).expand_by_scalar(tol_plane)
    mn, mx = box.min, box.max
    snaps = list(poly)
    snap_lim = E * 1e-4

    def to3(p):
        w = o.clone().add_scaled_vector(u, p[0]).add_scaled_vector(v, p[1])
        for s in snaps:
            if s.distance_to(w) <= snap_lim:
                return s.clone()
        return w

    cut, tris = [], []
    ok = True
    on_area = 0
    d = Vec3()
    for i, f in enumerate(faces):
        if _off_box(f, mn, mx):
            continue
        if any(abs(d.sub_vectors(p, o).dot(n)) > tol_plane for p in (f.va, f.vb, f.vc)):
            continue
        if poly_area([f.va, f.vb, f.vc]) < E * E * 1e-9:
            continue
        T = _ccw([to2(f.va), to2(f.vb), to2(f.vc)])
        if not _overlaps(T, P, tol_in):
            continue
        if f.is_skirt or _shell_ann(f):
            ok = False
        cut.append(i)
        tris.append(T)
        snaps.extend((f.va, f.vb, f.vc))
        inter = T
        for k in range(len(P)):
            if inter is None:
                break
            inter = _tidy(_clip_half(inter, P[k], P[(k + 1) % len(P)], False, eps), tol, min_area)
        if inter is not None:
            on_area += _area2(inter)

    r = Cut()
    r.keep = []
    for k, T in enumerate(tris):
        for piece in _subtract(T, P, eps, tol, min_area):
            r.keep.append((cut[k], [to3(p) for p in piece]))
    rest = [P]
    for T in tris:
        rest = [x for p in rest for x in _subtract(p, T, eps, tol, min_area)]
    off_area = 0
    for p in rest:
        off_area += _area2(p)
    if abs(on_area + off_area - _area2(P)) > _area2(P) * 1e-3:
        ok = False
    r.ok = ok
    r.cut = cut
    r.uncovered = [[to3(p) for p in q] for q in rest]
    return r


# ── a smaller mouth cut out of the middle of a face ─────────────────────────
class _At:
    __slots__ = ("p", "q", "a")

    def __init__(self, p, q, a):
        self.p, self.q, self.a = p, q, a


def ring_around(face, mouth, shell, E):
    """The ring of `face` left standing round convex `mouth`, which lies wholly
    inside it: (indices of the face's triangles, ring triangles). None when it
    is not that case (the caller cuts the general way), "thin" when the ring
    would be under RING_MIN of an edge."""
    tol_plane, tol_in = E * 1e-3, E * 1e-3
    n = poly_normal(face)
    o = face[0]
    d = Vec3()
    if any(abs(d.sub_vectors(p, o).dot(n)) > tol_plane for p in mouth):
        return None
    u, v = _basis(n)

    def to2(p):
        d.sub_vectors(p, o)
        return (d.dot(u), d.dot(v))

    F = _ccw([to2(p) for p in face])
    M = _ccw([to2(p) for p in mouth])
    margin = min(_inside_by(p, F) for p in M)
    if margin <= tol_in:
        return None
    if margin < E * RING_MIN:
        return "thin"

    inside, status = coplanar_cover(face, shell, E)
    if status != "full":
        return None
    if any(shell[i].is_skirt or _shell_ann(shell[i]) for i in inside):
        return None

    outer = []
    for i in inside:
        for p in (shell[i].va, shell[i].vb, shell[i].vc):
            if abs(_inside_by(to2(p), F)) > tol_in:
                continue
            if not any(q.distance_to(p) <= E * 1e-5 for q in outer):
                outer.append(p)
    mc0, mc1 = 0, 0
    for p in M:
        mc0, mc1 = mc0 + p[0] / len(M), mc1 + p[1] / len(M)

    def loop(pts):
        out = []
        for p in pts:
            q = to2(p)
            out.append(_At(p, q, jsmath.atan2(q[1] - mc1, q[0] - mc0)))
        out.sort(key=lambda x: x.a)
        return out

    O = loop(outer)
    I = loop(mouth)
    if len(O) < 3 or len(I) < 3:
        return None
    a0 = O[0].a

    def norm(a):
        t = a - a0
        while t < 0:
            t += 2 * math.pi
        return t

    for x in O:
        x.a = norm(x.a)
    for x in I:
        x.a = norm(x.a)
    I.sort(key=lambda x: x.a)
    Oc = O + [_At(O[0].p, O[0].q, O[0].a + 2 * math.pi)]
    Ic = I + [_At(I[0].p, I[0].q, I[0].a + 2 * math.pi)]

    tris = []
    min_area = E * E * 1e-9
    i = j = 0
    while i < len(O) or j < len(I):
        outer_next = j >= len(I) or (i < len(O) and Oc[i + 1].a <= Ic[j + 1].a)
        t = (Oc[i], Oc[i + 1], Ic[j]) if outer_next else (Oc[i], Ic[j + 1], Ic[j])
        a, b, c = t[0].q, t[1].q, t[2].q
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 2 * min_area:
            return None
        tris.append((t[0].p, t[1].p, t[2].p))
        if outer_next:
            i += 1
        else:
            j += 1
    return inside, tris


def face_on_polygon(f, poly, E):
    """Is the centroid of triangle `f` on convex polygon `poly` (coplanar and inside)?"""
    n = poly_normal(poly)
    c = Vec3().add(f.va).add(f.vb).add(f.vc).divide_scalar(3)
    if abs(c.clone().sub(poly[0]).dot(n)) > E * 1e-3:
        return False
    tf = poly_normal([f.va, f.vb, f.vc])
    if abs(tf.dot(n)) < 1 - 1e-4:
        return False
    _, _, to2 = _plane2(n, poly[0])
    return _inside_by(to2(c), _ccw([to2(p) for p in poly])) > E * 1e-4


# ── edge valence ─────────────────────────────────────────────────────────────
def _welder(tol):
    """Welds points within `tol`, a point joining any vertex in its own grid
    cell or a neighbouring one (auditFaces' rule)."""
    cells = {}
    nxt = [0]

    def wid(p):
        cx, cy, cz = js_round(p.x / tol), js_round(p.y / tol), js_round(p.z / tol)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    arr = cells.get((cx + dx, cy + dy, cz + dz))
                    if arr:
                        for q, i in arr:
                            if q.distance_to(p) <= tol:
                                return i
        i = nxt[0]
        nxt[0] += 1
        cells.setdefault((cx, cy, cz), []).append((p, i))
        return i
    return wid


def valence_damage(before, removed, added, E):
    """Does replacing `removed` with `added` leave an edge it touches worse
    off? "open", "pinch", or None when the result is sound."""
    wid = _welder(E * 1e-3)

    def ek(a, b):
        return (a, b) if a < b else (b, a)

    def edges_of(f):
        a, b, c = wid(f.va), wid(f.vb), wid(f.vc)
        return (ek(a, b), ek(b, c), ek(c, a))

    touched = set()
    for f in removed:
        touched.update(edges_of(f))
    for f in added:
        touched.update(edges_of(f))
    if not touched:
        return None
    was = {}
    for f in before:
        for k in edges_of(f):
            if k in touched:
                was[k] = was.get(k, 0) + 1
    now = dict(was)
    for f in removed:
        for k in edges_of(f):
            now[k] = now.get(k, 0) - 1
    for f in added:
        for k in edges_of(f):
            now[k] = now.get(k, 0) + 1
    worst = None
    for k in touched:
        n = now.get(k, 0)
        if n == 0 or n == 2:
            continue
        if n == was.get(k, 0):
            continue
        if n > 2:
            return "pinch"
        worst = "open"
    return worst


# ── splitting edges at corners lying on them (divide.ts) ─────────────────────
def _on_segment(p, a, b, tol):
    ab = b.clone().sub(a)
    L2 = ab.length_sq()
    if L2 < 1e-18:
        return None
    t = p.clone().sub(a).dot(ab) / L2
    if t < -tol / math.sqrt(L2) or t > 1 + tol / math.sqrt(L2):
        return None
    q = a.clone().add_scaled_vector(ab, t)
    return t if q.distance_to(p) <= tol else None


def _seg_dist(p, a, b):
    ab = b.clone().sub(a)
    L2 = ab.length_sq()
    t = 0 if L2 < 1e-18 else max(0, min(1, p.clone().sub(a).dot(ab) / L2))
    return a.clone().add_scaled_vector(ab, t).distance_to(p)


def split_at_points(faces, pts, tol):
    """Split every triangle edge that has one of `pts` strictly inside it,
    only on the edge the point lies nearest to. The pieces keep the face's
    tags and are marked `split`."""
    uniq = []
    for p in pts:
        if not any(q.distance_to(p) <= tol for q in uniq):
            uniq.append(p)
    if not uniq:
        return faces
    box = box_from_points(uniq).expand_by_scalar(tol)
    out = []
    fb = Box3()

    def emit(f):
        vs = (f.va, f.vb, f.vc)
        if not fb.set_from_points(vs).intersects_box(box):
            out.append(f)
            return
        for k in range(3):
            a, b, c = vs[k], vs[(k + 1) % 3], vs[(k + 2) % 3]
            ln = a.distance_to(b)
            inner = []
            for p in uniq:
                t = _on_segment(p, a, b, tol)
                if t is None or t * ln <= tol or (1 - t) * ln <= tol:
                    continue
                d = _seg_dist(p, a, b)
                if _seg_dist(p, b, c) < d or _seg_dist(p, c, a) < d:
                    continue
                inner.append((t, p))
            if not inner:
                continue
            inner.sort(key=lambda x: x[0])
            chain = [a] + [x[1] for x in inner] + [b]
            for i in range(len(chain) - 1):
                p, q, r = chain[i].clone(), chain[i + 1].clone(), c.clone()
                emit(f.copy_with(p, q, r, face_key(p, q, r), split=True))
            return
        out.append(f)

    for f in faces:
        emit(f)
    return out
