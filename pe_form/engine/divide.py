"""Port of pe-web/src/engine/divide.ts: face division, the pure geometry
behind BuilderEngine.divide, which splits one open face into N x N smaller
copies of itself so that modules can grow on them at 1/N size. (splitAtPoints,
the same file's other edge repair, is in carve.py, which needed it first.)

three.js r128's own order of operations, so the small faces land on the web's
points to the last bit.
"""

import math

from .geom import face_key

# Largest N offered for one division.
DIVIDE_MAX = 8
# Smallest size a face may be divided down to, as a fraction of the full
# module edge: below 1/16 neighbouring face centroids come close enough to the
# engine's tolerances (stated in fractions of the FULL edge) to be confused.
DIVIDE_MIN_FINE = 1 / 16


def subdivide_polygon(verts, n):
    """A triangle or parallelogram split into n x n copies of itself, each
    wound as `verts`. A triangle gives n^2 triangles, a parallelogram n^2
    parallelograms. Anything else: None."""
    if not isinstance(n, int) or n < 2:
        return None
    a, b = verts[0], verts[1]
    if len(verts) == 3:
        c = verts[2]
        u = b.clone().sub(a).divide_scalar(n)
        v = c.clone().sub(a).divide_scalar(n)

        def P(i, j):
            return a.clone().add_scaled_vector(u, i).add_scaled_vector(v, j)

        out = []
        for j in range(n):
            i = 0
            while i + j < n:
                out.append([P(i, j), P(i + 1, j), P(i, j + 1)])
                if i + j < n - 1:
                    out.append([P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)])
                i += 1
        return out
    if len(verts) == 4:
        d = verts[3]
        u = b.clone().sub(a).divide_scalar(n)
        v = d.clone().sub(a).divide_scalar(n)

        def P(i, j):
            return a.clone().add_scaled_vector(u, i).add_scaled_vector(v, j)

        return [[P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)] for j in range(n) for i in range(n)]
    return None


def is_parallelogram(v, tol):
    """Is a 4-gon a parallelogram (the only quad that tiles into copies of itself)?"""
    return len(v) == 4 and v[0].clone().add(v[2]).distance_to(v[1].clone().add(v[3])) <= tol


def on_segment(p, a, b, tol):
    """Parameter of p along a->b, or None when p is not on it (off the line
    by more than tol, or outside the segment)."""
    ab = b.clone().sub(a)
    L2 = ab.length_sq()
    if L2 < 1e-18:
        return None
    t = p.clone().sub(a).dot(ab) / L2
    if t < -tol / math.sqrt(L2) or t > 1 + tol / math.sqrt(L2):
        return None
    q = a.clone().add_scaled_vector(ab, t)
    return t if q.distance_to(p) <= tol else None


def split_along_segments(faces, segments, tol):
    """Remove the T-junctions a division leaves along the divided face's
    boundary: every triangle with an edge lying on one of `segments` is fanned
    from its opposite corner through every vertex strictly inside that edge.
    Winding and tags kept; the products are marked split. Faces nothing
    touched pass through unchanged and in order."""
    pts = []

    def add_pt(p):
        if not any(q.distance_to(p) <= tol for q in pts):
            pts.append(p)

    for f in faces:
        for p in (f.va, f.vb, f.vc):
            if any(on_segment(p, a, b, tol) is not None for a, b in segments):
                add_pt(p)
    if not pts:
        return faces
    out = []

    def emit(f):
        vs = (f.va, f.vb, f.vc)
        for k in range(3):
            a, b, c = vs[k], vs[(k + 1) % 3], vs[(k + 2) % 3]
            if not any(on_segment(a, s0, s1, tol) is not None and on_segment(b, s0, s1, tol) is not None
                       for s0, s1 in segments):
                continue
            inner = []
            ln = a.distance_to(b)
            for p in pts:
                t = on_segment(p, a, b, tol)
                if t is not None and t * ln > tol and (1 - t) * ln > tol:
                    inner.append((t, p))
            if not inner:
                continue
            inner.sort(key=lambda x: x[0])
            chain = [a] + [p for _, p in inner] + [b]
            for i in range(len(chain) - 1):
                p, q, r = chain[i].clone(), chain[i + 1].clone(), c.clone()
                emit(f.copy_with(p, q, r, face_key(p, q, r), split=True))
            return
        out.append(f)

    for f in faces:
        emit(f)
    return out
