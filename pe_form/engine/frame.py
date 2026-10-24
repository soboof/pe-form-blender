"""The formation's frame: where its origin is and which way its axes point.

In the web Form the base plate says what a formation stands on: a plane with
a point (the plate's centre, the Form's origin) and axes (X, and up). In
Blender that is the object's origin, so the plate is Set Origin: the
formation's mesh stands in its own local space with the plate's centre at
the origin, its floor the local XY plane and up local +Z.

The frame is the map from the formation as built (the engine's points, turned
Z up: meshplan.to_bl) to the mesh's local space: a turn, a uniform scale and
a shift, kept as a row-major 4 x 4. Whatever Blender does to the mesh's
points as a whole (Set Origin, Affect Only Origins, Apply Rotation or Scale)
changes it, and `fit` reads it back off the mesh, so the next rebuild keeps
it. It goes to the web as the save's `ground` and comes from it.

Pure: no bpy, so it is tested without Blender.
"""

import math

from .ground import clean_ground, ground_matrix

IDENT = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

# Engine Y-up to Blender Z-up, (x, y, z) -> (x, -z, y), and back.
_C = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
_CI = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


# ── small linear algebra ─────────────────────────────────────────────────────
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _len(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _len(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else None


def mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def apply(M, p):
    return (M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + M[0][3],
            M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + M[1][3],
            M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + M[2][3])


def turn(M, d):
    return (M[0][0] * d[0] + M[0][1] * d[1] + M[0][2] * d[2],
            M[1][0] * d[0] + M[1][1] * d[1] + M[1][2] * d[2],
            M[2][0] * d[0] + M[2][1] * d[1] + M[2][2] * d[2])


def scale_of(M):
    return _len((M[0][0], M[1][0], M[2][0]))


def inverse(M):
    """The inverse of a turn, uniform scale and shift."""
    s2 = scale_of(M) ** 2 or 1.0
    R = [[M[j][i] / s2 for j in range(3)] for i in range(3)]
    t = (M[0][3], M[1][3], M[2][3])
    out = [[R[i][0], R[i][1], R[i][2], -(R[i][0] * t[0] + R[i][1] * t[1] + R[i][2] * t[2])] for i in range(3)]
    return out + [[0.0, 0.0, 0.0, 1.0]]


def basis(X, Y, Z, o):
    """The matrix taking local (1,0,0), (0,1,0), (0,0,1) and 0 to X, Y, Z and o."""
    return [[X[0], Y[0], Z[0], o[0]], [X[1], Y[1], Z[1], o[1]], [X[2], Y[2], Z[2], o[2]], [0.0, 0.0, 0.0, 1.0]]


def flat(M):
    return [M[i][j] for i in range(4) for j in range(4)]


def unflat(v):
    try:
        M = [[float(v[i * 4 + j]) for j in range(4)] for i in range(4)]
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(x) for r in M for x in r) or not scale_of(M) > 1e-9:
        return None
    return M


def same(A, B, tol=1e-6):
    return all(abs(A[i][j] - B[i][j]) <= tol * (1 + abs(B[i][j])) for i in range(4) for j in range(4))


# ── read off the mesh ────────────────────────────────────────────────────────
def _triad(a, b, c):
    e1 = _unit(_sub(b, a))
    if e1 is None:
        return None
    ac = _sub(c, a)
    e2 = _unit(_sub(ac, tuple(x * _dot(ac, e1) for x in e1)))
    if e2 is None:
        return None
    return e1, e2, _cross(e1, e2)


def fit(base, cur, tol=1e-4):
    """The frame taking every point of `base` to the same point of `cur`, or
    None when no turn, uniform scale and shift does (the mesh was edited by
    hand, or mirrored). Exact, not least squares: Blender moves the mesh as a
    whole, so three points well apart say it all, and the rest check it."""
    n = len(base)
    if n < 3 or len(cur) != n:
        return None
    a = 0
    b = max(range(n), key=lambda i: _dot(_sub(base[i], base[a]), _sub(base[i], base[a])))
    u = _unit(_sub(base[b], base[a]))
    if u is None:
        return None

    def off_line(i):
        d = _sub(base[i], base[a])
        k = _dot(d, u)
        return _dot(d, d) - k * k

    c = max(range(n), key=off_line)
    if off_line(c) < 1e-12:
        return None
    tb = _triad(base[a], base[b], base[c])
    tc = _triad(cur[a], cur[b], cur[c])
    if tb is None or tc is None:
        return None
    s = _len(_sub(cur[b], cur[a])) / _len(_sub(base[b], base[a]))
    # R = Tc . Tb^T (columns the two triads).
    R = [[sum(tc[k][i] * tb[k][j] for k in range(3)) * s for j in range(3)] for i in range(3)]
    o = turn([r + [0.0] for r in R] + [[0, 0, 0, 1]], base[a])
    t = _sub(cur[a], o)
    M = [R[0] + [t[0]], R[1] + [t[1]], R[2] + [t[2]], [0.0, 0.0, 0.0, 1.0]]
    size = max(1.0, _len(_sub(base[b], base[a])) * s)
    step = max(1, n // 64)
    for i in list(range(0, n, step)) + [n - 1]:
        if _len(_sub(apply(M, base[i]), cur[i])) > tol * size:
            return None
    return M


# ── the web's ground ─────────────────────────────────────────────────────────
def from_ground(raw):
    """The frame a save's ground gives: the formation standing on its plate,
    the plate's centre at the origin. The identity when there is none."""
    m = ground_matrix(clean_ground(raw))
    if m is None:
        return [r[:] for r in IDENT]
    return mul(mul(_C, m.rows()), _CI)


def to_ground(M, tol=1e-6):
    """The save's ground for a frame: the plane under the origin, as the web
    keeps it (its outward normal, its point, its X, in the engine's own
    space). None for the seed plane (the frame is no turn and no shift)."""
    Mi = inverse(M)
    s = scale_of(M)
    plain = [[M[i][j] / (s if j < 3 and i < 3 else 1) for j in range(4)] for i in range(4)]
    if same(plain, IDENT, tol) and abs(M[0][3]) + abs(M[1][3]) + abs(M[2][3]) <= tol:
        return None
    eng = lambda v: (v[0], v[2], -v[1])   # Blender Z-up to the engine's Y-up
    p = eng(apply(Mi, (0.0, 0.0, 0.0)))
    n = _unit(eng(turn(Mi, (0.0, 0.0, -1.0))))
    x = _unit(eng(turn(Mi, (1.0, 0.0, 0.0))))
    r = lambda v: [round(c, 6) + 0.0 for c in v]
    return {"n": r(n), "p": r(p), "x": r(x)}


# ── standing on a face ───────────────────────────────────────────────────────
def newell(pts):
    nx = ny = nz = 0.0
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    return (nx, ny, nz)


def plate(pts, inside, face=True):
    """The move that stands the mesh on a plate through `pts` (in its local
    space now): a face's corners in order (`face`), or any 3 or more points.
    Returns K, local now to local after: the plate's centre to the origin,
    its outward normal straight down (away from `inside`, the formation's
    middle), its X along one of the plate's edges or square to one, whichever
    is nearest the X it has now, so it faces as nearly as it did. None when
    the points name no plane."""
    if len(pts) < 3:
        return None
    c = tuple(sum(p[k] for p in pts) / len(pts) for k in range(3))
    if face:
        n = _unit(newell(pts))
    else:
        # Points in no order: the normal of the largest triangle among them,
        # and every other point's plane turned to agree with it.
        best, n = 0.0, None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                for k in range(j + 1, len(pts)):
                    v = _cross(_sub(pts[j], pts[i]), _sub(pts[k], pts[i]))
                    if _len(v) > best:
                        best, n = _len(v), v
        n = _unit(n) if n is not None else None
    if n is None:
        return None
    if _dot(n, _sub(c, inside)) < 0 or (face is False and _dot(n, _sub(c, inside)) == 0):
        n = (-n[0], -n[1], -n[2])
    Z = (-n[0], -n[1], -n[2])
    # X: the edges' directions and their squares in the plane, the one
    # nearest the local X there now (else Y, if X is the plate's normal).
    want = (1.0, 0.0, 0.0)
    if abs(_dot(want, Z)) > 0.99:
        want = (0.0, 1.0, 0.0)
    w = _unit(_sub(want, tuple(z * _dot(want, Z) for z in Z)))
    cands = []
    ring = pts if face else pts[:]
    for i in range(len(ring)):
        e = _sub(ring[(i + 1) % len(ring)], ring[i])
        e = _unit(_sub(e, tuple(z * _dot(e, Z) for z in Z)))
        if e is None:
            continue
        sq = _cross(Z, e)
        for d in (e, sq):
            cands.append(d)
            cands.append((-d[0], -d[1], -d[2]))
    X = max(cands, key=lambda d: _dot(d, w)) if cands else w
    Y = _cross(Z, X)
    B = basis(X, Y, Z, c)    # plate frame, in local coords now
    return inverse(B)
