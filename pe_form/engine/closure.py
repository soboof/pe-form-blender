"""Keep the shell closed through a placement's removals.

A placement takes faces out of the shell two ways: wall cancellation (a new
face and an old one whose centroids lie within E*0.06 are dropped together)
and the interior purge (a face whose sample points all lie in some hull is
dropped). Both judge a face by where it is, neither by the edges it closes.
That is enough while modules meet face to face, but the SAT seat test shrinks
each hull 6%, so two modules may overlap a little, or leave a wedge a little
off a flat wall. A face then reads as buried (or as the twin of another) and
goes, and the faces around it are left with an edge that nothing covers.

The web draws a triangle soup and never notices. The mesh needs every edge in
an even number of faces (two, or four where two modules touch along an edge
alone), so after the removals the faces that were the only cover of an edge
come back (`faces_to_close`). Where the removals left every edge even this
finds nothing, so only shells that would have opened differ.
"""

import math


def faces_to_close(shell, pool, tol):
    """The faces of `pool` (taken out by a placement) to put back into
    `shell` so that fewer edges have an odd number of faces on them.

    The pool is taken a patch at a time, a patch being faces joined through
    edges no face of the shell is on (a hexagon's six triangles, a wall and
    the twin it was cancelled with). A patch goes back when that leaves
    fewer odd edges than before, and each one that does makes the count
    fall, so this ends. Corners within `tol` are the same corner. Returns the
    faces in pool order; `shell` and `pool` are not changed."""
    if not pool:
        return []
    tol2 = tol * tol
    inv = 1.0 / tol
    reps, grid = [], {}

    def cluster(p, make):
        cx, cy, cz = math.floor(p.x * inv), math.floor(p.y * inv), math.floor(p.z * inv)
        for ix in (cx - 1, cx, cx + 1):
            for iy in (cy - 1, cy, cy + 1):
                for iz in (cz - 1, cz, cz + 1):
                    for k in grid.get((ix, iy, iz), ()):
                        q = reps[k]
                        if (p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2 <= tol2:
                            return k
        if not make:
            return -1
        reps.append(p)
        grid.setdefault((cx, cy, cz), []).append(len(reps) - 1)
        return len(reps) - 1

    cand = []
    for f in pool:
        ks = (cluster(f.va, True), cluster(f.vb, True), cluster(f.vc, True))
        if len(set(ks)) == 3:
            cand.append((f, tuple(tuple(sorted(e)) for e in ((ks[0], ks[1]), (ks[1], ks[2]), (ks[2], ks[0])))))
    if not cand:
        return []

    # How many faces of the shell are on each edge a pool face has: the
    # shell's faces that touch both its corners.
    lo = [min(r.x for r in reps) - tol, min(r.y for r in reps) - tol, min(r.z for r in reps) - tol]
    hi = [max(r.x for r in reps) + tol, max(r.y for r in reps) + tol, max(r.z for r in reps) + tol]
    at = [set() for _ in reps]
    for f in shell:
        for p in (f.va, f.vb, f.vc):
            if p.x < lo[0] or p.x > hi[0] or p.y < lo[1] or p.y > hi[1] or p.z < lo[2] or p.z > hi[2]:
                continue
            k = cluster(p, False)
            if k >= 0:
                at[k].add(id(f))
    used = {}

    def uses(e):
        n = used.get(e)
        if n is None:
            n = used[e] = len(at[e[0]] & at[e[1]])
        return n

    # Patches: pool faces joined through edges the shell has no face on.
    par = list(range(len(cand)))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    seen = {}
    for i, (f, es) in enumerate(cand):
        for e in es:
            if uses(e) == 0:
                j = seen.setdefault(e, i)
                par[find(i)] = find(j)
    patches = {}
    for i in range(len(cand)):
        patches.setdefault(find(i), []).append(i)
    patches = list(patches.values())

    def odd_change(idx):
        add = {}
        for i in idx:
            for e in cand[i][1]:
                add[e] = add.get(e, 0) + 1
        return sum(((uses(e) + n) & 1) - (uses(e) & 1) for e, n in add.items())

    back = set()
    changed = True
    while changed:
        changed = False
        for idx in patches:
            if idx[0] in back or odd_change(idx) >= 0:
                continue
            for i in idx:
                back.add(i)
                for e in cand[i][1]:
                    used[e] = uses(e) + 1
            changed = True
    return [f for i, (f, es) in enumerate(cand) if i in back]
