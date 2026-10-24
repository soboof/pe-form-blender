"""The formation's shell as mesh polygons (pure: no bpy, so it is tested
without Blender). Used by bl/formation.py to build the Blender mesh.

  - each open face (where a module can go) is one polygon, wound outward;
  - a hexagon stays its six centre triangles, as the engine keeps it, each
    tagged with its sub-seat and the whole hexagon;
  - a divided face's small faces are each a polygon of their own (they share
    their module and face index with the face they were cut from, so their
    triangles are told apart by which small face each lies on);
  - the rest (skirts, faces nothing can attach to) stays triangles.

Where modules meet, a corner is worked out from each of them and the copies
land a hair apart (measured on the fixtures and the demo pack: up to 2.5e-4
of an edge, the formation's own geometry never closer than 1.3e-2). They
are joined into one vertex, the copy of the oldest module that has one
(`_join`), so the mesh is closed where the shell is, every polygon at a
corner uses the same exact point, and a face a placement did not touch
keeps its points (measured: all but 25 of 70,668 corners from step to step,
those where a copy of an older module appeared; usergeo finds such a face
again within the joining's reach).

Engine Y-up to Blender Z-up: (x, y, z) is (x, -z, y).
"""

import math

from .geom import get_edge

JOIN = 1e-3   # two corners closer than this many module edges are one
THIN = 0.1    # and never closer than this much of their polygon's shortest edge


def to_bl(v):
    return (v.x, -v.z, v.y)


def _srgb(h):
    h = h.lstrip("#") if isinstance(h, str) else "c44444"
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, 1.0)
    except ValueError:
        return (0.77, 0.27, 0.27, 1.0)


# ── the mesh, polygon by polygon ─────────────────────────────────────────────
def _newell(pts):
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        nx += (a.y - b.y) * (a.z + b.z)
        ny += (a.z - b.z) * (a.x + b.x)
        nz += (a.x - b.x) * (a.y + b.y)
    return nx, ny, nz


def _inside(c, pts, n):
    """Is point c on the convex polygon pts (normal n), within a hair?"""
    size = max(_d2(pts[0], p) for p in pts) ** 0.5 or 1.0
    eps = size * 1e-4
    a0 = pts[0]
    if abs((c.x - a0.x) * n[0] + (c.y - a0.y) * n[1] + (c.z - a0.z) * n[2]) > eps * 10:
        return False
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        ex, ey, ez = b.x - a.x, b.y - a.y, b.z - a.z
        cx, cy, cz = c.x - a.x, c.y - a.y, c.z - a.z
        # (b - a) x (c - a) . n: positive inside for a polygon wound about n.
        s = (ey * cz - ez * cy) * n[0] + (ez * cx - ex * cz) * n[1] + (ex * cy - ey * cx) * n[2]
        if s < -eps * size:
            return False
    return True


def _d2(a, b):
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2


def _unit(n):
    L = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]) ** 0.5 or 1.0
    return (n[0] / L, n[1] / L, n[2] / L)


def _outline(tris):
    """The one boundary loop of a fan of triangles, in their winding (a face
    re-cut along its edges, whose corners now include its neighbours' grid
    corners), or None when they make no single loop."""
    key = lambda v: (round(v.x, 6), round(v.y, 6), round(v.z, 6))
    pt, nxt, seen = {}, {}, set()
    for t in tris:
        vs = (t.va, t.vb, t.vc)
        for i in range(3):
            a, b = vs[i], vs[(i + 1) % 3]
            ka, kb = key(a), key(b)
            pt[ka], pt[kb] = a, b
            seen.add((ka, kb))
    for ka, kb in seen:
        if (kb, ka) in seen:
            continue
        if ka in nxt:
            return None
        nxt[ka] = kb
    if len(nxt) < 3:
        return None
    start = next(iter(nxt))
    loop, k = [], start
    for _ in range(len(nxt)):
        loop.append(pt[k])
        k = nxt.get(k)
        if k is None:
            return None
        if k == start:
            break
    return loop if k == start and len(loop) == len(nxt) else None


def _cen(t):
    class _P:
        __slots__ = ("x", "y", "z")
    p = _P()
    p.x = (t.va.x + t.vb.x + t.vc.x) / 3
    p.y = (t.va.y + t.vb.y + t.vc.y) / 3
    p.z = (t.va.z + t.vb.z + t.vc.z) / 3
    return p


def plan_polygons(engine):
    """The shell's polygons, each (engine points, colour, open, hex, module),
    the corners as the engine has them (before they are joined)."""
    e = engine
    poly_open, sub_open, hex_open, fine_open = {}, {}, {}, {}
    for i, o in enumerate(e.open_faces):
        if o.fine is not None and not o.hex_sub:
            fine_open.setdefault((o.module_idx, o.canonical_face_idx), []).append(i)
        elif o.hex_sub:
            sub_open[(o.module_idx, o.canonical_face_idx, o.sub_k)] = i
        elif o.face_type == 6 and o.module_idx >= 0:
            hex_open[(o.module_idx, o.canonical_face_idx)] = i
        else:
            poly_open[(o.module_idx, o.canonical_face_idx)] = i
    out = []          # (engine points, colour, open, hex)
    groups = {}
    order = []
    for t in e.all_faces:
        m = t.mod_idx if t.mod_idx is not None else -1
        if t.sub_k is not None:
            out.append(([t.va, t.vb, t.vc], t.color, sub_open.get((m, t.poly_idx, t.sub_k), -1),
                        hex_open.get((m, t.poly_idx), -1), m))
            continue
        if t.is_skirt or (t.poly_idx is None and m != -1):
            out.append(([t.va, t.vb, t.vc], t.color, -1, -1, m))
            continue
        key = (m, t.poly_idx)
        g = groups.get(key)
        if g is None:
            g = groups[key] = []
            order.append(key)
            out.append(key)   # placeholder, keeps the shell's order
        g.append(t)
    plan = []
    for item in out:
        if not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], int):
            plan.append(item)
            continue
        tris = groups[item]
        m = item[0]
        fines = fine_open.get(item)
        if fines:
            # A divided face: each triangle to the small face it lies on.
            per = {}
            rest = []
            norms = {i: _unit(_newell(e.open_faces[i].verts3)) for i in fines}
            for t in tris:
                c = _cen(t)
                hit = next((i for i in fines if _inside(c, e.open_faces[i].verts3, norms[i])), None)
                if hit is None:
                    rest.append(t)
                else:
                    per.setdefault(hit, []).append(t)
            for i in fines:
                ts = per.get(i)
                if not ts:
                    continue
                o = e.open_faces[i]
                if len(ts) == len(o.verts3) - 2:
                    pts = list(o.verts3)
                    n = _newell(pts)
                    if n[0] * o.normal3.x + n[1] * o.normal3.y + n[2] * o.normal3.z < 0:
                        pts.reverse()
                    plan.append((pts, ts[0].color, i, -1, m))
                else:
                    ring = _outline(ts)
                    if ring is not None:
                        plan.append((ring, ts[0].color, i, -1, m))
                    else:
                        for t in ts:
                            plan.append(([t.va, t.vb, t.vc], t.color, i, -1, m))
            for t in rest:
                plan.append(([t.va, t.vb, t.vc], t.color, poly_open.get(item, -1), -1, m))
            continue
        oi = poly_open.get(item, -1)
        o = e.open_faces[oi] if oi >= 0 else None
        if o is not None and len(tris) == len(o.verts3) - 2:
            pts = list(o.verts3)
            n = _newell(pts)
            if n[0] * o.normal3.x + n[1] * o.normal3.y + n[2] * o.normal3.z < 0:
                pts.reverse()
            plan.append((pts, tris[0].color, oi, -1, m))
        else:
            # An open face re-cut along its edges (beside a divided face) is
            # still one polygon: its outline, through the new corners.
            ring = _outline(tris) if o is not None and any(t.split for t in tris) else None
            if ring is not None:
                plan.append((ring, tris[0].color, oi, -1, m))
            else:
                for t in tris:
                    plan.append(([t.va, t.vb, t.vc], t.color, oi, -1, m))
    return plan


def module_edges(engine):
    """Each module's edge by history index, and the seed's under -1: the size
    it was placed at (its step's Height X), which a later Height X does not
    change, so the corners are joined the same whatever X is picked now."""
    e = engine
    x0 = next((d.step_x for d in e.divides if d.at == 0), None)
    if x0 is None and e.history:
        x0 = e.history[0].step_x
    out = {-1: get_edge(e.X if x0 is None else x0)}
    for i, h in enumerate(e.history):
        out[i] = get_edge(e.X if h.step_x is None else h.step_x)
    return out


def join_tol(engine, m, edges=None):
    """How far a corner of module m's faces can be moved by the joining."""
    edges = module_edges(engine) if edges is None else edges
    return JOIN * edges.get(m, edges[-1])


def _join(plan, edges):
    """Every corner copy (an exact point) to the copy it is joined to.

    A copy may join another no further than JOIN of the edge of every module
    whose polygons use either, and THIN of their polygons' shortest edges, so
    a thin ring or a small module's face is never closed up. Copies are taken
    oldest module first (one module's own, by the point), and each joins the
    nearest copy kept before it: the copy kept is the oldest module's, which
    a later placement seldom changes, whatever order the shell lists its
    faces in."""
    tol, mod = {}, {}
    for pts, col, oi, hx, m in plan:
        b = [to_bl(v) for v in pts]
        short = min(math.dist(b[i], b[(i + 1) % len(b)]) for i in range(len(b)))
        t = JOIN * edges.get(m, edges[-1])
        if short > 0:
            t = min(t, THIN * short)
        for p in b:
            if p not in tol:
                tol[p] = t
                mod[p] = m
            else:
                tol[p] = min(tol[p], t)
                mod[p] = min(mod[p], m)
    cell = max(tol.values(), default=0.0)
    to = {}
    if cell <= 0:
        return {p: p for p in tol}
    grid, keep = {}, {}
    for p in sorted(tol, key=lambda p: (mod[p], p)):
        c = (math.floor(p[0] / cell), math.floor(p[1] / cell), math.floor(p[2] / cell))
        best, bd = None, None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for q in grid.get((c[0] + dx, c[1] + dy, c[2] + dz), ()):
                        d = math.dist(p, q)
                        if d <= min(tol[p], keep[q]) and (bd is None or d < bd):
                            best, bd = q, d
        if best is None:
            grid.setdefault(c, []).append(p)
            keep[p] = tol[p]
            to[p] = p
        else:
            keep[best] = min(keep[best], tol[p])
            to[p] = best
    return to


def mesh_plan(engine, mods=None):
    """(verts, polys, colours, open index, hex index) for the engine's shell.
    Given a list as `mods`, it is filled with each polygon's module (its
    history index, -1 for none), in the order of `polys`."""
    plan = plan_polygons(engine)
    to = _join(plan, module_edges(engine))
    verts, index, polys, cols, opens, hexes = [], {}, [], [], [], []
    for pts, col, oi, hx, m in plan:
        ids = []
        for v in pts:
            b = to[to_bl(v)]
            i = index.get(b)
            if i is None:
                i = len(verts)
                index[b] = i
                verts.append(b)
            if not ids or ids[-1] != i:
                ids.append(i)
        if len(ids) > 1 and ids[0] == ids[-1]:
            ids.pop()
        if len(ids) < 3 or len(set(ids)) != len(ids):
            continue
        polys.append(ids)
        cols.append(_srgb(col))
        opens.append(oi)
        hexes.append(hx)
        if mods is not None:
            mods.append(m)
    return verts, polys, cols, opens, hexes


