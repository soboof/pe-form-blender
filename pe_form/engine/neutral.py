"""The Theater's neutral pose (rig.ts computeNeutral, neutralPose), ported
operation for operation: the pose every library motion plays from.

Legs and paws straight down, arms hanging at the sides (ARM_OUT, 15 degrees
out, and raised a few degrees at a time until none of their modules runs into
the torso or the legs), wings spread level out to the sides; spine, head and
tail stay as built. The root is lifted along up by however far that puts the
lowest point of the shell under the floor it stood on (legs straightened down
reach lower).

What is ported is the pose, its lift and its arms: the walk's plan of each leg
(soles, reach, which way the knee bends) is the motions', which the add-on
does not play. A neutral pose set by hand in the Theater is not read: in
Blender a pose of one's own is a pose asset.

The body (SkelBody, bodyHulls.ts makeSkelBody) is built here as Rigid modules
binds it: every module whole to the joint that owns it (module_owners).

Pure Python: no bpy. Held to the web's answers by tests/golden/rigs.json.
"""

import math

from . import jsmath
from .vec import Vec3, Quat, box_from_points
from .collision import shrink_verts, sat_convex, sat_hulls, Hull, ISECT_SHRINK
from .jsnum import to_fixed
from .rig import skel_chains, skel_facing, _chain_dir

try:
    import numpy as _np
except ImportError:      # outside Blender with no numpy: plain sat_hulls
    _np = None


def _np_hull(h):
    """A Hull as arrays: its points, and its face normals and edge directions."""
    P = _np.array([h.xs, h.ys, h.zs], dtype=float)
    ax = _np.array(h.axes, dtype=float).reshape(-1, 3)
    ed = _np.array(h.edges, dtype=float).reshape(-1, 3)
    return P, ax, ed


def _np_separates(PA, PB, ax):
    """Any axis (rows of ax) on which the two point sets' spans do not meet."""
    if not len(ax):
        return False
    da = PA[0][:, None] * ax[:, 0] + PA[1][:, None] * ax[:, 1] + PA[2][:, None] * ax[:, 2]
    db = PB[0][:, None] * ax[:, 0] + PB[1][:, None] * ax[:, 1] + PB[2][:, None] * ax[:, 2]
    return bool(_np.any((da.max(0) < db.min(0)) | (db.max(0) < da.min(0))))


def _np_sat(A, B):
    """sat_hulls' verdict, every axis at once: the hulls meet unless one of
    the axes it tests (both hulls' face normals, every edge pair's cross
    product, and the line between their centres) separates them."""
    PA, axa, eda = A
    PB, axb, edb = B
    ca, cb = PA.mean(1), PB.mean(1)
    d = cb - ca
    l2 = float(d @ d)
    axes = [axa, axb]
    if l2 > 1e-20:
        axes.append((d / l2 ** 0.5)[None, :])
    if len(eda) and len(edb):
        c = _np.cross(eda[:, None, :], edb[None, :, :]).reshape(-1, 3)
        n2 = (c * c).sum(1)
        keep = n2 > 1e-10
        axes.append(c[keep] / _np.sqrt(n2[keep])[:, None])
    return not _np_separates(PA, PB, _np.vstack(axes))


NEUTRAL_TYPES = ("leg", "paw", "arm", "wing")
ARM_OUT = (15 * math.pi) / 180
ARM_STEP, ARM_MAX = (3 * math.pi) / 180, math.pi
TOUCH_GROW = 0.03
ANKLE_FRAC, ANKLE_DROP = 0.25, 0.5
BODY_TYPES = {"spine", "head", "tail", "shoulder", "arm", "leg", "paw", "wing"}
_AY = Vec3(0, 1, 0)


# ── three's bits the neutral pose uses, as r128 does them ───────────────────
def _invert(q):
    """Quaternion.invert in r128: the conjugate."""
    return Quat(-q.x, -q.y, -q.z, q.w)


def _angle_to(a, b):
    den = math.sqrt(a.length_sq() * b.length_sq())
    if den == 0:
        return math.pi / 2
    t = a.dot(b) / den
    return math.acos(max(-1.0, min(1.0, t)))


def fk_pose(J, quats, root_offset=None):
    """fkPose: each joint's place and world turn under local turns `quats`."""
    n = len(J)
    pos, rot = [None] * n, [None] * n
    done = [False] * n
    left, guard = n, 0
    idq = Quat()
    while left > 0 and guard <= n:
        guard += 1
        for i in range(n):
            if done[i]:
                continue
            par = J[i].parent
            q = quats[i] if quats[i] is not None else idq
            if par < 0:
                rot[i] = q.clone().normalize()
                pos[i] = J[i].pos.clone()
                if root_offset is not None:
                    pos[i].add(root_offset)
                done[i] = True
                left -= 1
            elif done[par]:
                rot[i] = rot[par].clone().multiply(q).normalize()
                pos[i] = J[i].pos.clone().sub(J[par].pos).apply_quaternion(rot[par]).add(pos[par])
                done[i] = True
                left -= 1
    return pos, rot


# ── The frame and chains (computeMotionCtx, ctxInFrame) ──────────────────────
class Chain:
    __slots__ = ("type", "run", "pivot", "side")


class Ctx:
    __slots__ = ("J", "by_type", "U", "F", "L")


def motion_ctx(J):
    """The skeleton's frame (up, facing, left) and its chains by type."""
    chains = skel_chains(J)
    U = _AY.clone()
    F = skel_facing(J, chains)
    L = Vec3().cross_vectors(U, F).normalize()
    l_mn, l_mx, f_sum = math.inf, -math.inf, 0.0
    for j in J:
        l = j.pos.dot(L)
        if l < l_mn:
            l_mn = l
        if l > l_mx:
            l_mx = l
        f_sum += j.pos.dot(F)
    l_cx = (l_mn + l_mx) / 2 if math.isfinite(l_mn) else 0
    by = {}
    for t, run in chains:
        c = Chain()
        c.type, c.run = t, run
        first = J[run[0]] if run[0] < len(J) else None
        c.pivot = first.parent if first is not None else -1
        c.side = 1 if ((first.pos.dot(L) if first is not None else 0) - l_cx) >= 0 else -1
        by.setdefault(t, []).append(c)
    cx = Ctx()
    cx.J, cx.by_type, cx.U, cx.F, cx.L = J, by, U, F, L
    return cx


# ── The body: each module's hull and its points on the shell ────────────────
class BodyModule:
    __slots__ = ("verts", "faces", "pts", "pw", "pj", "joint")


class Body:
    def __init__(self, mods):
        self.mods = mods
        self.touch = {}


def module_body(src, owners):
    """makeSkelBody, weighed as Rigid modules weighs: every module whole to
    the joint that owns it."""
    M = len(src.modules)
    pts = [[] for _ in range(M)]
    seen = [set() for _ in range(M)]
    for f in src.ghost_faces or ():
        m = f.mod
        if m is None or m < 0 or m >= M:
            continue
        for v in (f.a, f.b, f.c):
            k = to_fixed(v.x, 4) + "_" + to_fixed(v.y, 4) + "_" + to_fixed(v.z, 4)
            if k not in seen[m]:
                seen[m].add(k)
                pts[m].append(v)
    out = []
    for i, m in enumerate(src.modules):
        if m.hull is None or len(m.hull[0]) < 4 or not m.hull[1]:
            continue
        j = owners[i] if i < len(owners) and owners[i] is not None else 0
        b = BodyModule()
        b.verts, b.faces = m.hull[0], m.hull[1]
        b.pts = pts[i]
        b.pw = [[(j, 1)] for _ in pts[i]]
        b.pj = [j for _ in pts[i]]
        b.joint = j
        out.append(b)
    return Body(out) if out else None


def _body_hull(m, J, pos, rot):
    j = m.joint
    if j < 0 or j >= len(J) or pos[j] is None:
        return [v.clone() for v in m.verts]
    rj = J[j]
    return [v.clone().sub(rj.pos).apply_quaternion(rot[j]).add(pos[j]) for v in m.verts]


def _posed_points(pts, ws, J, pos, rot):
    out = []
    for v, w in zip(pts, ws):
        o = Vec3()
        tot = 0
        for j, x in w:
            if j < 0 or j >= len(J) or pos[j] is None:
                continue
            lv = v.clone().sub(J[j].pos).apply_quaternion(rot[j]).add(pos[j])
            o.add_scaled_vector(lv, x)
            tot += x
        out.append(o.multiply_scalar(1 / tot) if tot > 0 else v.clone())
    return out


def _hulls_meet(a, fa, b, fb, shrink):
    sa, sb = shrink_verts(a, shrink), shrink_verts(b, shrink)
    if not box_from_points(sa).intersects_box(box_from_points(sb)):
        return False
    return sat_convex(sa, fa, sb, fb)


def _touching_as_built(body, i, k):
    a, b = min(i, k), max(i, k)
    key = a * len(body.mods) + b
    t = body.touch.get(key)
    if t is None:
        if _np is None:
            A, B = body.mods[a], body.mods[b]
            t = _hulls_meet(A.verts, A.faces, B.verts, B.faces, -TOUCH_GROW)
        else:
            ga, gb = _grown(body, a), _grown(body, b)
            t = ga[0].intersects_box(gb[0]) and _np_sat(ga[1], gb[1])
        body.touch[key] = t
    return t


def _grown(body, i):
    """A module's hull as built, grown by TOUCH_GROW: its box and its arrays."""
    g = getattr(body, "grown", None)
    if g is None:
        g = body.grown = {}
    if i not in g:
        m = body.mods[i]
        v = shrink_verts(m.verts, -TOUCH_GROW)
        g[i] = (box_from_points(v), _np_hull(Hull(v, m.faces)))
    return g[i]


# ── Aiming a limb ────────────────────────────────────────────────────────────
def neutral_dir(t, side, cx, out=ARM_OUT):
    if t == "arm":
        return cx.U.clone().multiply_scalar(-jsmath.cos(out)).add_scaled_vector(cx.L, side * jsmath.sin(out))
    if t == "wing":
        return cx.L.clone().multiply_scalar(side)
    return cx.U.clone().negate()


def arc_to(a, b, about):
    if a.dot(b) < -0.9999:
        ax = about.clone().add_scaled_vector(a, -about.dot(a))
        if ax.length_sq() < 1e-9:
            ax.cross_vectors(a, Vec3(1, 0, 0) if abs(a.x) < 0.9 else _AY)
        return Quat().set_from_axis_angle(ax.normalize(), math.pi)
    return Quat().set_from_unit_vectors(a, b)


def aim_limb(J, pose, path, end, want, about, keep_foot):
    """aimLimb: every bone from path[0] to path[end] aimed at `want`, written
    into `pose`; with keep_foot what is past path[end] keeps its world turn."""
    foot = fk_pose(J, pose)[1][path[end]].clone() if keep_foot and end < len(path) - 1 else None
    did = False
    for k in range(end):
        j = path[k]
        pos, rot = fk_pose(J, pose)
        d = pos[path[k + 1]].clone().sub(pos[j])
        if d.length_sq() < 1e-12:
            continue
        d.normalize()
        if d.dot(want) > 1 - 1e-10:
            continue
        pose[j] = _invert(rot[J[j].parent]).multiply(arc_to(d, want, about)).multiply(rot[j])
        did = True
    if did and foot is not None:
        a = path[end]
        rot = fk_pose(J, pose)[1]
        pose[a] = _invert(rot[J[a].parent]).multiply(foot)
    return did


def _kids(J):
    n = len(J)
    kids = [[] for _ in J]
    for i, j in enumerate(J):
        if 0 <= j.parent < n:
            kids[j.parent].append(i)
    return kids


def _below(kids, j):
    out = []
    stack = list(kids[j])
    while stack:
        k = stack.pop()
        out.append(k)
        stack.extend(kids[k])
    return out


class LimbPlan:
    __slots__ = ("type", "c", "aim", "path", "end", "side")


def limb_plans(J, cx):
    """limbPlans: the limbs the neutral pose puts in place, the arms last."""
    n = len(J)
    kids = _kids(J)
    height = lambda p: p.dot(cx.U)
    joint_floor = min(height(j.pos) for j in J) if n else 0

    def depth(j):
        d, k = 0, j
        while k >= 0 and d <= n:
            d += 1
            k = J[k].parent
        return d
    out = []
    for t in NEUTRAL_TYPES:
        for c in cx.by_type.get(t, ()):
            aim = c.pivot if c.pivot >= 0 and len(kids[c.pivot]) == 1 else c.run[0]
            if aim is None or aim < 0 or J[aim].parent < 0:
                continue
            own = set(c.run)
            if any(k not in own and J[k].type in BODY_TYPES for k in _below(kids, aim)):
                continue
            path = [aim] + list(c.run) if aim == c.pivot else list(c.run)
            if len(path) < 2:
                continue
            end = len(path) - 1
            if t in ("leg", "paw"):
                ln = 0.0
                for k in range(1, len(path)):
                    ln += J[path[k]].pos.distance_to(J[path[k - 1]].pos)
                top = height(J[path[0]].pos)
                for k in range(1, len(path)):
                    h = height(J[path[k]].pos)
                    if top - h >= ANKLE_DROP * ln and h - joint_floor <= ANKLE_FRAC * ln:
                        end = k
                        break
            elif t == "arm" and len(c.run) >= 3 and len(path) >= 3:
                end = len(path) - 2
            p = LimbPlan()
            p.type, p.c, p.aim, p.path, p.end, p.side = t, c, aim, path, end, c.side
            out.append(p)
    return sorted(out, key=lambda p: ((1 if p.type == "arm" else 0), depth(p.aim)))


class Neutral:
    __slots__ = ("pose", "world", "lift", "turned", "raised", "kept", "floor", "arms", "active", "cx")


def neutral_pose(J, body=None, cx=None):
    """computeNeutral: the neutral pose of skeleton J (joints with .pos,
    .parent, .type), with its body (module_body) to hang the arms clear of and
    to stand on its lowest corner."""
    cx = cx or motion_ctx(J)
    n = len(J)
    pose = [None] * n
    kids = _kids(J)
    height = lambda p: p.dot(cx.U)
    all_m = body.mods if body else []
    mods = [m for m in all_m if 0 <= m.joint < n]
    joint_floor = min(height(j.pos) for j in J) if n else 0
    drawn = [m for m in mods if m.pts]
    floor = min(height(p) for m in drawn for p in m.pts) if drawn else joint_floor

    plans = limb_plans(J, cx)
    turned = raised = kept = 0

    arm_moves = set()
    for p in plans:
        if p.type == "arm":
            arm_moves.add(p.aim)
            arm_moves.update(_below(kids, p.aim))

    arms = []
    jobs = []
    for p in plans:
        about = cx.F if p.type == "arm" else cx.U if p.type == "wing" else cx.L
        keep_foot = p.type in ("leg", "paw")
        if p.type != "arm" or not mods:
            if aim_limb(J, pose, p.path, p.end, neutral_dir(p.type, p.c.side, cx), about, keep_foot):
                turned += 1
            continue
        jobs.append({"c": p.c, "aim": p.aim, "path": p.path, "end": p.end, "about": about, "res": None})

    in_mods = set(id(m) for m in mods)
    others = [i for i, m in enumerate(all_m) if id(m) in in_mods and m.joint not in arm_moves]

    # hullsMeet over every pair, each hull shrunk once and its SAT hull made
    # once (the same verdicts). The body's modules no arm moves stand still
    # through every arm's trials (an arm's joints carry only the arm), so
    # they are posed once.
    class _Posed:
        __slots__ = ("verts", "faces", "box", "_hull")

        def __init__(self, verts, faces):
            self.verts = shrink_verts(verts, ISECT_SHRINK)
            self.faces = faces
            self.box = box_from_points(self.verts)
            self._hull = None

        def hull(self):
            if self._hull is None:
                h = Hull(self.verts, self.faces)
                self._hull = _np_hull(h) if _np is not None else h
            return self._hull
    still = {}

    def hits_of(mine, p):
        pos, rot = fk_pose(J, p)
        A = [_Posed(_body_hull(all_m[i], J, pos, rot), all_m[i].faces) for i in mine]
        if not still:
            p0, r0 = fk_pose(J, pose)
            for k in others:
                still[k] = _Posed(_body_hull(all_m[k], J, p0, r0), all_m[k].faces)
        hits = 0
        for a, i in enumerate(mine):
            for k in others:
                if _touching_as_built(body, i, k):
                    continue
                b = still[k]
                if A[a].box.intersects_box(b.box) and (_np_sat if _np is not None else sat_hulls)(A[a].hull(), b.hull()):
                    hits += 1
        return hits

    def mine_of(aim):
        moves = {aim}
        moves.update(_below(kids, aim))
        return [i for i, m in enumerate(all_m) if id(m) in in_mods and m.joint in moves]

    def place_arm(job):
        c, path, end, about = job["c"], job["path"], job["end"], job["about"]
        mine = mine_of(job["aim"])
        at = lambda p: hits_of(mine, p) if mine else 0
        p0 = fk_pose(J, pose)[0]
        reach = p0[path[end]].clone().sub(p0[path[0]])
        built = _angle_to(reach.normalize(), cx.U.clone().negate()) if reach.length_sq() > 1e-12 else ARM_OUT

        def try_at(out):
            p = list(pose)
            did = aim_limb(J, p, path, end, neutral_dir("arm", c.side, cx, out), about, False)
            return {"p": p, "did": did, "hits": at(p), "out": out}
        pick = best = None
        out = ARM_OUT
        while out < built - 1e-9 and out <= ARM_MAX + 1e-9:
            t = try_at(out)
            if not t["hits"]:
                pick = t
                break
            if best is None or t["hits"] < best["hits"]:
                best = t
            out += ARM_STEP
        if pick is None:
            as_built = {"p": pose, "did": False, "hits": at(pose), "out": built}
            if not as_built["hits"]:
                pick = as_built
            else:
                if best is None or as_built["hits"] < best["hits"]:
                    best = as_built
                out = max(built, ARM_OUT) + ARM_STEP
                while out <= ARM_MAX + 1e-9:
                    t = try_at(out)
                    if not t["hits"]:
                        pick = t
                        break
                    if t["hits"] < best["hits"]:
                        best = t
                    out += ARM_STEP
                pick = pick or best
        if pick["p"] is not pose:
            pose[:] = pick["p"]
            pick["p"] = pose
        job["res"] = pick

    if all_m:
        for job in jobs:
            place_arm(job)
        for _ in range(2):
            again = False
            for job in jobs:
                mine = mine_of(job["aim"])
                if not mine or not hits_of(mine, pose):
                    continue
                for k in range(job["end"]):
                    pose[job["path"][k]] = None
                place_arm(job)
                again = True
            if not again:
                break
        for job in jobs:
            r = job["res"]
            mine = mine_of(job["aim"])
            if r["did"]:
                turned += 1
                if r["out"] > ARM_OUT + 1e-9:
                    raised += 1
            else:
                kept += 1
            arms.append((job["aim"], r["out"], hits_of(mine, pose) if mine else 0))

    active = any(q is not None for q in pose)
    pos, rot = fk_pose(J, pose)
    pts = []
    if drawn:
        for m in drawn:
            pts.extend(_posed_points(m.pts, m.pw, J, pos, rot))
    else:
        pts = list(pos)
    low = min(height(p) for p in pts)
    nx = Neutral()
    nx.pose, nx.world, nx.turned, nx.raised, nx.kept = pose, rot, turned, raised, kept
    nx.floor, nx.arms, nx.active, nx.cx = floor, arms, active, cx
    nx.lift = cx.U.clone().multiply_scalar(floor - low)
    return nx
