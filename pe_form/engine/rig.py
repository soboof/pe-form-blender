"""The Theater's auto rig, ported from pe-web: engine/rig.ts (rigFromEngine,
skelFromRig, skelChains, the facing vote, sideName), autoType.ts
(autoTypeSkeleton, rigByPlan), bodyPlan.ts (the plans), limbRig.ts
(rerigLimbs: limbs typed again, fingers found, limb joints onto their
seams), skelEdit.ts (seamHinges) and moduleSkin.ts (Rigid modules: which
joint owns each module, the seam caps, a weld that never joins two modules).

A skeleton is one joint per module centre, parented as the formation was
built; a joint's `type` names the bone to its parent (its chain). Everything
is in the engine's own space (Y up), as built. Line for line with the web, in
its order of operations (V8's atan2, sin and cos where it uses them), so the
same formation gets the same rig: tests/test_rig.py holds it to the Theater's
answers on the demo pack's formations.

Pure Python, no bpy.
"""

import math
import re

from . import jsmath
from .vec import Vec3, Box3
from .ground import clean_ground, ground_matrix

# ── Body plans (bodyPlan.ts) ─────────────────────────────────────────────────
BODY_PLANS = [
    ("auto", "Read from its shape",
     "No plan: legs, arms, a spine, a head and a tail guessed from the shape alone."),
    ("biped", "Two legs",
     "Stands upright: the limbs reaching down are legs, the ones at the sides arms on shoulders, "
     "a spine up the middle and a head on top."),
    ("quadruped", "Four legs",
     "Lies along the ground: every limb reaching down is a leg (Rigify's animal leg), a spine along "
     "the body, a head at the high end and a tail at the other."),
    ("bird", "Bird", "The limbs reaching down are legs, the ones at the sides wings; a spine, a head and a tail."),
    ("insect", "Flying insect",
     "Legs below, as many as it has; wings at the sides and above; a head, and the abdomen as a tail."),
    ("snake", "Snake", "Its whole length one tail chain, which slithers. Nothing else bends."),
    ("column", "Column", "Rooted at its foot, its height one spine chain: it sways, and nothing else bends."),
    ("rigid", "Rigid", "One joint at its foot: it moves as a whole and never bends (walls, floors, anything built)."),
]
_PLAN_LABEL = {p: label for p, label, _ in BODY_PLANS}

LYING = {"quadruped", "bird", "insect", "snake"}
STANDING = {"biped", "column"}

TYPE_COL = {"spine": 0xe8c547, "shoulder": 0xb0d152, "arm": 0x4caf7d, "leg": 0x4ca8d8, "tail": 0xc44444,
            "head": 0x9e5cdb, "finger": 0xd8884c, "paw": 0x3aa6a0, "robot": 0x8a93a6, "wing": 0xe87fa0,
            "eye": 0xd94fd0}


def plan_label(p):
    return _PLAN_LABEL.get(p, BODY_PLANS[0][1])


def js_round(x):
    return math.floor(x + 0.5)


# ── Sources and skeletons ────────────────────────────────────────────────────
class RigModule:
    __slots__ = ("idx", "mod_id", "center", "parent", "conn_cen", "seam", "color", "hull")

    def __init__(self, idx, mod_id, center, parent):
        self.idx, self.mod_id, self.center, self.parent = idx, mod_id, center, parent
        self.conn_cen = None
        self.seam = None
        self.color = None
        self.hull = None     # (verts, faces): the module's hull as placed, whole


class RigFace:
    __slots__ = ("a", "b", "c", "mod", "color")

    def __init__(self, a, b, c, mod=None, color=None):
        self.a, self.b, self.c, self.mod, self.color = a, b, c, mod, color


class RigSource:
    __slots__ = ("name", "X", "modules", "ghost_faces", "stand")

    def __init__(self, name, X, modules, ghost_faces=None):
        self.name, self.X, self.modules, self.ghost_faces = name, X, modules, ghost_faces or []
        self.stand = None   # the turn and shift that stood it on its ground, if it has one


class Joint:
    __slots__ = ("name", "pos", "parent", "type")

    def __init__(self, name, pos, parent, type=""):
        self.name, self.pos, self.parent, self.type = name, pos, parent, type

    def copy(self, **kw):
        j = Joint(self.name, self.pos.clone(), self.parent, self.type)
        for k, v in kw.items():
            setattr(j, k, v)
        return j

    def __repr__(self):
        return "Joint(%r, %r, %r)" % (self.name, self.parent, self.type)


def rig_from_engine(engine, name="formation", stand=True):
    """rigFromEngine: a module per placed step (carves are cavities, not
    modules), each with its parent, the face it was placed on (where its seam
    is), and the shell's triangles each tagged with its module; standing on
    its ground when the save has one (`stand`), as the Theater rigs it."""
    H = engine.history
    if not H:
        raise ValueError("empty formation")
    kept = [i for i, h in enumerate(H) if not h.carve]
    renum = {i: k for k, i in enumerate(kept)}

    def solid_parent(idx):
        k = idx
        while 0 <= k < len(H) and H[k].carve:
            p = H[k].parent_mod_idx
            k = p if p is not None else -1
        return k

    mods = []
    for k, i in enumerate(kept):
        h = H[i]
        p = renum.get(solid_parent(h.parent_mod_idx if h.parent_mod_idx is not None else -1), -1)
        m = RigModule(k, h.mod_id, h.mod_centroid.clone(), p if p >= 0 else -1)
        m.color = h.color
        f = h.used_face
        if m.parent >= 0 and f is not None and f.center3 is not None:
            m.conn_cen = f.center3.clone()
            plain = not f.conn_type or f.conn_type == "open"
            if plain and len(f.verts3) >= 3:
                m.seam = [v.clone() for v in f.verts3]
        mods.append(m)
    ghost = []
    for f in engine.all_faces:
        k = None
        if f.mod_idx is not None and f.mod_idx >= 0:
            k = renum.get(solid_parent(f.mod_idx))
        ghost.append(RigFace(f.va.clone(), f.vb.clone(), f.vc.clone(), k, f.color))
    # Each module's hull as placed, whole (the shell has lost the faces two
    # modules share): what an arm hangs clear of and a foot stands on.
    W = getattr(engine, "module_world_data", None) or []
    for k, i in enumerate(kept):
        w = W[i] if i < len(W) else None
        if w is not None and w.verts:
            mods[k].hull = ([v.clone() for v in w.verts], w.faces)
    src = RigSource(name, engine.X, mods, ghost)
    # Standing on the face its maker chose in the web Form (a save's ground):
    # the joints and the shell turned and moved so that face is the floor.
    m = ground_matrix(clean_ground(engine.extras.get("ground"))) if stand else None
    if m is not None:
        for mod in mods:
            mod.center = m.through(mod.center)
            if mod.conn_cen is not None:
                mod.conn_cen = m.through(mod.conn_cen)
            if mod.seam:
                mod.seam = [m.through(v) for v in mod.seam]
            if mod.hull is not None:
                mod.hull = ([m.through(v) for v in mod.hull[0]], mod.hull[1])
        for f in ghost:
            m.apply(f.a)
            m.apply(f.b)
            m.apply(f.c)
    src.stand = m
    return src


def skel_from_rig(rig):
    out = []
    for i, m in enumerate(rig.modules):
        s = re.sub(r"[^a-z0-9]+", "", m.mod_id or "mod", flags=re.I).lower() or "mod"
        out.append(Joint(s + "_" + ("0" + str(i) if i < 10 else str(i)), m.center.clone(),
                         -1 if m.parent < 0 else m.parent, ""))
    return out


def skel_x_span(J):
    mn, mx = math.inf, -math.inf
    for j in J:
        if j.pos.x < mn:
            mn = j.pos.x
        if j.pos.x > mx:
            mx = j.pos.x
    if not math.isfinite(mn):
        mn = mx = 0
    return (mn + mx) / 2, max(mx - mn, 1e-6)


def side_name(nm, rel_x, width):
    if re.search(r"\.(L|R)$", nm):
        return nm
    t = width * 0.04
    if rel_x > t:
        return nm + ".L"
    if rel_x < -t:
        return nm + ".R"
    return nm


def reroot_skeleton(J, i):
    out = [j.copy() for j in J]
    if i >= len(out) or out[i].parent < 0:
        return out
    cur, new_parent = i, -1
    guard = 0
    while cur >= 0 and guard < len(out):
        old = out[cur].parent
        out[cur].parent = new_parent
        new_parent = cur
        cur = old
        guard += 1
    return out


def skel_chains(J):
    """(type, run) per chain: a same-type run from its first bone, following
    the longest same-type branch at a fork."""
    kids = {}
    for i, j in enumerate(J):
        if j.parent >= 0:
            kids.setdefault(j.parent, []).append(i)

    def is_bone(i):
        return 0 <= i < len(J) and J[i].parent >= 0

    memo = {}

    def depth_of(i, t):
        key = (i, t)
        if key in memo:
            return memo[key]
        best = 0
        for c in kids.get(i, ()):
            if J[c].type == t:
                best = max(best, 1 + depth_of(c, t))
        memo[key] = best
        return best

    chains = []
    for i, j in enumerate(J):
        if not is_bone(i) or not j.type:
            continue
        pt = J[j.parent].type if is_bone(j.parent) else ""
        if pt == j.type:
            continue
        run = [i]
        cur = i
        while True:
            ks = [c for c in kids.get(cur, ()) if J[c].type == j.type]
            if not ks:
                break
            pick, pd = ks[0], -1
            for c in ks:
                d = depth_of(c, j.type)
                if d > pd:
                    pd, pick = d, c
            cur = pick
            run.append(cur)
        chains.append((j.type, run))
    return chains


# ── Facing: a vote of what says which way is ahead (rig.ts) ─────────────────
_FACING_WEIGHT = {"eyes": 3, "head": 2, "tail": 1.5, "spine": 1}


def _chain_dir(J, run):
    first = J[run[0]] if run and run[0] < len(J) else None
    head = J[first.parent].pos if first is not None and first.parent >= 0 else (first.pos if first else None)
    tip = J[run[-1]].pos if run[-1] < len(J) else None
    d = tip.clone().sub(head) if tip is not None and head is not None else Vec3()
    return Vec3(0, 1, 0) if d.length_sq() < 1e-12 else d.normalize()


def facing_evidence(J, chains):
    out = []

    def level(v):
        return Vec3(v.x, 0, v.z)

    if not J:
        return out
    box = Box3()
    for j in J:
        box.expand_by_point(j.pos)
    size = Vec3().sub_vectors(box.max, box.min)
    span = max(1e-9, size.length())
    spine_j = [i for t, run in chains if t == "spine" for i in run]
    mid = Vec3()
    for i in (spine_j if spine_j else range(len(J))):
        mid.add(J[i].pos)
    mid.multiply_scalar(1 / (len(spine_j) or len(J)))

    eyes = [(j, i) for i, j in enumerate(J) if j.type == "eye" and j.parent >= 0]
    if eyes:
        d = Vec3()
        for j, _ in eyes:
            d.add(level(j.pos.clone().sub(J[j.parent].pos)))
        if d.length() > 1e-6 * span:
            out.append(("eyes", d.normalize(), _FACING_WEIGHT["eyes"], eyes[0][1]))

    def furthest(type_):
        best, bd = -1, 0.15 * span
        for t, run in chains:
            if t != type_:
                continue
            for i in run:
                d = level(J[i].pos.clone().sub(mid)).length()
                if d > bd:
                    bd, best = d, i
        return best

    head = furthest("head")
    if head >= 0:
        out.append(("head", level(J[head].pos.clone().sub(mid)).normalize(), _FACING_WEIGHT["head"], head))
    tail = furthest("tail")
    if tail >= 0:
        out.append(("tail", level(mid.clone().sub(J[tail].pos)).normalize(), _FACING_WEIGHT["tail"], tail))
    spine_dir = Vec3()
    for t, run in chains:
        if t == "spine":
            spine_dir.add(_chain_dir(J, run))
    if spine_dir.length_sq() > 1e-9:
        h = level(spine_dir.normalize())
        if h.length() > 0.35:
            first = next(run for t, run in chains if t == "spine")
            out.append(("spine", h.normalize(), _FACING_WEIGHT["spine"], first[0]))
    return out


def skel_facing(J, chains):
    ev = facing_evidence(J, chains)
    spine = next((e for e in ev if e[0] == "spine"), None)
    vote = Vec3()
    for e in ev:
        if e is not spine:
            vote.add_scaled_vector(e[1], e[2])
    if spine is not None:
        s = vote.dot(spine[1])
        return spine[1].clone().multiply_scalar(-1 if s < -1e-9 else 1)
    if vote.length() > 1e-6:
        return vote.normalize()
    return Vec3(0, 0, 1)


# ── Auto-type (autoType.ts) ──────────────────────────────────────────────────
def _adjacency(J):
    adj = [[] for _ in J]
    for i, j in enumerate(J):
        if 0 <= j.parent < len(J) and j.parent != i:
            adj[i].append(j.parent)
            adj[j.parent].append(i)
    return adj


def _bfs_prev(adj, start, blocked):
    prev = {}
    seen = {start}
    queue = [start]
    while queue:
        nxt = []
        for cur in queue:
            for nb in adj[cur]:
                if nb in seen or nb in blocked:
                    continue
                seen.add(nb)
                prev[nb] = cur
                nxt.append(nb)
        queue = nxt
    return prev


def _path_to(prev, start, end):
    if start == end:
        return [start]
    if end not in prev:
        return []
    out = [end]
    c = end
    while c in prev:
        c = prev[c]
        out.append(c)
    return out[::-1] if out[-1] == start else []


def _farthest_path(adj, start, blocked):
    prev = _bfs_prev(adj, start, blocked)
    best, best_d = start, -1

    def depth(i):
        d = 0
        c = i
        while c in prev:
            c = prev[c]
            d += 1
        return d

    for node in prev:
        d = depth(node)
        if d > best_d:
            best_d, best = d, node
    return _path_to(prev, start, best)


def _largest_component(adj):
    seen = set()
    best, count = [], 0
    for i in range(len(adj)):
        if i in seen:
            continue
        count += 1
        comp = []
        stack = [i]
        seen.add(i)
        while stack:
            c = stack.pop()
            comp.append(c)
            for nb in adj[c]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(comp) > len(best):
            best = comp
    return best, count


_AY, _AZ, _AX = Vec3(0, 1, 0), Vec3(0, 0, 1), Vec3(1, 0, 0)


def _principal_horizontal(J, up):
    e1 = (_AZ if abs(up.dot(_AZ)) < 0.9 else _AX).clone()
    e1.add_scaled_vector(up, -e1.dot(up))
    if e1.length_sq() < 1e-12:
        return _AZ.clone()
    e1.normalize()
    e2 = Vec3().cross_vectors(up, e1).normalize()
    mu = mv = 0
    pts = []
    for j in J:
        u, v = j.pos.dot(e1), j.pos.dot(e2)
        mu += u
        mv += v
        pts.append((u, v))
    n = len(pts) or 1
    mu /= n
    mv /= n
    suu = svv = suv = 0
    for u, v in pts:
        du, dv = u - mu, v - mv
        suu += du * du
        svv += dv * dv
        suv += du * dv
    if suu + svv < 1e-12:
        return e1
    ang = 0.5 * jsmath.atan2(2 * suv, suu - svv)
    return e1.clone().multiply_scalar(jsmath.cos(ang)).add_scaled_vector(e2, jsmath.sin(ang)).normalize()


def _best_extent_path(J, adj, start, blocked, d):
    prev = _bfs_prev(adj, start, blocked)
    frm = J[start].pos
    best, best_s = start, 0
    for node in prev:
        s = abs(J[node].pos.clone().sub(frm).dot(d))
        if s > best_s:
            best_s, best = s, node
    return _path_to(prev, start, best)


def _extend_path(J, adj, core, up, fwd):
    if len(core) < 2:
        return core
    head, tail = core[-1], core[0]
    core_dir = J[head].pos.clone().sub(J[tail].pos)
    ext = up if abs(core_dir.dot(up)) > abs(core_dir.dot(fwd)) else fwd
    back = _best_extent_path(J, adj, tail, {x for x in core if x != tail}, ext)
    front = _best_extent_path(J, adj, head, {x for x in core if x != head}, ext)
    return back[1:][::-1] + core + front[1:]


def _ek(a, b):
    return (a, b) if a < b else (b, a)


class AutoTypeReport:
    def __init__(self, frame="derived"):
        self.frame = frame
        self.added = []      # (type, [joints])
        self.preserved = 0
        self.untyped = []
        self.notes = []


def auto_type_skeleton(J0, plan="auto", guide=None):
    """autoTypeSkeleton: fills chain types (never overwrites one), read along
    the body plan. Returns (joints, report)."""
    n = len(J0)
    joints = [j.copy() for j in J0]
    report = AutoTypeReport()

    def finish():
        report.untyped = [i for i, j in enumerate(joints) if not j.type]
        return joints, report

    if n < 2:
        report.notes.append("Fewer than 2 joints; nothing to classify.")
        return finish()
    J = J0
    adj = _adjacency(J)
    comp, comp_count = _largest_component(adj)
    if comp_count > 1:
        report.notes.append("%d disconnected pieces; typed the largest (%d joints)." % (comp_count, len(comp)))
    if len(comp) < 2:
        report.notes.append("No connected pair of joints to classify.")
        return finish()

    # 1. Frame.
    existing = skel_chains(J)
    spine_chain = next((c for c in existing if c[0] == "spine"), None)
    has_axis_chain = any(c[0] in ("spine", "tail") for c in existing)
    up = (guide[0].clone() if guide else _AY.clone()).normalize()
    if guide:
        fwd = guide[1].clone()
        report.frame = "guide"
    elif has_axis_chain:
        fwd = skel_facing(J, existing)
        report.frame = "chains"
    else:
        fwd = _principal_horizontal(J, up)
        report.frame = "derived"
    fwd.add_scaled_vector(up, -fwd.dot(up))
    if fwd.length_sq() < 1e-9:
        fwd = _AZ.clone() if abs(up.dot(_AZ)) < 0.9 else _AX.clone()
    fwd.normalize()

    # 2. Axis.
    if spine_chain and spine_chain[1]:
        run = spine_chain[1]
        pivot = J[run[0]].parent if run[0] < n else -1
        axis = _extend_path(J, adj, ([pivot] + list(run)) if pivot >= 0 else list(run), up, fwd)
        report.notes.append("Axis taken from the existing spine chain.")
    else:
        leaves = [i for i in comp if len(adj[i]) <= 1]
        ends = leaves if len(leaves) >= 2 else comp
        best_score = -1
        axis = []
        none = set()
        for a in ends:
            prev = _bfs_prev(adj, a, none)
            for b in ends:
                if b <= a:
                    continue
                d = J[b].pos.clone().sub(J[a].pos)
                if plan in STANDING:
                    score = abs(d.dot(up))
                elif plan in LYING:
                    score = abs(d.dot(fwd))
                else:
                    score = max(abs(d.dot(fwd)), abs(d.dot(up)))
                if score <= best_score:
                    continue
                p = _path_to(prev, a, b)
                if len(p) < 2:
                    continue
                best_score, axis = score, p
    if len(axis) < 2:
        report.notes.append("Could not find a body axis.")
        return finish()

    # Tail to head.
    axis_vec = J[axis[-1]].pos.clone().sub(J[axis[0]].pos)
    if plan in STANDING:
        along_up = True
    elif plan in LYING:
        along_up = False
    else:
        along_up = abs(axis_vec.dot(up)) > abs(axis_vec.dot(fwd))
    headward = up if along_up else fwd
    if plan in LYING and not spine_chain:
        dy = J[axis[-1]].pos.dot(up) - J[axis[0]].pos.dot(up)
        reach = axis_vec.length()
        if abs(dy) > reach * 0.04:
            if dy < 0:
                axis.reverse()
        elif J[axis[0]].pos.dot(headward) > J[axis[-1]].pos.dot(headward):
            axis.reverse()
    elif J[axis[0]].pos.dot(headward) > J[axis[-1]].pos.dot(headward):
        axis.reverse()

    # 3. Branch points split the axis.
    axis_set = set(axis)
    branch_at = [k for k, j in enumerate(axis) if any(nb not in axis_set for nb in adj[j])]
    last_k = len(axis) - 1
    no_limbs = not branch_at
    if len(branch_at) >= 2:
        first, last = branch_at[0], branch_at[-1]
    elif len(branch_at) == 1:
        first, last = max(0, branch_at[0] - 1), min(last_k, branch_at[0] + 1)
        report.notes.append("One branch point; torso taken as the edges either side of it.")
    else:
        first, last = 0, last_k
        report.notes.append("No limbs found; typed the body as a tail chain.")

    torso = J[axis[last]].pos.clone().sub(J[axis[first]].pos)
    upright = torso.length_sq() > 1e-9 and abs(torso.clone().normalize().dot(up)) > 0.6

    def limb_type(d):
        dn = -d.dot(up)
        if plan == "biped":
            return "leg" if dn > 0.4 else "arm"
        if plan == "quadruped":
            return "paw" if dn > 0.2 else ""
        if plan == "bird":
            return "leg" if dn > 0.45 else "wing"
        if plan == "insect":
            return "leg" if dn > 0.3 else "wing"
        if plan in ("snake", "column", "rigid"):
            return ""
        return ("leg" if upright else "paw") if dn > 0.5 else "arm"

    edge_type = {}

    def mark(a, b, t):
        if t:
            edge_type[_ek(a, b)] = t

    def mark_run(nodes, t):
        for i in range(len(nodes) - 1):
            mark(nodes[i], nodes[i + 1], t)

    whole_axis = "tail" if plan == "snake" else "spine" if plan == "column" else None
    if whole_axis:
        mark_run(axis, whole_axis)
    elif no_limbs:
        if plan == "auto":
            mark_run(axis, "tail")
        else:
            mark_run(axis, "spine")
            report.notes.append("No limbs found for %s: the body is one spine." % plan_label(plan).lower())
    else:
        for k in range(first, last):
            mark(axis[k], axis[k + 1], "spine")

    # 4. The axis' two end spans.
    if not no_limbs and not whole_axis:
        def end_span(frm, to, fallback):
            if frm == to:
                return
            nodes = axis[frm:to + 1] if frm < to else axis[to:frm + 1][::-1]
            d = J[nodes[-1]].pos.clone().sub(J[nodes[0]].pos)
            if d.length_sq() < 1e-12:
                return
            d.normalize()
            if -d.dot(up) > abs(d.dot(fwd)) and limb_type(d):
                mark(nodes[0], nodes[1], "shoulder")
                mark_run(nodes[1:], limb_type(d))
            else:
                mark_run(nodes, fallback)
        end_span(last, last_k, "head")
        end_span(first, 0, "tail")

    # 5. Off-axis limbs.
    for k in ([] if whole_axis else branch_at):
        b = axis[k]
        for nb in adj[b]:
            if nb in axis_set:
                continue
            limb = _farthest_path(adj, nb, {b})
            d = J[limb[-1]].pos.clone().sub(J[b].pos)
            if d.length_sq() < 1e-12:
                continue
            d.normalize()
            lt = limb_type(d)
            if not lt:
                continue
            mark(b, nb, "shoulder")
            mark_run(limb, lt)

    # 6. Types onto whichever end is the child.
    added = {}
    for i in range(n):
        p = joints[i].parent
        if p < 0 or p >= n:
            continue
        if joints[i].type:
            report.preserved += 1
            continue
        t = edge_type.get(_ek(i, p))
        if not t:
            continue
        joints[i].type = t
        added.setdefault(t, []).append(i)
    report.added = list(added.items())
    return finish()


def auto_type_summary(r):
    if not r.added:
        return "Auto-type: nothing to add" + ((" (%d kept)" % r.preserved) if r.preserved else "")
    parts = ["%d %s" % (len(js), t) for t, js in sorted(r.added, key=lambda a: -len(a[1]))]
    return "Auto-type (%s frame): %s%s" % (r.frame, ", ".join(parts), (", %d kept" % r.preserved) if r.preserved else "")


def rig_by_plan(J, plan, guide=None):
    """rigByPlan: from scratch, as the body plan says (every type cleared
    first). Rigid is one joint at the foot; a column is rooted at its lowest
    joint; the rest are auto-typed by the plan."""
    if not J:
        r = AutoTypeReport("guide" if guide else "derived")
        r.notes = ["No joints."]
        return [], r
    up = (guide[0].clone() if guide else Vec3(0, 1, 0)).normalize()
    if plan == "rigid":
        mid = Vec3()
        for j in J:
            mid.add(j.pos)
        mid.divide_scalar(len(J))
        low = min(j.pos.dot(up) for j in J)
        pos = mid.add_scaled_vector(up, low - mid.dot(up))
        r = AutoTypeReport("guide" if guide else "derived")
        r.untyped = [0]
        r.notes = ["Rigid: one joint at its foot, so it moves as a whole."]
        return [Joint("root", pos, -1, "")], r
    base = [j.copy(type="") for j in J]
    if plan == "column":
        lowest = 0
        for i, j in enumerate(J):
            if j.pos.dot(up) < J[lowest].pos.dot(up):
                lowest = i
        base = reroot_skeleton(base, lowest)
    return auto_type_skeleton(base, plan, guide)


# ── Seam hinges (skelEdit.ts) and Re-rig limbs (limbRig.ts) ─────────────────
def seam_hinges(J, mods, E, only=None):
    """Each joint at a module centre onto the face its module meets its
    parent joint's module on. (joints or None, moved, error)."""
    near = E * 0.02
    mod_of = []
    for j in J:
        best, bd = -1, near
        for k, m in enumerate(mods):
            d = m.center.distance_to(j.pos)
            if d <= bd:
                bd, best = d, k
        mod_of.append(best)
    out = [j.copy() for j in J]
    moved = 0
    for i, j in enumerate(J):
        if j.parent < 0 or (only is not None and not only(i)):
            continue
        m, p = mod_of[i], mod_of[j.parent]
        if m < 0 or p < 0 or m == p:
            continue
        at = mods[m].conn_cen if mods[m].parent == p else mods[p].conn_cen if mods[p].parent == m else None
        if at is None or at.distance_to(j.pos) < 1e-9:
            continue
        out[i].pos = at.clone()
        moved += 1
    if not moved:
        if any(j.parent >= 0 and mod_of[i] >= 0 for i, j in enumerate(J)):
            return None, 0, "No joint has a seam to move to: each sits in a module that is not next to its parent's."
        return None, 0, ("No joint sits at a module's centre: they are on their seams already, "
                         "or the skeleton is not the modules'.")
    return out, moved, None


LIMB_TYPES = {"shoulder", "arm", "leg", "paw", "wing"}
BENDING = {"arm", "leg", "paw", "wing", "finger"}
HANDS = {"arm", "wing"}


def _longest_down(kids, start):
    best = []
    for c in kids[start]:
        p = _longest_down(kids, c)
        if len(p) > len(best):
            best = p
    return [start] + best


def _type_fingers(joints):
    kids = [[] for _ in joints]
    for i, j in enumerate(joints):
        if j.parent >= 0:
            kids[j.parent].append(i)
    made = 0
    for t, run in skel_chains(joints):
        if t not in HANDS:
            continue
        wrist = -1
        for k in range(len(run) - 1, 0, -1):
            if len(kids[run[k]]) >= 2:
                wrist = k
                break
        if wrist < 0:
            continue
        w = run[wrist]
        branches = [p for p in (_longest_down(kids, b) for b in kids[w]) if len(p) >= 2]
        if len(branches) < 2:
            continue
        below = list(kids[w])
        while below:
            j = below.pop()
            if joints[j].type in HANDS:
                joints[j].type = ""
            below.extend(kids[j])
        for p in branches:
            for j in p:
                joints[j].type = "finger"
                made += 1
    return made


def rerig_limbs(J, plan, mods, E, guide=None, hinge=True):
    """rerigLimbs: the limbs typed again by the plan (fingers found on every
    hand), and each bending limb joint moved onto its seam. Returns
    (joints, report dict)."""
    report = {"retyped": 0, "limbs": 0, "fingers": 0, "hinged": 0}
    if plan in ("rigid", "column", "snake"):
        report["note"] = "This body plan has no limbs; nothing changed."
        return J, report

    def hand_finger(i):
        p = J[i].parent
        while p >= 0 and J[p].type == "finger":
            p = J[p].parent
        return p >= 0 and J[p].type in HANDS

    cleared = []
    for i, j in enumerate(J):
        clear = bool(j.type) and (j.type in LIMB_TYPES or (j.type == "finger" and hand_finger(i)))
        if clear:
            report["retyped"] += 1
        cleared.append(j.copy(type="" if clear else j.type))
    nxt, _ = auto_type_skeleton(cleared, plan, guide)
    _type_fingers(nxt)
    report["limbs"] = sum(1 for j in nxt if j.type and j.type in LIMB_TYPES)
    report["fingers"] = sum(1 for j in nxt if j.type == "finger")
    if hinge:
        hinged, moved, _ = seam_hinges(nxt, mods, E, lambda i: bool(nxt[i].type) and nxt[i].type in BENDING)
        if hinged is not None:
            nxt = hinged
            report["hinged"] = moved
    return nxt, report


# ── Rigid modules (moduleSkin.ts) ────────────────────────────────────────────
def _normal_of(a, b, c):
    n = Vec3().cross_vectors(Vec3().sub_vectors(b, a), Vec3().sub_vectors(c, a))
    ln = n.length()
    return n.multiply_scalar(1 / ln) if ln > 1e-20 else Vec3()


def _keyer(tol):
    inv = 1 / tol

    def pt(p):
        return (js_round(p.x * inv), js_round(p.y * inv), js_round(p.z * inv))
    return pt


def _nearest_module(mods, p):
    best, bd = -1, math.inf
    for m, mod in enumerate(mods):
        d = mod.center.distance_to_squared(p)
        if d < bd:
            bd, best = d, m
    return best


def tri_modules(src, tris, tol):
    """The module of each of `tris`: found among the shell's own triangles by
    centroid and facing, else the module whose centre is nearest."""
    pt = _keyer(tol)
    known = {}
    for f in src.ghost_faces:
        if f.mod is None or f.mod < 0 or f.mod >= len(src.modules):
            continue
        c = Vec3().copy(f.a).add(f.b).add(f.c).multiply_scalar(1 / 3)
        known.setdefault(pt(c), []).append((_normal_of(f.a, f.b, f.c), f.mod))
    out = []
    for a, b, cc in tris:
        c = Vec3().copy(a).add(b).add(cc).multiply_scalar(1 / 3)
        lst = known.get(pt(c))
        if lst:
            n = _normal_of(a, b, cc)
            best, bd = lst[0], -math.inf
            for e in lst:
                d = e[0].dot(n)
                if d > bd:
                    bd, best = d, e
            out.append(best[1])
        else:
            out.append(_nearest_module(src.modules, c))
    return out


def _credited_segments(J, E):
    segs = [[] for _ in J]
    has_kid = [False] * len(J)
    for j in J:
        if 0 <= j.parent < len(J):
            has_kid[j.parent] = True
            segs[j.parent].append((J[j.parent].pos, j.pos))
    for i, j in enumerate(J):
        if has_kid[i]:
            continue
        d = j.pos.clone().sub(J[j.parent].pos) if 0 <= j.parent < len(J) else Vec3(0, 1, 0)
        if d.length_sq() < 1e-12:
            d.set(0, 1, 0)
        segs[i].append((j.pos, j.pos.clone().add_scaled_vector(d.normalize(), E * 0.6)))
    return segs


def module_owners(J, src, tris, tri_mod, E):
    """The joint that owns each module. Its own joint when that lies inside
    it (every skeleton the auto rig makes); otherwise the joint whose bone
    passes nearest the module's centre. (The Theater first narrows that to
    the bones whose geodesic field reaches the module's corners; the auto rig
    never needs it: its one skeleton that leaves rule one, Rigid, has a
    single joint.)"""
    M = len(src.modules)
    owners = [0] * M
    if not J:
        return owners
    radius = [0.0] * M
    for t, m in zip(tris, tri_mod):
        if m < 0 or m >= M:
            continue
        c = src.modules[m].center
        for v in t:
            radius[m] = max(radius[m], v.distance_to(c))
    slack = E * 1e-6
    rest = []
    for m in range(M):
        mod = src.modules[m]
        r = radius[m] or (mod.conn_cen.distance_to(mod.center) if mod.conn_cen is not None else E * 0.6)
        if m < len(J) and J[m].pos.distance_to(mod.center) <= r * (1 + 1e-9) + slack:
            owners[m] = m
        else:
            rest.append(m)
    if not rest:
        return owners
    segs = _credited_segments(J, E)
    for m in rest:
        c = src.modules[m].center
        best, bs = 0, math.inf
        for j in range(len(J)):
            for h, t in segs[j]:
                se = Vec3().sub_vectors(t, h)
                se2 = se.dot(se)
                u = se.dot(Vec3().sub_vectors(c, h)) / se2 if se2 > 0 else 0
                u = max(0, min(1, u))
                cp = se.multiply_scalar(u).add(h)
                s = cp.distance_to(c) + E * 1e-3 * u
                if s < bs:
                    bs, best = s, j
        owners[m] = best
    return owners


def seam_caps(src):
    """Each seam closed on both sides: the face a module was placed on, once
    facing out of its parent and once out of itself. (a, b, c, module, colour)."""
    out = []
    mods = src.modules
    for m, mod in enumerate(mods):
        p, poly = mod.parent, mod.seam
        if p < 0 or p >= len(mods) or not poly or len(poly) < 3:
            continue
        par = mods[p]
        fc = Vec3()
        for v in poly:
            fc.add(v)
        fc.multiply_scalar(1 / len(poly))
        out1 = Vec3().sub_vectors(fc, par.center)
        if out1.length_sq() < 1e-20:
            out1 = Vec3().sub_vectors(mod.center, par.center)
        n = _normal_of(poly[0], poly[1], poly[2])
        flip = n.dot(out1) < 0
        for i in range(1, len(poly) - 1):
            a, b, c = poly[0], poly[i], poly[i + 1]
            out.append((a, c, b, p, par.color) if flip else (a, b, c, p, par.color))
            out.append((a, b, c, m, mod.color) if flip else (a, c, b, m, mod.color))
    return out


def split_weld(tris, tri_mod, tol):
    """A weld that never joins two modules: (vertices, triangles as index
    triples, each vertex's module)."""
    pt = _keyer(tol)
    index = {}
    V, TRI, vmod = [], [], []

    def vid(p, m):
        key = (m, pt(p))
        i = index.get(key)
        if i is None:
            i = len(V)
            index[key] = i
            V.append(p)
            vmod.append(m)
        return i

    for t, m in zip(tris, tri_mod):
        a, b, c = vid(t[0], m), vid(t[1], m), vid(t[2], m)
        if a != b and b != c and a != c:
            TRI.append((a, b, c))
    return V, TRI, vmod


# ── The auto rig: what the Theater does when a formation is sent to it ─────
class AutoRig:
    """The skeleton and its Rigid modules skin for a formation."""

    def __init__(self, src, joints, owners, report, notes):
        self.src = src
        self.joints = joints
        self.owners = owners
        self.report = report
        self.notes = notes


def auto_rig(engine, plan="auto", hinge="limbs", name="formation", stand=True):
    """Rig as the plan (rigByPlan), then Re-rig limbs: fingers found and, with
    `hinge` "limbs", the bending limb joints onto their seams; "all" puts
    every joint on its seam (Hinge at seams), "none" leaves them at the
    module centres."""
    from .geom import get_edge
    src = rig_from_engine(engine, name, stand)
    E = get_edge(src.X)
    J, rep = rig_by_plan(skel_from_rig(src), plan)
    notes = list(rep.notes)
    J, lr = rerig_limbs(J, plan, src.modules, E, hinge=hinge != "none")
    if hinge == "all":
        h, _, _ = seam_hinges(J, src.modules, E)
        if h is not None:
            J = h
    tris = [(f.a, f.b, f.c) for f in src.ghost_faces]
    tri_mod = tri_modules(src, tris, E * 0.015)
    owners = module_owners(J, src, tris, tri_mod, E)
    counts = {}
    for j in J:
        if j.type:
            counts[j.type] = counts.get(j.type, 0) + 1
    return AutoRig(src, J, owners, {"counts": counts, "limbs": lr, "tri_mod": tri_mod}, notes)
