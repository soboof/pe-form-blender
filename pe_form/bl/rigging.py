"""Skeleton from Modules: a joint per module, as the Theater's skeleton is
before anything is typed or moved (skelFromRig; engine/rig.py), built as a
Blender armature with the formation skinned to it, and rigged in three steps,
each made from the one before it:

  1 Skeleton     "<name> Skeleton", an armature of connected bones, made from
                 the modules whatever was done before: a joint on every
                 module's centre (a carve is a cavity, not a module, and has
                 none) and a bone into every one from the module it hangs
                 from, named after the module it runs to (the module and its
                 number), each bone joined to the one before; the root
                 module's bone is a short one under it. A module moves with
                 the bone into it, so turning a bone swings its module and
                 everything after it about the bone's head. Bones wear the
                 colour of the module they run to. It opens in Edit Mode, to
                 be edited and typed: the joints moved where the figure should
                 bend (Blender's snapping takes a joint being moved onto a
                 module's centre or a seam where two modules meet: snap
                 points, bones of the skeleton drawn as points, and nothing
                 else, other objects and the skin included), each bone given
                 a type as in the Theater's Bones panel (Bone Type: spine,
                 leg, arm, wing...: the Theater's colour for it and a bone
                 collection of it), and the root chosen (Make Root, the Theater's: the tree
                 hung from that module).
  2 Control Rig  made from the skeleton as it was edited and typed: over the
                 modules' bones (which deform, and are the FK controls) a root
                 on the floor everything hangs from, and for every Leg or Paw
                 chain an IK foot control at its ankle and a knee pole, with
                 an IK/FK switch ("the control rig" below). It opens in Edit
                 Mode on the controls, to be edited in turn (a pole moved, a
                 foot control turned); what is done to them is kept through
                 every control rig made after.
  3 Pose         Pose Mode, each knee's pole fitted to the controls as they
                 were left; poses saved by name, and the Neutral Pose.
  "<name> Skin"  the formation's own mesh, modelling and all, skinned to it
                 (each face's module from "pe_mod", face_modules):
                 - Modules: one connected mesh, exactly the formation's; a
                   vertex of one module bound whole to the bone into that
                   module, so a module keeps its shape, and a vertex where
                   modules meet shared equally between their bones, so the
                   geometry stays joined where a joint bends;
                 - Automatic weights: the welded shell, weighted by Blender's
                   own bone heat, worked out again from the joints as they
                   were edited when the control rig is made.
It stands where the formation stands, in its frame (its origin is its base
plate). The formation itself is hidden, not changed, and comes back with
"Back to the Formation". Making the skeleton again replaces the skeleton and
the skin and starts again at step 1, the joints moved by hand kept on their
modules (Keep Moved Joints), and the types, the root, the poses and what was
done to the controls kept.
"""

import bpy
import bmesh
from bpy.app.handlers import persistent
from bpy.props import EnumProperty, BoolProperty

from ..engine import frame as FR
from ..engine import rig as R
from ..engine.geom import get_edge
from ..engine.meshplan import to_bl, _srgb
from .formation import (Formation, edit_formation, follow_selection, is_formation, material, read_frame,
                        COLOR_ATTR)
from .panels import _wrap

RIG_OF = "pe_rig_of"        # on the armature and the skin: the formation (an object pointer)
RIG_WID = "pe_rig_wid"      # the formation's write id when the skeleton was made
RIG_SUMMARY = "pe_rig_summary"
RIG_PROP, SKIN_PROP = "pe_rig", "pe_skin"   # on the formation: its skeleton and skin (object pointers)
RIG_SET = "pe_rig_set"      # on the skeleton: {root: joint name, stage: the step it is at}
BONE_TYPE = "pe_type"       # on a bone: its chain type (the Theater's), "" for none
SKEL_ID = "pe_skel_id"      # on the skeleton, and on every pose made on it: which skeleton the pose is for
BUILT_AT = "pe_at"          # on a module's bone: its tail (its joint) as made from the modules
SNAP_KIND = "pe_snap_kind"  # on a snap bone: 0 a module's centre, 1 a seam
SNAP_WAS = "pe_snap_was"    # on the skeleton, while it is edited: the scene's snapping before
SNAP_PROP = "pe_snap"       # the version before's snap points object (on the formation), cleaned up
SEL_WAS = "pe_sel_was"      # the version before's mark on a skin made unselectable, cleaned up
SKIN_KIND = "pe_skin_kind"  # on the skin: "modules" or "auto"
STAGE_SKELETON, STAGE_CONTROLS, STAGE_POSE = "skeleton", "controls", "pose"


def _obj(v):
    """An object a property names: a pointer (since saving to an asset
    library, which renames on append), or a name (older files)."""
    if isinstance(v, bpy.types.Object):
        return v
    if isinstance(v, str) and v:
        return bpy.data.objects.get(v)
    return None

SKIN_ITEMS = [
    ("modules", "Modules", "One connected mesh, the formation's own: each module moves with its own bone, and "
                           "where modules meet the corners follow both, so the geometry stays joined"),
    ("auto", "Automatic weights", "The welded shell, weighted by Blender's bone heat (bends smoothly)"),
]


def formation_of(context):
    """The formation the Skeleton panel is about: the one in Edit Mode, the
    active one, or the one the active skeleton or skin was made from."""
    obj = edit_formation(context)
    if obj is not None:
        return obj
    act = context.active_object
    if act is None:
        return None
    if is_formation(act):
        return act
    f = _obj(act.get(RIG_OF))
    if f is not None and is_formation(f):
        return f
    return None


def rig_objects(form):
    if form is None:
        return None, None
    arm, skin = _obj(form.get(RIG_PROP)), _obj(form.get(SKIN_PROP))
    if arm is not None and _obj(arm.get(RIG_OF)) != form:
        arm = None
    if skin is not None and _obj(skin.get(RIG_OF)) != form:
        skin = None
    return arm, skin


# ── the skeleton ─────────────────────────────────────────────────────────────
class Skeleton:
    """A joint per module (engine space), hung from the root chosen, a bone
    into every one, and the shell's triangles each with its module."""

    def __init__(self, engine, name, M=None, root=None, at=None):
        self.src = R.rig_from_engine(engine, name, stand=False)
        J = R.skel_from_rig(self.src)
        self.E = get_edge(self.src.X)
        self.M = M if M is not None else FR.IDENT
        self.sc = FR.scale_of(self.M) or 1.0
        # The root chosen (Make Root): the tree hung from that module, the
        # parent chain from it to the first module turned round.
        if root:
            k = next((i for i, j in enumerate(J) if j.name == root), None)
            if k is not None and J[k].parent >= 0:
                J = R.reroot_skeleton(J, k)
        self.root = next(i for i, j in enumerate(J) if j.parent < 0)
        self.joints = J
        kids = [[] for _ in J]
        for i, j in enumerate(J):
            if j.parent >= 0:
                kids[j.parent].append(i)
        self.kids = kids
        tris = [(f.a, f.b, f.c) for f in self.src.ghost_faces]
        self.tri_mod = R.tri_modules(self.src, tris, self.E * 0.015)
        self.moved = 0
        if at:
            self.move(at)

    def move(self, at):
        """The joints where they were moved by hand (Edit Skeleton): `at`,
        {joint name: its place in the armature's space}, read back into the
        engine's. Counts the joints off their module's centre."""
        Mi = FR.inverse(self.M)
        J = self.joints
        for i, j in enumerate(J):
            b = at.get(j.name)
            if b is None:
                continue
            q = FR.apply(Mi, (b[0], b[1], b[2]))
            J[i] = j.copy()
            J[i].pos = R.Vec3(q[0], q[2], -q[1])
        mods = self.src.modules
        self.moved = sum(1 for i, j in enumerate(J)
                         if i < len(mods) and j.pos.distance_to(mods[i].center) > self.E * 1e-4)

    def seam_point(self, i, p):
        """The middle of the face modules i and p meet on (the one either was
        placed on the other by), or None."""
        mods = self.src.modules
        if mods[i].parent == p and mods[i].conn_cen is not None:
            return mods[i].conn_cen
        if mods[p].parent == i and mods[p].conn_cen is not None:
            return mods[p].conn_cen
        return None

    # Standing as the formation stands in Blender: its frame read back to the
    # Theater's Y up, unscaled. A Blender point is sc * to_bl(standing).
    def stand(self, p):
        b = FR.apply(self.M, to_bl(p))
        sc = self.sc
        return R.Vec3(b[0] / sc, b[2] / sc, -b[1] / sc)

    def typed(self, types=None, guess=True):
        """The joints standing as the formation stands, typed by the bones'
        types (Bone Type, by joint name); with no leg, paw, arm or wing among
        them, and `guess`, read from the shape (the Theater's Auto-type,
        "Read from its shape"). (joints, read from shape?)"""
        from ..engine import neutral as N
        Js = [j.copy() for j in self.joints]
        for j in Js:
            j.pos = self.stand(j.pos)
            j.type = (types or {}).get(j.name, "") if j.parent >= 0 else ""
        guessed = guess and not any(j.type in N.NEUTRAL_TYPES for j in Js)
        if guessed:
            T, _ = R.rig_by_plan(Js, "auto")
            T, _ = R.rerig_limbs(T, "auto", self.src.modules, self.E, hinge=False)
            for j, t in zip(Js, T):
                j.type = t.type if t.parent == j.parent else ""
        return Js, guessed

    def legs(self, types=None, guess=True):
        """The legs a control rig puts IK on: the neutral pose's leg and paw
        plans (each its path of joints from where it is joined on, and the
        ankle along it), a leg whose body is below it (a skeleton rooted at a
        foot) left out. (plans, stood joints, frame, read from shape?)"""
        from ..engine import neutral as N
        Js, guessed = self.typed(types, guess)
        cx = N.motion_ctx(Js)
        return [p for p in N.limb_plans(Js, cx) if p.type in ("leg", "paw")], Js, cx, guessed

    def neutral(self, types=None):
        """The Theater's neutral pose (engine/neutral.py) of this skeleton,
        standing as it stands, with the body bound as Rigid modules binds it.
        `types` are the bones' types by joint name (Bone Type); with no leg,
        paw, arm or wing among them, the limbs are read from the shape (the
        Theater's Auto-type, "Read from its shape"), for the pose only.
        (neutral, stood joints, read from shape?), or (None, None, ...) when
        it has no limb to put in place."""
        from ..engine import neutral as N
        Js, guessed = self.typed(types)
        if not any(j.type in N.NEUTRAL_TYPES for j in Js):
            return None, None, guessed
        st = R.RigSource(self.src.name, self.src.X, [], [])
        for m in self.src.modules:
            c = R.RigModule(m.idx, m.mod_id, self.stand(m.center), m.parent)
            c.conn_cen = self.stand(m.conn_cen) if m.conn_cen is not None else None
            c.seam = [self.stand(v) for v in m.seam] if m.seam else None
            c.hull = ([self.stand(v) for v in m.hull[0]], m.hull[1]) if m.hull is not None else None
            st.modules.append(c)
        st.ghost_faces = [R.RigFace(self.stand(f.a), self.stand(f.b), self.stand(f.c), f.mod, f.color)
                          for f in self.src.ghost_faces]
        tris = [(f.a, f.b, f.c) for f in st.ghost_faces]
        owners = R.module_owners(Js, st, tris, self.tri_mod, self.E)
        return N.neutral_pose(Js, N.module_body(st, owners)), Js, guessed

    def head_joint(self, i):
        """The joint a bone (named after joint i) turns about: its head."""
        p = self.joints[i].parent
        return p if p >= 0 else i

    # A bone into every joint, from the joint it hangs from (head) to it
    # (tail), joined to the bone before; the root's is a short bone under it
    # (bone_ends). A module moves with the bone into it.
    def bones(self):
        """The joints that have a bone (named after them): all."""
        return list(range(len(self.joints)))

    def head(self, i):
        """A bone's head (engine space), the root's aside (bone_ends)."""
        j = self.joints[i]
        return self.joints[j.parent].pos.clone() if j.parent >= 0 else None

    def tail(self, i):
        J = self.joints
        j = J[i]
        h = self.head(i)
        if h is None or j.pos.distance_to(h) > self.E * 1e-3:
            return j.pos.clone()
        g = J[j.parent].parent
        d = h.clone().sub(J[g].pos) if g >= 0 else R.Vec3(0, 1, 0)
        if d.length_sq() < 1e-12:
            d = R.Vec3(0, 1, 0)
        return h.add_scaled_vector(d.normalize(), self.E * 0.6)

    def bone_parent(self, i):
        """The bone a bone hangs from (the bone into its head's joint) and
        whether it is joined to it: always, the root's having none."""
        p = self.joints[i].parent
        return (p, True) if p >= 0 else (None, False)

    def owner(self, m):
        """The bone a module moves with: the one into it."""
        return m


def bone_ends(sk, i, M):
    """A bone's head and tail in the armature's space (the formation's,
    Blender's): from the joint it hangs from to its own; the root's a short
    bone up into the root joint, from under it."""
    from mathutils import Vector
    put = lambda p: Vector(FR.apply(M, to_bl(p)))
    tail = put(sk.tail(i))
    h = sk.head(i)
    if h is None:
        return tail - Vector((0, 0, sk.E * sk.sc * 0.6)), tail
    return put(h), tail


def _remove(obj):
    if obj is None:
        return
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Armature):
            bpy.data.armatures.remove(data)
        elif isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)


def _link_like(new, like):
    cols = list(like.users_collection) or [bpy.context.scene.collection]
    for c in cols:
        c.objects.link(new)


def _mix(c, w, t):
    return tuple(a + (b - a) * t for a, b in zip(c, w))


def _module_colour(sk, i):
    return _srgb(sk.src.modules[i].color if i < len(sk.src.modules) else "#cccccc")[:3]


def _tint(b, c):
    if not hasattr(b, "color"):
        return
    b.color.palette = "CUSTOM"
    b.color.custom.normal = c
    b.color.custom.select = _mix(c, (1, 1, 1), 0.45)
    b.color.custom.active = _mix(c, (1, 1, 1), 0.75)


# ── bone types (the Theater's chain types) ──────────────────────────────────
TYPE_ITEMS = [("none", "None", "No type")] + [
    ("spine", "Spine", "The body's spine"), ("shoulder", "Shoulder", "A collarbone or a hip an arm or leg hangs from"),
    ("arm", "Arm", "An arm: it hangs at the side in the neutral pose"),
    ("leg", "Leg", "A leg: straight down in the neutral pose"),
    ("paw", "Paw", "An animal's leg: straight down in the neutral pose"),
    ("wing", "Wing", "A wing: spread level out to the side in the neutral pose"),
    ("tail", "Tail", "A tail"), ("head", "Head", "A head or a neck"), ("eye", "Eye", "An eye"),
    ("finger", "Finger", "A finger"), ("robot", "Robot", "A mechanical part"),
]
TYPE_NAMES = {"spine": "Spine", "shoulder": "Shoulders", "arm": "Arms", "leg": "Legs", "paw": "Paws",
              "wing": "Wings", "tail": "Tail", "head": "Head", "finger": "Fingers", "eye": "Eyes",
              "robot": "Robot"}


def bone_types(arm):
    """Every bone's type, by bone name (= its joint's name)."""
    return {b.name: b.get(BONE_TYPE, "") for b in arm.data.bones}


def set_bone_type(arm, sk, name, t):
    """Give a bone a type: the Theater's colour for it and a bone collection
    of it (none: back to its module's colour, out of every type's
    collection)."""
    b = arm.data.bones.get(name)
    if b is None:
        return
    t = "" if t == "none" else t
    b[BONE_TYPE] = t
    for c in list(b.collections):
        if c.get(BONE_TYPE):
            c.unassign(b)
    if t:
        cname = TYPE_NAMES.get(t, t.title())
        bc = arm.data.collections.get(cname)
        if bc is None:
            bc = arm.data.collections.new(cname)
            bc[BONE_TYPE] = t
        bc.assign(b)
        _tint(b, _srgb("#%06x" % R.TYPE_COL[t])[:3] if t in R.TYPE_COL else (0.8, 0.8, 0.8))
    else:
        i = next((k for k, j in enumerate(sk.joints) if j.name == name), None) if sk is not None else None
        if i is not None:
            _tint(b, _module_colour(sk, i))
    for c in list(arm.data.collections):
        if c.get(BONE_TYPE) and not len(c.bones):
            arm.data.collections.remove(c)


def build_armature(context, form, sk, M, home=None):
    """The skeleton's armature: a bone into every joint, in the Modules bone
    collection, each knowing where its joint was made (`home`, {name: tail},
    the modules' centres: Keep Moved Joints tells a joint moved by hand by
    it)."""
    J = sk.joints
    name = form.name + " Skeleton"
    arm = bpy.data.armatures.new(name)
    arm.display_type = "OCTAHEDRAL"
    ao = bpy.data.objects.new(name, arm)
    _link_like(ao, form)
    ao.matrix_world = form.matrix_world.copy()
    ao.show_in_front = True
    ao[RIG_OF] = form
    names = [j.name for j in J]
    vl = context.view_layer
    for o in context.selected_objects:
        o.select_set(False)
    ao.select_set(True)
    vl.objects.active = ao
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = arm.edit_bones
    made = {}
    for i in sk.bones():
        eb = ebs.new(names[i])
        eb.head, eb.tail = bone_ends(sk, i, M)
        made[i] = eb
    for i, eb in made.items():
        p, joined = sk.bone_parent(i)
        if p is not None and p in made:
            eb.parent = made[p]
            eb.use_connect = joined and (made[p].tail - eb.head).length < 1e-6
    bpy.ops.object.mode_set(mode="OBJECT")
    mod_c = _collection(arm, MOD_COLL)
    for i in made:
        b = arm.bones[names[i]]
        _tint(b, _module_colour(sk, i))
        b[BUILT_AT] = list(home[names[i]]) if home and names[i] in home else list(b.tail_local)
        mod_c.assign(b)
    return ao, names


def joints_of(arm):
    """Where each module's joint is, as the skeleton stands now (edited or
    not): the tail of the bone into it, in the armature's space, at rest."""
    if arm.mode == "EDIT":
        return {eb.name: tuple(eb.tail) for eb in arm.data.edit_bones if not eb.get(CTRL)}
    return {b.name: tuple(b.tail_local) for b in arm.data.bones if not is_control(b)}


def _kept_joints(arm, home, tol):
    """The joints moved by hand on skeleton `arm` that are still on the module
    they were moved from (its centre still where it was made): {name:
    place}. `home` is every joint's place as made from the modules now."""
    from mathutils import Vector
    out = {}
    for name, at in joints_of(arm).items():
        h = home.get(name)
        if h is None:
            continue
        b = arm.data.bones.get(name)
        made = Vector(b[BUILT_AT]) if b is not None and BUILT_AT in b.keys() else Vector(h)
        if (Vector(at) - made).length > tol and (Vector(h) - made).length <= tol:
            out[name] = at
    return out


# ── the control rig ──────────────────────────────────────────────────────────
# Step 2, made from the skeleton as it was edited and typed. The module bones
# deform the skin and are the FK controls (each posed by turning it). Over
# them, controls to animate with, none of them deforming, in the "Controls"
# bone collection, each drawn as a shape of its own:
#   "root"         on the floor at the object's origin, a circle round the
#                  formation: every bone hangs from it, so moving it moves all;
#   "IK.<bone>"    for every leg, at its ankle (<bone>: the one into the
#                  ankle), a box: the leg reaches it by IK from where it is
#                  joined on, and the foot past the ankle turns with it;
#   "Pole.<bone>"  where its knee points, a diamond.
# The legs are the chains typed Leg or Paw (Bone Type); nothing is read from
# the shape here, the types are the user's. The IK is solved on a hidden copy
# of the leg ("MCH-IK.<bone>", in the "Mechanism" collection). A straight
# leg's copy is bent at the knee by PRE_BEND of its length, toward its pole:
# Blender's IK does not bend a chain that is exactly straight (measured: a
# two-bone leg stays straight with the foot control moved up, whichever pole
# or solver; it bends from 1e-3 of its length). Each bone of the copy carries
# the leg's bone as it rests ("MCH-LEG.<bone>", its child), and the leg's bone
# takes its place from that (Copy Transforms): the rest pose is the rest pose
# and the ankle lands on its control (copying the copy's turns instead, local
# to local, put the foot up to 0.6% of the leg off its control). For that a
# leg's bones below its hip are unjoined from the bone before (their heads
# still on its tail at rest, their places locked, so FK turns them as
# before): a joined bone keeps its head on its parent's tail whatever it
# copies. A knee then opens by the pre-bend times the bend (at most about 0.01
# of an edge on the demo pack), and the skin's corners shared there, weighted
# between the two bones, close it. The foot copies the turn of a hidden bone
# riding on its control ("MCH-FOOT.<bone>", the foot as it rests), so the
# control can be turned in Edit Controls without turning the foot. A foot
# control's "ik" (1: IK; 0: FK, the leg's bones' own turns) drives its leg's
# constraints.
#
# The control rig can be edited (Edit Controls): every control remembers where
# it was made (BUILT), and what was done to it there (moved, turned) is kept,
# as a change from where it was made, through every control rig made after it
# (the skeleton edited again, a bone typed), as are the controls' poses. The
# knees' poles are fitted to the controls as they were left (fit_controls),
# so a leg still rests as it was made.
CTRL = "pe_ctrl"            # on a control rig bone: "root", "ik", "pole" or "mch"; "snap" on a snap point
CTRL_SET = "pe_ctrl_set"    # on the skeleton: {legs: [the bones into the ankles], single: [...], s: an edge}
CTRL_EDITS = "pe_ctrl_edits"  # on the skeleton: each control as edited: {name: [head, tail, roll changes]}
CTRL_POSE = "pe_ctrl_pose"    # on the skeleton while its control rig is off: {name: {m: matrix, ik}}
BUILT = "pe_built"          # on a control: where it was made, [head, tail, roll]
IK_PROP = "ik"
UNJOINED = "pe_ik_leg"       # on an IK leg's bone below its hip: unjoined from the one before for the rig
CON_PREFIX = "PE "
CON_IK, CON_LEG, CON_FOOT = "PE IK", "PE Leg", "PE Foot"
ROOT_BONE = "root"
CTRL_COLL, MCH_COLL, MOD_COLL = "Controls", "Mechanism", "Modules"
WGT_COLL = "PE Widgets"
PRE_BEND = 2e-3
ROOT_COL, LEFT_COL, RIGHT_COL, POLE_COL = "#e8c547", "#4ca8d8", "#c44444", "#b0d152"


def is_control(b):
    """A bone of the control rig, not a module's."""
    return bool(b.get(CTRL))


def module_bones(arm):
    return [b for b in arm.data.bones if not is_control(b)]


def has_controls(arm):
    """Whether skeleton `arm` has its control rig on."""
    return arm is not None and (CTRL_SET in arm.keys() or
                                any(b.get(CTRL) not in (None, "snap") for b in arm.data.bones))


def _plain(v):
    """An ID property group as a dict ({} for none)."""
    if v is None:
        return {}
    return v.to_dict() if hasattr(v, "to_dict") else dict(v)


def _layer_coll(lc, coll):
    if lc.collection == coll:
        return lc
    for c in lc.children:
        hit = _layer_coll(c, coll)
        if hit is not None:
            return hit
    return None


def _widget(kind):
    """A control's shape: a wire mesh, made once and kept out of the view
    (the "PE Widgets" collection, excluded from every view layer)."""
    import math
    name = "WGT-PE-" + kind
    ob = bpy.data.objects.get(name)
    if ob is not None and ob.type == "MESH":
        return ob
    if kind == "Root":
        n = 32
        v = [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n), 0.0) for k in range(n)]
        e = [(k, (k + 1) % n) for k in range(n)]
    elif kind == "Box":
        v = [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        e = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
    else:   # "Diamond"
        v = [(1.0, 0, 0), (-1.0, 0, 0), (0, 1.0, 0), (0, -1.0, 0), (0, 0, 1.0), (0, 0, -1.0)]
        e = [(a, b) for a in range(6) for b in range(a + 1, 6) if a // 2 != b // 2]
    me = bpy.data.meshes.new(name)
    me.from_pydata(v, e, [])
    ob = bpy.data.objects.new(name, me)
    coll = bpy.data.collections.get(WGT_COLL)
    if coll is None:
        coll = bpy.data.collections.new(WGT_COLL)
        coll.hide_render = True
    coll.objects.link(ob)
    sc = bpy.context.scene
    if sc.collection.children.get(coll.name) is None:
        sc.collection.children.link(coll)
    for vl in sc.view_layers:
        lc = _layer_coll(vl.layer_collection, coll)
        if lc is not None:
            lc.exclude = True
    return ob


def _signed_angle(u, v, n):
    a = u.angle(v)
    c = u.cross(v)
    return a if c.length < 1e-9 else (-a if c.angle(n) < 1 else a)


def _pole_angle(head, tail, x_axis, tip_tail, pole):
    """The IK pole angle that leaves a chain as it rests: its first bone from
    `head` to `tail` (its x axis `x_axis`), its last ending at `tip_tail`, its
    pole at `pole` (armature space)."""
    normal = (tip_tail - head).cross(pole - head)
    axis = tail - head
    return _signed_angle(x_axis, normal.cross(axis), axis)


def _collection(arm, name, visible=True):
    bc = arm.collections.get(name)
    if bc is None:
        bc = arm.collections.new(name)
    bc.is_visible = visible
    return bc


def _drive(con, ao, ctrl):
    """The constraint's influence: control `ctrl`'s "ik" (no expression, so
    it needs no scripts allowed to run)."""
    fc = con.driver_add("influence")
    d = fc.driver
    d.type = "AVERAGE"
    v = d.variables.new()
    v.name = IK_PROP
    v.type = "SINGLE_PROP"
    v.targets[0].id = ao
    v.targets[0].data_path = 'pose.bones["%s"]["%s"]' % (ctrl.replace('"', '\\"'), IK_PROP)


def _leg_bend(chain, axis_dir, fallbacks):
    """Where a leg's knee is and which way it bends (perpendicular to the
    leg from hip to ankle): the interior joint furthest off that line, else
    (a straight leg) its middle joint and the first fallback direction (the
    figure's facing) not along the leg. (knee, unit bend, straight?); a
    leg of one bone has no knee: (None, None, False)."""
    if len(chain) < 2:
        return None, None, False
    base = chain[0].head
    knee, bend = None, None
    for b in chain[:-1]:
        o = b.tail - base
        o = o - axis_dir * o.dot(axis_dir)
        if bend is None or o.length > bend.length:
            knee, bend = b.tail.copy(), o
    L = (chain[-1].tail - base).length
    if bend.length > PRE_BEND * L:
        return knee, bend.normalized(), False
    knee = chain[(len(chain) - 1) // 2].tail.copy()
    for d in fallbacks:
        o = d - axis_dir * d.dot(axis_dir)
        if o.length > 1e-3:
            return knee, o.normalized(), True
    return knee, bend, True


def _as_active(context, ob):
    """`ob` shown and active, in Object Mode; what to put back (_put_back)."""
    vl = context.view_layer
    was = (vl.objects.active, ob.mode, ob.hide_get())
    ob.hide_set(False)
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    vl.objects.active = ob
    return was


def _put_back(context, ob, was):
    active, mode, hidden = was
    if mode != "OBJECT":
        bpy.ops.object.mode_set(mode=mode)
    if active is not None and active != ob:
        bpy.ops.object.mode_set(mode="OBJECT")
        context.view_layer.objects.active = active
    ob.hide_set(hidden)


def _placement(b):
    """A bone's head, tail and roll (armature space, at rest)."""
    if isinstance(b, bpy.types.EditBone):
        return b.head.copy(), b.tail.copy(), b.roll
    _, roll = bpy.types.Bone.AxisRollFromMatrix(b.matrix_local.to_3x3())
    return b.head_local.copy(), b.tail_local.copy(), roll


def _wrap_angle(a):
    import math
    return (a + math.pi) % (2 * math.pi) - math.pi


def _ctrl_keep(ao):
    """What was done by hand to the control rig on `ao`, to keep through the
    next one: each control's edit (Edit Controls: its head, tail and roll less
    where it was made) and each control's pose (and IK switch), by name, over
    what was kept from before (a control rig taken off, a control gone with
    its leg's type). (edits, poses)"""
    from mathutils import Matrix, Vector
    edits = {k: list(v) for k, v in _plain(ao.get(CTRL_EDITS)).items()}
    poses = {k: dict(v) for k, v in _plain(ao.get(CTRL_POSE)).items()}
    tol = float(_plain(ao.get(CTRL_SET)).get("s", 1.0)) * 1e-5
    ident = Matrix.Identity(4)
    for b in ao.data.bones:
        kind = b.get(CTRL)
        if not kind or kind in ("mch", "snap"):
            continue
        made = b.get(BUILT)
        if made is not None and len(made) == 7:
            made = list(made)
            h, t, r = _placement(b)
            d = list(h - Vector(made[0:3])) + list(t - Vector(made[3:6])) + [_wrap_angle(r - made[6])]
            if max(abs(x) for x in d[:6]) > tol or abs(d[6]) > 1e-5:
                edits[b.name] = d
            else:
                edits.pop(b.name, None)
        pb = ao.pose.bones.get(b.name)
        if pb is None:
            continue
        ik = float(pb[IK_PROP]) if IK_PROP in pb.keys() else None
        mb = pb.matrix_basis
        posed = any(abs(mb[i][j] - ident[i][j]) > 1e-6 for i in range(4) for j in range(4))
        if posed or (ik is not None and abs(ik - 1.0) > 1e-6):
            p = {"m": [mb[i][j] for i in range(4) for j in range(4)]}
            if ik is not None:
                p["ik"] = ik
            poses[b.name] = p
        else:
            poses.pop(b.name, None)
    return edits, poses


def _made(eb, edits):
    """A control just made: where it was made remembered, and what was done to
    it by hand before (`edits`) done again."""
    from mathutils import Vector
    eb[BUILT] = [*eb.head, *eb.tail, eb.roll]
    d = edits.get(eb.name)
    if d is not None and len(d) == 7:
        h, t = eb.head + Vector(d[0:3]), eb.tail + Vector(d[3:6])
        if (t - h).length > 1e-6:
            eb.head, eb.tail = h, t
            eb.roll = eb.roll + d[6]


def _take_off(ao):
    """(The skeleton active, in Object Mode.) The control rig's constraints,
    then in Edit Mode its bones, and the legs' bones joined again (where their
    heads are on their parents' tails, as every module's bone is). Leaves it
    in Edit Mode; returns the bones joined again."""
    for pb in ao.pose.bones:
        for con in list(pb.constraints):
            if con.name.startswith(CON_PREFIX):
                con.driver_remove("influence")
                pb.constraints.remove(con)
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = ao.data.edit_bones
    for eb in [eb for eb in ebs if eb.get(CTRL)]:
        ebs.remove(eb)
    was_leg = set()
    for eb in ebs:
        if eb.get(UNJOINED):
            del eb[UNJOINED]
            was_leg.add(eb.name)
            if eb.parent is not None and (eb.parent.tail - eb.head).length < 1e-6:
                eb.use_connect = True
    return was_leg


def remove_controls(context, ao):
    """Take the control rig off skeleton `ao`, to edit the skeleton under it:
    its controls and their constraints gone, the legs' bones joined again, and
    what was done to the controls by hand (their edits and poses) kept on the
    skeleton for the next control rig."""
    if not has_controls(ao):
        return
    was = _as_active(context, ao)
    edits, poses = _ctrl_keep(ao)
    was_leg = _take_off(ao)
    bpy.ops.object.mode_set(mode="OBJECT")
    for nm in was_leg:
        pb = ao.pose.bones.get(nm)
        if pb is not None:
            pb.lock_location = (False,) * 3
    for name in (CTRL_COLL, MCH_COLL):
        bc = ao.data.collections.get(name)
        if bc is not None:
            ao.data.collections.remove(bc)
    ao[CTRL_EDITS] = edits
    ao[CTRL_POSE] = poses
    if CTRL_SET in ao.keys():
        del ao[CTRL_SET]
    _put_back(context, ao, was)


def build_controls(context, ao, sk, types=None):
    """Put the control rig on skeleton `ao` (made from `sk`, its joints as
    they were edited), in place of the one it has: IK on the chains typed Leg
    or Paw (Bone Type), and what was done to the controls by hand (their edits
    and poses, by name) kept. Returns (legs given IK, legs of one bone left
    FK)."""
    from mathutils import Matrix, Vector
    was = _as_active(context, ao)
    edits, poses = _ctrl_keep(ao)
    plans, Js, cx, _ = sk.legs(bone_types(ao) if types is None else types, guess=False)
    J = sk.joints
    arm = ao.data
    was_leg = _take_off(ao)
    ebs = arm.edit_bones
    s = sk.E * sk.sc
    up = Vector((0, 0, 1))
    fwd = Vector(to_bl(cx.F))
    side = Vector(to_bl(cx.L))
    # The root, flat on the floor at the origin, round every bone.
    reach = max((Vector((p.x, p.y)).length for eb in ebs for p in (eb.head, eb.tail)), default=0.0)
    root = ebs.new(ROOT_BONE)
    root.head, root.tail = Vector((0, 0, 0)), Vector((0, reach + s * 0.5, 0))
    root[CTRL] = "root"
    root.use_deform = False
    _made(root, edits)
    first = ebs.get(J[sk.root].name)
    if first is not None:
        first.use_connect = False
        first.parent = root
    legs, single = [], []
    for p in plans:
        names = [J[i].name for i in p.path]
        chain = [ebs.get(n) for n in names[1:p.end + 1]]
        if not chain or any(b is None for b in chain):
            continue
        if len(chain) < 2:
            # One bone has no knee: IK could only aim it, and its foot would
            # leave a control it cannot reach. It stays FK.
            single.append(chain[0].name)
            continue
        foot = ebs.get(names[p.end + 1]) if p.end + 1 < len(names) else None
        base, tip = chain[0], chain[-1]
        ankle = tip.tail.copy()
        axis = ankle - base.head
        L = axis.length
        if L < s * 1e-3:
            continue
        a = axis.normalized()
        # Its foot control: at the ankle, turned as the foot is.
        ik = ebs.new("IK." + tip.name)
        ik.head = ankle
        if foot is not None and foot.length > 1e-9:
            ik.tail = ankle + (foot.tail - foot.head)
            ik.roll = foot.roll
        else:
            d = fwd - up * fwd.dot(up)
            ik.tail = ankle + (d.normalized() if d.length > 1e-6 else Vector((0, 1, 0))) * s * 0.5
        ik[CTRL] = "ik"
        ik.use_deform = False
        ik.parent = root
        _made(ik, edits)
        # The foot as it rests, riding on its control: what the foot turns as.
        if foot is not None:
            mf = ebs.new("MCH-FOOT." + tip.name)
            mf.head, mf.tail, mf.roll = foot.head.copy(), foot.tail.copy(), foot.roll
            mf.parent = ik
            mf[CTRL] = "mch"
            mf.use_deform = False
        # The copy of the leg IK is solved on.
        knee, bend, straight = _leg_bend(chain, a, (fwd, side, up, Vector((1, 0, 0))))
        mch, prev = [], chain[0].parent
        for k, b in enumerate(chain):
            h = ebs.new("MCH-IK." + b.name)
            h.head = b.head.copy()
            h.tail = b.tail.copy()
            if straight and k < len(chain) - 1:
                h.tail = h.tail + bend * PRE_BEND * L
            if mch:
                h.head = mch[-1].tail.copy()
            h.roll = b.roll
            h.parent = prev
            h.use_connect = bool(mch)
            h[CTRL] = "mch"
            h.use_deform = False
            mch.append(h)
            prev = h
            # The leg's bone as it rests, riding on that copy: what the
            # bone takes its place from.
            f = ebs.new("MCH-LEG." + b.name)
            f.head, f.tail, f.roll = b.head.copy(), b.tail.copy(), b.roll
            f.parent = h
            f[CTRL] = "mch"
            f.use_deform = False
            # A joined bone keeps its head on its parent's tail whatever it
            # copies, and the knees of a pre-bent copy are not the leg's: the
            # ankle would miss its control by the pre-bend times the bend.
            if k:
                b.use_connect = False
                b[UNJOINED] = 1
        # Its knee's pole, beside the knee the way it bends.
        at = knee + bend * max(L * 0.5, s)
        pe = ebs.new("Pole." + tip.name)
        pe.head, pe.tail = at, at + up * s * 0.3
        pe[CTRL] = "pole"
        pe.use_deform = False
        pe.parent = root
        _made(pe, edits)
        pole = (pe.name, _pole_angle(mch[0].head, mch[0].tail, mch[0].x_axis, mch[-1].tail, pe.head))
        legs.append({"chain": [b.name for b in chain], "foot": foot.name if foot is not None else None,
                     "ik": ik.name, "pole": pole, "side": p.side})
    bpy.ops.object.mode_set(mode="OBJECT")
    # Shapes, colours and collections; the constraints and their switch.
    ctrl_c, mch_c, mod_c = _collection(arm, CTRL_COLL), _collection(arm, MCH_COLL, False), _collection(arm, MOD_COLL)
    for b in arm.bones:
        if not is_control(b) and MOD_COLL not in b.collections:
            mod_c.assign(b)
    pbs = ao.pose.bones
    rb = pbs[ROOT_BONE]
    rb.custom_shape = _widget("Root")
    ctrl_c.assign(rb.bone)
    _tint(rb.bone, _srgb(ROOT_COL)[:3])
    for leg in legs:
        col = _srgb(LEFT_COL if leg["side"] > 0 else RIGHT_COL)[:3]
        ikb = pbs[leg["ik"]]
        ikb.custom_shape = _widget("Box")
        ikb.use_custom_shape_bone_size = False
        ikb.custom_shape_scale_xyz = (s * 0.3,) * 3
        ikb[IK_PROP] = 1.0
        ikb.id_properties_ui(IK_PROP).update(
            min=0.0, max=1.0, soft_min=0.0, soft_max=1.0,
            description="1: the leg reaches this control (IK); 0: it follows its own bones' turns (FK)")
        ctrl_c.assign(ikb.bone)
        _tint(ikb.bone, col)
        tipm = pbs["MCH-IK." + leg["chain"][-1]]
        con = tipm.constraints.new("IK")
        con.name = CON_IK
        con.target, con.subtarget = ao, leg["ik"]
        con.chain_count = len(leg["chain"])
        con.use_stretch = False
        if leg["pole"] is not None:
            pn, angle = leg["pole"]
            con.pole_target, con.pole_subtarget, con.pole_angle = ao, pn, angle
            pp = pbs[pn]
            pp.custom_shape = _widget("Diamond")
            pp.use_custom_shape_bone_size = False
            pp.custom_shape_scale_xyz = (s * 0.18,) * 3
            ctrl_c.assign(pp.bone)
            _tint(pp.bone, col)
        for nm in leg["chain"]:
            mch_c.assign(pbs["MCH-IK." + nm].bone)
            mch_c.assign(pbs["MCH-LEG." + nm].bone)
            ct = pbs[nm].constraints.new("COPY_TRANSFORMS")
            ct.name = CON_LEG
            ct.target, ct.subtarget = ao, "MCH-LEG." + nm
            _drive(ct, ao, leg["ik"])
        if leg["foot"] is not None:
            mf = "MCH-FOOT." + leg["chain"][-1]
            mch_c.assign(pbs[mf].bone)
            cf = pbs[leg["foot"]].constraints.new("COPY_ROTATION")
            cf.name = CON_FOOT
            cf.target, cf.subtarget = ao, mf
            _drive(cf, ao, leg["ik"])
    unjoined = {b.name for b in arm.bones if b.get(UNJOINED)}
    for nm in was_leg | unjoined:
        if nm in pbs:
            pbs[nm].lock_location = (nm in unjoined,) * 3
    for nm, p in poses.items():
        pb = pbs.get(nm)
        m = p.get("m")
        if pb is None or not is_control(pb.bone) or m is None or len(m) != 16:
            continue
        pb.matrix_basis = Matrix([m[0:4], m[4:8], m[8:12], m[12:16]])
        if "ik" in p and IK_PROP in pb.keys():
            pb[IK_PROP] = float(p["ik"])
    ao[CTRL_SET] = {"legs": [leg["chain"][-1] for leg in legs], "single": single, "s": s}
    ao[CTRL_EDITS] = edits
    if CTRL_POSE in ao.keys():
        del ao[CTRL_POSE]
    _put_back(context, ao, was)
    return len(legs), len(single)


def fit_controls(ao):
    """Each knee's pole fitted to where the controls stand now (edited or
    not): the leg rests as it was made, its knee bending toward its pole."""
    bones = ao.data.bones
    for pb in ao.pose.bones:
        con = pb.constraints.get(CON_IK)
        if con is None or not con.pole_subtarget:
            continue
        tip = bones.get(pb.name)
        pole = bones.get(con.pole_subtarget)
        if tip is None or pole is None:
            continue
        base = tip
        for _ in range(con.chain_count - 1):
            if base.parent is None:
                break
            base = base.parent
        x = base.matrix_local.to_3x3().col[0]
        angle = _pole_angle(base.head_local, base.tail_local, x, tip.tail_local, pole.head_local)
        if abs(angle - con.pole_angle) > 1e-6:
            con.pole_angle = angle


def leg_of(arm, ankle):
    """The leg whose ankle bone is `ankle`: (its IK chain, hip to ankle;
    the bones past the ankle). ([], []) for none."""
    pb = arm.pose.bones.get("MCH-IK." + ankle)
    con = next((c for c in pb.constraints if c.name == CON_IK), None) if pb is not None else None
    if con is None or ankle not in arm.data.bones:
        return [], []
    b = arm.data.bones[ankle]
    chain = [b.name]
    for _ in range(con.chain_count - 1):
        b = b.parent
        chain.insert(0, b.name)
    return chain, [c.name for c in arm.data.bones[ankle].children_recursive]


def controls_note(arm):
    """What the control rig holds, in a line."""
    st = _plain(arm.get(CTRL_SET))
    n, one = len(st.get("legs", [])), len(st.get("single", []))
    if not n and not one:
        return "Controls: the root. No legs for IK: type the leg bones Leg (or Paw) in the skeleton"
    out = "Controls: the root"
    if n:
        out += ", and %d leg%s with IK (a foot and a knee each)" % (n, "" if n == 1 else "s")
    if one:
        out += "; %d leg%s of one bone left FK (IK needs two: type the module above it Leg too)" % (
            one, "" if one == 1 else "s")
    return out


# ── snapping a joint onto the modules (Edit Skeleton) ────────────────────────
# While the skeleton is edited, Blender's own snapping takes a joint being
# moved onto a snap point: one on every module's centre and one on every seam
# (the middle of the face a module was placed on). The points are bones of
# the skeleton itself ("SNAP.<n>", in the "Snap Points" bone collection,
# a thousandth of an edge long, unselectable, deforming nothing), drawn as
# points over everything (a centre white, a seam gold), and snapping is set
# to the vertices of the skeleton being edited and of nothing else. They were
# a mesh of loose vertices beside it, snapped to as another object: then
# every other object took part too, and on a scene with a second formation
# standing in the same place a joint landed on it (measured: 52 units off the
# centre aimed at), and the skin had to be made unselectable to stay out of
# the snapping's way, which kept Weight Paint off it. The snapping is the
# skeleton's while it is in Edit Mode at step 1, whichever way it got there
# (a button, Tab), and the scene's own again once it leaves (_follow_modes).
# The snap bones go when the control rig is made (they are not module bones).
SNAP_COLL = "Snap Points"
SNAP_SET = (("use_snap", True), ("snap_elements_base", "VERTEX"), ("snap_elements", "VERTEX"),
            ("snap_target", "CLOSEST"), ("use_snap_self", True), ("use_snap_edit", True),
            ("use_snap_nonedit", False), ("use_snap_translate", True))
CENTRE_COL, SEAM_COL = (0.91, 0.89, 0.86, 1.0), (0.91, 0.77, 0.28, 1.0)


def _snap_on(ts, arm):
    """The scene's snapping set to take a joint onto the snap points, what it
    was kept on the skeleton (once, while it is edited)."""
    if SNAP_WAS not in arm.keys():
        was = {}
        for k, _ in SNAP_SET:
            if hasattr(ts, k):
                v = getattr(ts, k)
                was[k] = ",".join(sorted(v)) if isinstance(v, set) else v
        arm[SNAP_WAS] = was
    done = set()
    for k, v in SNAP_SET:
        if not hasattr(ts, k) or (k == "snap_elements" and "snap_elements_base" in done):
            continue
        try:
            setattr(ts, k, {v} if k.startswith("snap_elements") else v)
            done.add(k)
        except (TypeError, AttributeError, ValueError):
            pass


def _snap_off(ts, arm):
    """The scene's snapping as it was before the skeleton was edited."""
    was = _plain(arm.get(SNAP_WAS))
    if SNAP_WAS in arm.keys():
        del arm[SNAP_WAS]
    for k, v in was.items():
        if k == "snap_elements" and "snap_elements_base" in was:
            continue
        try:
            setattr(ts, k, set(x for x in v.split(",") if x) if k.startswith("snap_elements") else v)
        except (TypeError, AttributeError, ValueError):
            pass


def snap_targets(sk):
    """Every snap point of skeleton `sk`, in the armature's space: [(point,
    kind)], kind 0 a module's centre, 1 a seam; a point given once."""
    from mathutils import Vector
    tol = sk.E * sk.sc * 1e-4
    out, seen = [], set()

    def add(p, kind):
        b = Vector(FR.apply(sk.M, to_bl(p)))
        key = tuple(round(c / tol) for c in b)
        if key not in seen:
            seen.add(key)
            out.append((b, kind))
    for m in sk.src.modules:
        add(m.center, 0)
    for m in sk.src.modules:
        if m.parent >= 0 and m.conn_cen is not None:
            add(m.conn_cen, 1)
    return out


def snap_bones(arm):
    """The skeleton's snap bones (edit bones in Edit Mode)."""
    bones = arm.data.edit_bones if arm.mode == "EDIT" else arm.data.bones
    return [b for b in bones if b.get(CTRL) == "snap"]


def make_snap_bones(arm, sk):
    """(Edit Mode) The snap points as bones of skeleton `arm`, in place of
    the ones it has."""
    from mathutils import Vector
    ebs = arm.data.edit_bones
    for eb in snap_bones(arm):
        ebs.remove(eb)
    bc = _collection(arm.data, SNAP_COLL)
    up = Vector((0, 0, sk.E * sk.sc * 1e-3))
    for k, (p, kind) in enumerate(snap_targets(sk)):
        eb = ebs.new("SNAP.%03d" % k)
        eb.head, eb.tail = p, p + up
        eb[CTRL] = "snap"
        eb[SNAP_KIND] = kind
        eb.use_deform = False
        eb.hide_select = True
        bc.assign(eb)


def snap_points(form):
    """The loose-vertex snap points object of the version before (cleaned up)."""
    return _obj(form.get(SNAP_PROP)) if form is not None else None


def _draw_snaps():
    """The snap points, while the skeleton they belong to is edited at step
    1: a module's centre white, a seam gold, over everything (they are inside
    the skin)."""
    ctx = bpy.context
    ob = ctx.active_object
    if ob is None or ob.type != "ARMATURE" or ob.mode != "EDIT" or RIG_SET not in ob.keys():
        return
    bc = ob.data.collections.get(SNAP_COLL)
    if bc is None or not bc.is_visible:
        return
    import gpu
    from gpu_extras.batch import batch_for_shader
    try:
        W = ob.matrix_world
        cen, seam = [], []
        for eb in ob.data.edit_bones:
            if eb.get(CTRL) == "snap":
                (seam if eb.get(SNAP_KIND) == 1 else cen).append(tuple(W @ eb.head))
        if not cen and not seam:
            return
        try:
            sh = gpu.shader.from_builtin("POINT_UNIFORM_COLOR")
        except ValueError:
            sh = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        for group, size, col in ((cen + seam, 9.0, (0.1, 0.09, 0.08, 0.85)), (cen, 6.0, CENTRE_COL),
                                 (seam, 6.0, SEAM_COL)):
            if not group:
                continue
            gpu.state.point_size_set(size)
            sh.uniform_float("color", col)
            batch_for_shader(sh, "POINTS", {"pos": group}).draw(sh)
    except Exception as ex:  # never take Blender's draw loop down
        print("PE Form snap points:", repr(ex))
    finally:
        gpu.state.point_size_set(1.0)
        gpu.state.blend_set("NONE")


# The skin is the formation's own mesh, as it stands: what the Form built
# and everything modelled on it with Blender's tools (an extrusion, a bevel,
# a face moved), which the engine's shell knows nothing of. It used to be
# rebuilt from the engine's shell, and a formation modelled on came out
# without its modelling: an extrusion's sides gone, the face it was extruded
# from closed again, a module placed on the extruded face hanging over a gap.
SKIN_DROP = ("pe_open", "pe_hex", "pe_wid", "pe_role", "pe_src", "pe_srcn", "pe_mod", "pe_base")


def _hist_modules(engine):
    """Each history step's module in the rig (rig_from_engine's numbering):
    a carve's own faces, its cavity's walls, go to the module it was cut
    from. -1 for none."""
    H = engine.history
    kept = [i for i, h in enumerate(H) if not h.carve]
    renum = {i: k for k, i in enumerate(kept)}
    out = []
    for i in range(len(H)):
        k = i
        while 0 <= k < len(H) and H[k].carve:
            p = H[k].parent_mod_idx
            k = p if p is not None else -1
        out.append(renum.get(k, -1))
    return out


def face_modules(form, bm, sk):
    """The rig module of every face of the formation's mesh (`bm`, read from
    it): the module the Form wrote it for ("pe_mod"), which Blender's tools
    copy onto what they make of a face; for a face of a file from before
    that, the Form's own polygon it is; anything still unknown, the module of
    a face beside it, else the module whose centre is nearest."""
    from . import usergeo
    st = Formation.of(form).state
    hist = _hist_modules(st.engine)
    nmod = len(sk.src.modules)
    lm = bm.faces.layers.int.get(usergeo.MOD_ATTR)
    lb = bm.verts.layers.float_vector.get("pe_base")
    out = {}
    for f in bm.faces:
        k = f[lm] - 1 if lm is not None else -1
        if 0 <= k < len(hist) and 0 <= hist[k] < nmod:
            out[f] = hist[k]
    if len(out) < len(bm.faces) and lb is not None:
        plan = {it[4]: it[6] for it in usergeo.build_plan(st)}
        for f in bm.faces:
            if f in out:
                continue
            k = plan.get(usergeo.poly_key([tuple(v[lb]) for v in f.verts]), -1)
            if 0 <= k < len(hist) and 0 <= hist[k] < nmod:
                out[f] = hist[k]
    todo = [f for f in bm.faces if f not in out]
    while todo:
        left = []
        for f in todo:
            near = next((out[g] for e in f.edges for g in e.link_faces if g in out), None)
            if near is None:
                left.append(f)
            else:
                out[f] = near
        if len(left) == len(todo):
            M = sk.M
            cen = [FR.apply(M, to_bl(m.center)) for m in sk.src.modules]
            for f in left:
                c = f.calc_center_median()
                out[f] = min(range(nmod), key=lambda i: (cen[i][0] - c.x) ** 2 + (cen[i][1] - c.y) ** 2
                             + (cen[i][2] - c.z) ** 2) if nmod else 0
            break
        todo = left
    _onto_parents(out, sk)
    return out


def _hull_planes(sk, m):
    """Module m's hull as outward planes (unit normal, a point on it), in the
    formation's own space (Blender's, its frame applied)."""
    from mathutils import Vector
    h = sk.src.modules[m].hull
    if h is None:
        return None
    pts = [Vector(FR.apply(sk.M, to_bl(p))) for p in h[0]]
    ctr = sum(pts, Vector()) / len(pts)
    out = []
    for f in h[1]:
        a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
        n = (b - a).cross(c - a)
        if n.length < 1e-12:
            continue
        n.normalize()
        if n.dot(a - ctr) < 0:
            n = -n
        out.append((n, a))
    return out


def _outside(planes, c):
    return max(n.dot(c - a) for n, a in planes)


def _onto_parents(fmod, sk):
    """A face filed under a module but lying outside it, on the module it was
    placed on, goes with that one: the Form files the adapter ring between a
    small module and the bigger face it sits on (in that face's plane) under
    the small module, and in Rigid modules it swung out with it. Modelling
    outside every module stays with its own."""
    tol = sk.E * sk.sc * 0.01
    planes = {}

    def pl(m):
        if m not in planes:
            planes[m] = _hull_planes(sk, m)
        return planes[m]
    mods = sk.src.modules
    for f, m in list(fmod.items()):
        p = mods[m].parent if 0 <= m < len(mods) else -1
        if p < 0:
            continue
        own, par = pl(m), pl(p)
        if not own or not par:
            continue
        c = f.calc_center_median()
        if _outside(own, c) > tol and _outside(par, c) <= tol:
            fmod[f] = p


SKIN_MOD = "pe_skin_mod"     # a face's module while the skin is made


def skin_mesh(form, sk):
    """The formation's own mesh as the skin: one connected mesh, the Form's
    attributes left behind (it is a mesh, not a formation), and each vertex's
    weights {bone joint: weight} (skin_weights).

    Joined where it meets. The Form's mesh has one vertex where modules meet
    (meshplan joins the copies of a corner each module works out), but a
    mesh written before that keeps each module's own copy, a hair apart.
    Seen still, nothing shows; in a skin each module pulls its own copy and
    the mesh came apart at the seams when posed. So the skin also joins every
    two vertices closer than a thousandth of a module's edge (the finest
    thing the Form makes, a carve's ring, is ten times that)."""
    bm = bmesh.new()
    bm.from_mesh(form.data)
    try:
        fmod = face_modules(form, bm, sk)
        mods = [fmod.get(f, 0) for f in bm.faces]
        # (A new layer makes the faces anew: by index, not by the face.)
        lm = bm.faces.layers.int.new(SKIN_MOD)
        bm.faces.ensure_lookup_table()
        for k, m in enumerate(mods):
            bm.faces[k][lm] = m
        tol = sk.E * sk.sc * 1e-3
        bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=tol)
        bm.verts.index_update()
        weights = []
        for v in bm.verts:
            bones = sorted({sk.owner(f[lm]) for f in v.link_faces})
            if not bones:
                bones = [sk.owner(0)]
            w = 1.0 / len(bones)
            weights.append({b: w for b in bones})
        me = bpy.data.meshes.new(form.name + " Skin")
        bm.to_mesh(me)
    finally:
        bm.free()
    for name in SKIN_DROP + (SKIN_MOD,):
        a_ = me.attributes.get(name)
        if a_ is not None:
            me.attributes.remove(a_)
    for mat in form.data.materials:
        me.materials.append(mat)
    if not me.materials:
        me.materials.append(material())
    attr = me.color_attributes.get(COLOR_ATTR)
    if attr is not None:
        me.color_attributes.active_color = attr
    me.update()
    return me, weights


def _skin_rigid(context, form, sk, M, ao, names):
    """Each vertex of one module's faces whole to the bone into that module;
    one where modules meet shared equally between their bones."""
    me, weights = skin_mesh(form, sk)
    skin = bpy.data.objects.new(form.name + " Skin", me)
    _link_like(skin, form)
    skin[RIG_OF] = form
    skin[SKIN_KIND] = "modules"
    skin.parent = ao
    groups = {}
    for vi, ws in enumerate(weights):
        for b, w in ws.items():
            groups.setdefault(names[b], {}).setdefault(w, []).append(vi)
    for nm, by_w in groups.items():
        vg = skin.vertex_groups.new(name=nm)
        for w, vs in by_w.items():
            vg.add(vs, w, "REPLACE")
    mod = skin.modifiers.new("Armature", "ARMATURE")
    mod.object = ao
    return skin


def _skin_auto(context, form, M, ao, sk):
    # The formation's own mesh, modelling and all, joined where it meets,
    # weighted by bone heat.
    me, _ = skin_mesh(form, sk)
    skin = bpy.data.objects.new(form.name + " Skin", me)
    _link_like(skin, form)
    skin[RIG_OF] = form
    skin[SKIN_KIND] = "auto"
    skin.matrix_world = form.matrix_world.copy()
    vl = context.view_layer
    for o in context.selected_objects:
        o.select_set(False)
    skin.select_set(True)
    ao.select_set(True)
    vl.objects.active = ao
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    return skin


def _reweigh(context, skin, ao):
    """Automatic weights worked out again: Blender's bone heat from the joints
    as they stand now (edited)."""
    vl = context.view_layer
    sel, hidden = skin.hide_select, skin.hide_get()
    skin.hide_select = False
    skin.hide_set(False)
    skin.vertex_groups.clear()
    for m in [m for m in skin.modifiers if m.type == "ARMATURE"]:
        skin.modifiers.remove(m)
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    for o in context.selected_objects:
        o.select_set(False)
    skin.select_set(True)
    ao.select_set(True)
    vl.objects.active = ao
    bpy.ops.object.parent_set(type="ARMATURE_AUTO", keep_transform=True)
    skin.select_set(False)
    skin.hide_select = sel
    skin.hide_set(hidden)


# ── the three steps ──────────────────────────────────────────────────────────
def stage_of(arm):
    """The step skeleton `arm` is at: "skeleton" (1), "controls" (2) or
    "pose" (3). One from before the steps: posing if it has a control rig."""
    s = _plain(arm.get(RIG_SET)).get("stage")
    if s in (STAGE_SKELETON, STAGE_CONTROLS, STAGE_POSE):
        return s
    return STAGE_POSE if has_controls(arm) else STAGE_SKELETON


def _set_stage(arm, stage):
    st = _plain(arm.get(RIG_SET))
    st["stage"] = stage
    for k in ("seams", "hinge"):
        st.pop(k, None)
    arm[RIG_SET] = st


def _show(context, form):
    """The formation's skeleton and skin shown, the formation hidden, the
    skeleton active in Object Mode. (arm, skin)"""
    arm, skin = rig_objects(form)
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    form.hide_set(True)
    for o in (arm, skin):
        if o is not None:
            o.hide_set(False)
    for o in context.selected_objects:
        o.select_set(False)
    arm.select_set(True)
    context.view_layer.objects.active = arm
    return arm, skin


def _bones_shown(arm, modules=True, snap=False):
    """The modules' bones shown or not, the controls shown, the mechanism
    hidden, the snap points shown only while the skeleton is edited."""
    for name, on in ((MOD_COLL, modules), (CTRL_COLL, True), (MCH_COLL, False), (SNAP_COLL, snap)):
        bc = arm.data.collections.get(name)
        if bc is not None:
            bc.is_visible = on


def _tidy(form):
    """What the version before left while the skeleton was edited: a snap
    points object beside it, and the skin made unselectable."""
    _, skin = rig_objects(form)
    if skin is not None and SEL_WAS in skin.keys():
        skin.hide_select = bool(skin[SEL_WAS])
        del skin[SEL_WAS]
    _remove(snap_points(form))
    if form is not None and SNAP_PROP in form.keys():
        del form[SNAP_PROP]


def leave_skeleton(context, form):
    """Step 1 left: the scene's snapping as it was, the snap points hidden
    (they go with the control rig)."""
    arm, _ = rig_objects(form)
    if arm is not None:
        _snap_off(context.scene.tool_settings, arm)
        bc = arm.data.collections.get(SNAP_COLL)
        if bc is not None:
            bc.is_visible = False
    _tidy(form)


def enter_skeleton(context, form):
    """Step 1: the skeleton in Edit Mode, to be edited and typed: its control
    rig taken off (and kept for the next one), the snap points made and the
    snapping set to take a joint being moved onto them."""
    arm, skin = _show(context, form)
    _tidy(form)
    remove_controls(context, arm)
    _snap_on(context.scene.tool_settings, arm)
    _bones_shown(arm, modules=True, snap=True)
    _set_stage(arm, STAGE_SKELETON)
    bpy.ops.object.mode_set(mode="EDIT")
    make_snap_bones(arm, skeleton_of(form, arm))
    _MODES[arm.as_pointer()] = "EDIT"
    return arm, skin


def enter_controls(context, form):
    """Step 2: the controls in Edit Mode, to be edited; the modules' bones
    hidden meanwhile (they are the skeleton, step 1)."""
    arm, skin = _show(context, form)
    leave_skeleton(context, form)
    _bones_shown(arm, modules=False)
    _set_stage(arm, STAGE_CONTROLS)
    bpy.ops.object.mode_set(mode="EDIT")
    return arm, skin


def enter_pose(context, form):
    """Step 3: Pose Mode, each knee's pole fitted to the controls as they
    were left."""
    arm, skin = _show(context, form)
    leave_skeleton(context, form)
    fit_controls(arm)
    _bones_shown(arm, modules=True)
    _set_stage(arm, STAGE_POSE)
    bpy.ops.object.mode_set(mode="POSE")
    return arm, skin


def enter_heat(context, form):
    """The skin's weights as a heat map: the skin in Weight Paint, the
    skeleton beside it in Pose Mode (Ctrl+click a bone to see its share of the
    skin), its active bone's share shown."""
    arm, skin = _show(context, form)
    leave_skeleton(context, form)
    _bones_shown(arm, modules=True)
    b = arm.data.bones.active
    if b is None or is_control(b):
        b = next((x for x in module_bones(arm) if x.name in skin.vertex_groups), None)
        if b is not None:
            arm.data.bones.active = b
    # Blender puts a selected armature the skin deforms by in Pose Mode as
    # the skin goes into Weight Paint.
    skin.hide_select = False
    for o in context.selected_objects:
        o.select_set(False)
    arm.select_set(True)
    skin.select_set(True)
    context.view_layer.objects.active = skin
    bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
    if b is not None and b.name in skin.vertex_groups:
        skin.vertex_groups.active_index = skin.vertex_groups[b.name].index
    return arm, skin


# Blender's own mode switches follow the steps: Tab into the skeleton's Edit
# Mode at step 1 turns the snapping onto the snap points on, and leaving it
# (Tab, Ctrl+Tab, the mode menu) gives the scene its own snapping back. Seen
# after the depsgraph updates a mode change makes, and through the message
# bus (the UI's own notice of a mode change).
_MODES = {}


def _follow_modes(scene=None):
    ts = (scene or bpy.context.scene).tool_settings
    for ob in bpy.data.objects:
        if ob.type != "ARMATURE" or RIG_SET not in ob.keys():
            continue
        key, m = ob.as_pointer(), ob.mode
        prev = _MODES.get(key)
        if prev == m:
            continue
        _MODES[key] = m
        if prev is None or stage_of(ob) != STAGE_SKELETON:
            continue
        if m == "EDIT":
            _snap_on(ts, ob)
            bc = ob.data.collections.get(SNAP_COLL)
            if bc is not None:
                bc.is_visible = True
        elif prev == "EDIT":
            _snap_off(ts, ob)


def _on_mode(*_):
    try:
        _follow_modes()
    except (ReferenceError, RuntimeError, AttributeError):
        pass


class PE_OT_rig_heat(bpy.types.Operator):
    """The skin's weights as a heat map: Weight Paint on the skin, the skeleton beside it in Pose Mode; Ctrl+click a bone to see the share of the skin it moves, and paint to change it"""
    bl_idname = "pe.rig_heat"
    bl_label = "Heat Map"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        f = formation_of(context)
        return f is not None and all(o is not None for o in rig_objects(f))

    def execute(self, context):
        arm, skin = enter_heat(context, formation_of(context))
        g = skin.vertex_groups.active
        self.report({"INFO"}, "Heat map%s: Ctrl+click a bone for its share of the skin, paint to change it"
                    % (" of " + g.name if g is not None else ""))
        return {"FINISHED"}


def _rig_poll(context, controls=False):
    f = formation_of(context)
    if f is None:
        return False
    arm = rig_objects(f)[0]
    return arm is not None and (not controls or has_controls(arm))


class PE_OT_skeleton(bpy.types.Operator):
    """Step 1: a joint on every module's centre and a bone to every module built on it, as the Theater draws it, and the formation skinned to it. It opens in Edit Mode, to move its joints and type its bones"""
    bl_idname = "pe.skeleton"
    bl_label = "Skeleton from Modules"
    bl_options = {"REGISTER", "UNDO"}

    keep: BoolProperty(name="Keep Moved Joints", default=True,
                       description="The joints moved by hand stay where they were put, while their module is "
                                   "still there; off: every joint back on its module's centre")

    @classmethod
    def poll(cls, context):
        f = formation_of(context)
        return f is not None and Formation.of(f).state.engine.module_count > 0

    def execute(self, context):
        wm = context.window_manager
        form = formation_of(context)
        if context.mode == "EDIT_MESH":
            follow_selection(form)
        st = Formation.of(form).state
        if not any(not h.carve for h in st.engine.history):
            self.report({"WARNING"}, "No modules yet: place one first")
            return {"CANCELLED"}
        M = read_frame(form)
        old_arm, old_skin = rig_objects(form)
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        sk = Skeleton(st.engine, form.name, M, _root_of(old_arm) if old_arm is not None else None)
        # Every joint where the modules put it; a joint moved by hand stays
        # where it was put while its module is still there.
        home = {sk.joints[i].name: tuple(bone_ends(sk, i, M)[1]) for i in sk.bones()}
        if self.keep and old_arm is not None:
            sk.move(_kept_joints(old_arm, home, sk.E * sk.sc * 1e-4))
        # The poses made on the old skeleton are kept for the new one (its
        # bones have the same names: the module and its number), and so are
        # the bones' types, and what was done to its controls by hand.
        skel_id = old_arm.get(SKEL_ID) if old_arm is not None else None
        old_types = bone_types(old_arm) if old_arm is not None else {}
        edits, poses = {}, {}
        if old_arm is not None:
            leave_skeleton(context, form)
            edits, poses = _ctrl_keep(old_arm)
        if not skel_id:
            import uuid
            skel_id = str(uuid.uuid4())
        _remove(old_skin)
        _remove(old_arm)
        form.hide_set(False)
        form.hide_viewport = False   # a formation from an asset library comes hidden under its skin
        ao, names = build_armature(context, form, sk, M, home)
        for nm, t in old_types.items():
            if t and nm in ao.data.bones:
                set_bone_type(ao, sk, nm, t)
        ao[CTRL_EDITS] = edits
        ao[CTRL_POSE] = poses
        if wm.pe_rig_skin == "auto":
            skin = _skin_auto(context, form, M, ao, sk)
        else:
            skin = _skin_rigid(context, form, sk, M, ao, names)
        summary = _summary(sk, st.engine)
        ao[RIG_SUMMARY] = summary
        ao[RIG_WID] = form.get("pe_wid", 0)
        ao[SKEL_ID] = skel_id
        ao[RIG_SET] = {"root": sk.joints[sk.root].name, "stage": STAGE_SKELETON}
        form[RIG_PROP], form[SKIN_PROP] = ao, skin
        enter_skeleton(context, form)
        self.report({"INFO"}, "Skeleton: " + summary + ". Edit it: move the joints (they snap to the modules' "
                    "centres and seams) and type the bones (Leg for IK); then Make Control Rig")
        return {"FINISHED"}


def _summary(sk, engine):
    n, c = len(sk.joints), engine.carve_count
    out = "%d bone%s, one per module, rooted at %s" % (n, "" if n == 1 else "s", sk.joints[sk.root].name)
    if sk.moved:
        out += "; %d joint%s moved off %s module%s centre" % (sk.moved, "" if sk.moved == 1 else "s",
                                                             "its" if sk.moved == 1 else "their",
                                                             "'s" if sk.moved == 1 else "s'")
    if c:
        out += " (%d carve%s: no bone)" % (c, "" if c == 1 else "s")
    return out


def _root_of(arm):
    """The joint an armature is rooted at, by name (None: the first module)."""
    return _plain(arm.get(RIG_SET)).get("root") or None


def skeleton_of(form, arm):
    """The Skeleton an armature was made from, as it stands now: rooted where
    it is, its joints where they were moved."""
    return Skeleton(Formation.of(form).state.engine, form.name, read_frame(form), _root_of(arm), joints_of(arm))


def _active_bone(arm):
    """The active bone (an edit bone in Edit Mode), or None."""
    if arm.mode == "EDIT":
        return arm.data.edit_bones.active
    return arm.data.bones.active


class PE_OT_edit_skeleton(bpy.types.Operator):
    """Step 1: the skeleton in Edit Mode, to move its joints (they snap to the modules' centres and seams) and type its bones. A control rig on it comes off while it is edited, and is made again from it with Make Control Rig, what you did to the controls kept"""
    bl_idname = "pe.edit_skeleton"
    bl_label = "Edit Skeleton"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _rig_poll(context)

    def execute(self, context):
        form = formation_of(context)
        had = has_controls(rig_objects(form)[0])
        enter_skeleton(context, form)
        msg = "Edit the skeleton: move the joints (they snap to the modules' centres and seams), type the bones"
        if had:
            msg += ". Its control rig is off meanwhile: Make Control Rig puts it back"
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class PE_OT_make_root(bpy.types.Operator):
    """Make the active bone's module the root: the skeleton hung from it (the Theater's Make root), its bones made again with their types, poses and the joints you moved kept"""
    bl_idname = "pe.make_root"
    bl_label = "Make Root"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        f = formation_of(context)
        if f is None:
            return False
        arm = rig_objects(f)[0]
        return arm is not None and _active_bone(arm) is not None

    def execute(self, context):
        form = formation_of(context)
        arm, _ = rig_objects(form)
        b = _active_bone(arm)
        if b.get(CTRL):
            self.report({"WARNING"}, "Pick a module's bone: a control has no module to root at")
            return {"CANCELLED"}
        name = b.name
        sk = Skeleton(Formation.of(form).state.engine, form.name, read_frame(form), _root_of(arm))
        if sk.joints[sk.root].name == name:
            self.report({"INFO"}, "%s is the root already" % name)
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        st = _plain(arm.get(RIG_SET))
        st["root"] = name
        arm[RIG_SET] = st
        r = bpy.ops.pe.skeleton(keep=True)
        if r != {"FINISHED"}:
            return r
        self.report({"INFO"}, "Rooted at %s" % name)
        return {"FINISHED"}


class PE_OT_control_rig(bpy.types.Operator):
    """Step 2: the control rig, made from the skeleton as you edited and typed it: a root on the floor everything hangs from, and for every Leg and Paw chain an IK foot control at its ankle and a knee pole, with an IK/FK switch. It opens in Edit Mode on the controls, to edit them in turn; what you do to them is kept through every control rig made after"""
    bl_idname = "pe.control_rig"
    bl_label = "Make Control Rig"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _rig_poll(context)

    def execute(self, context):
        form = formation_of(context)
        arm, skin = _show(context, form)
        leave_skeleton(context, form)
        sk = skeleton_of(form, arm)
        build_controls(context, arm, sk)
        if skin is not None and skin.get(SKIN_KIND) == "auto":
            _reweigh(context, skin, arm)
        arm[RIG_SUMMARY] = _summary(sk, Formation.of(form).state.engine)
        enter_controls(context, form)
        self.report({"INFO"}, controls_note(arm) + ". Edit the controls here (move a pole, turn a foot), then Pose")
        return {"FINISHED"}


class PE_OT_edit_controls(bpy.types.Operator):
    """Step 2: the controls in Edit Mode, to move a pole, turn a foot control or move the root; the modules' bones hidden meanwhile"""
    bl_idname = "pe.edit_controls"
    bl_label = "Edit Controls"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _rig_poll(context, controls=True)

    def execute(self, context):
        enter_controls(context, formation_of(context))
        return {"FINISHED"}


class PE_OT_rig_pose(bpy.types.Operator):
    """Step 3: pose it, in Pose Mode: each knee's pole fitted to the controls as you left them"""
    bl_idname = "pe.rig_pose"
    bl_label = "Pose"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _rig_poll(context, controls=True)

    def execute(self, context):
        enter_pose(context, formation_of(context))
        return {"FINISHED"}


def _joint_moves(sk, nx, Js):
    """Each joint's move in the neutral pose (armature space): its turn,
    about its place, lifted."""
    from mathutils import Matrix, Quaternion, Vector
    from ..engine import neutral as N
    pos, rot = N.fk_pose(Js, nx.pose, nx.lift)
    sc = sk.sc

    def bl(v):
        return Vector(to_bl(v)) * sc
    X = []
    for j, p, q in zip(Js, pos, rot):
        R_ = Quaternion((q.w, q.x, -q.z, q.y)).to_matrix().to_4x4()
        X.append(Matrix.Translation(bl(p)) @ R_ @ Matrix.Translation(-bl(j.pos)))
    return X


def neutral_basis(sk, nx, Js, arm):
    """Each bone's pose (matrix_basis) for the neutral pose: joint j's turn,
    about its place, lifted, carried into the armature's space; a bone turns
    as the joint at its head does, after the bone it hangs from. The
    controls: the root at rest, a foot control where its ankle goes, turned
    as its foot is, a knee's pole carried with the leg from where it is
    joined on (so the legs' IK puts them where the neutral pose does)."""
    from mathutils import Matrix
    from ..engine import neutral as N
    X = _joint_moves(sk, nx, Js)
    J = sk.joints
    by_name = {j.name: i for i, j in enumerate(J)}
    out = {}
    for b in arm.data.bones:
        i = by_name.get(b.name)
        if i is None:
            continue
        B = b.matrix_local
        Xj = X[sk.head_joint(i)]
        if b.parent is not None and b.parent.name in by_name:
            Xk = X[sk.head_joint(by_name[b.parent.name])]
            out[b.name] = B.inverted() @ Xk.inverted() @ Xj @ B
        else:
            out[b.name] = B.inverted() @ Xj @ B
    bones = arm.data.bones
    if ROOT_BONE in bones and is_control(bones[ROOT_BONE]):
        out[ROOT_BONE] = Matrix.Identity(4)
    for p in N.limb_plans(Js, N.motion_ctx(Js)):
        if p.type not in ("leg", "paw"):
            continue
        tip = J[p.path[p.end]].name
        for nm, j in (("IK." + tip, p.path[p.end]), ("Pole." + tip, p.path[0])):
            b = bones.get(nm)
            if b is not None and is_control(b):
                B = b.matrix_local
                out[nm] = B.inverted() @ X[j] @ B
    return out


class PE_OT_pose_neutral(bpy.types.Operator):
    """The Theater's neutral pose, the pose its motions play from: legs and paws straight down, arms hanging clear of the body, wings level out to the sides, the spine, head and tail as built, and the body lifted to stand on its floor"""
    bl_idname = "pe.pose_neutral"
    bl_label = "Neutral Pose"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _rig_poll(context)

    def execute(self, context):
        form = formation_of(context)
        arm, _ = rig_objects(form)
        if arm.mode == "EDIT":
            self.report({"WARNING"}, "Pose it first (step 3): the neutral pose is a pose")
            return {"CANCELLED"}
        sk = skeleton_of(form, arm)
        names = {j.name for j in sk.joints}
        if any(b.name not in names for b in module_bones(arm)):
            self.report({"WARNING"}, "The formation changed since the skeleton was made: make it again")
            return {"CANCELLED"}
        nx, Js, guessed = sk.neutral(bone_types(arm))
        if nx is None:
            self.report({"WARNING"}, "No legs, arms or wings found: give the limbs a Bone Type")
            return {"CANCELLED"}
        for name, M in neutral_basis(sk, nx, Js, arm).items():
            pb = arm.pose.bones[name]
            pb.rotation_mode = "QUATERNION"
            pb.matrix_basis = M
        arm.update_tag()
        msg = "Neutral pose: %d limb%s put in place" % (nx.turned, "" if nx.turned == 1 else "s")
        if nx.raised:
            msg += ", %d arm%s raised clear of the body" % (nx.raised, "" if nx.raised == 1 else "s")
        if nx.kept:
            msg += ", %d arm%s left as built (already clear)" % (nx.kept, "" if nx.kept == 1 else "s")
        if guessed:
            msg += " (limbs read from its shape: give bones a Bone Type to choose them)"
        self.report({"INFO"}, msg + ". Save it as a pose, or Pose > Apply > Apply Pose as Rest Pose to rest in it")
        return {"FINISHED"}


class PE_OT_bone_type(bpy.types.Operator):
    """Give the selected bones a type, as the Theater's Bones panel does: it tints them, files them in a bone collection of that type, the control rig puts IK on the Leg and Paw chains, and the Neutral Pose puts legs, arms and wings in place by it"""
    bl_idname = "pe.bone_type"
    bl_label = "Bone Type"
    bl_options = {"REGISTER", "UNDO"}

    type: EnumProperty(name="Type", items=TYPE_ITEMS)

    @classmethod
    def poll(cls, context):
        return _rig_poll(context)

    def execute(self, context):
        form = formation_of(context)
        arm, _ = rig_objects(form)
        if context.mode == "POSE" and context.active_object == arm:
            names = [pb.name for pb in context.selected_pose_bones or ()]
        elif context.mode == "EDIT_ARMATURE" and context.active_object == arm:
            names = [eb.name for eb in arm.data.edit_bones if eb.select and not eb.hide]
        else:
            names = [b.name for b in arm.data.bones if b.select]
        names = [nm for nm in names if nm in arm.data.bones and not is_control(arm.data.bones[nm])]
        if not names:
            self.report({"WARNING"}, "Select the modules' bones first")
            return {"CANCELLED"}
        edit = context.mode == "EDIT_ARMATURE"
        if edit:
            bpy.ops.object.mode_set(mode="POSE")
        sk = skeleton_of(form, arm)
        for nm in names:
            set_bone_type(arm, sk, nm, self.type)
        ctrl = has_controls(arm)
        if ctrl:
            build_controls(context, arm, sk)
        if edit:
            bpy.ops.object.mode_set(mode="EDIT")
        label = next((lab for k, lab, _ in TYPE_ITEMS if k == self.type), self.type)
        self.report({"INFO"}, "%d bone%s: %s%s" % (len(names), "" if len(names) == 1 else "s", label,
                                                   ". " + controls_note(arm) if ctrl else ""))
        return {"FINISHED"}


class PE_OT_rig_back(bpy.types.Operator):
    """Back to the formation: its skeleton and skin are hidden and the formation opens in Edit Mode"""
    bl_idname = "pe.rig_back"
    bl_label = "Back to the Formation"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return formation_of(context) is not None and edit_formation(context) is None

    def execute(self, context):
        form = formation_of(context)
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        leave_skeleton(context, form)
        arm, skin = rig_objects(form)
        for o in (arm, skin):
            if o is not None:
                o.hide_set(True)
        form.hide_set(False)
        form.hide_viewport = False   # a formation from an asset library comes hidden under its skin
        for o in context.selected_objects:
            o.select_set(False)
        form.select_set(True)
        context.view_layer.objects.active = form
        bpy.ops.object.mode_set(mode="EDIT")
        context.tool_settings.mesh_select_mode = (False, False, True)
        return {"FINISHED"}


def _step(lay, n, title, icon, on):
    """A step's box, its title lit while it is the step being done."""
    box = lay.box()
    row = box.row()
    row.label(text="%d  %s" % (n, title), icon=icon)
    if on:
        row.label(text="", icon="RADIOBUT_ON")
    return box


class PE_PT_rig(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PE"
    bl_label = "Skeleton"
    bl_idname = "PE_PT_rig"
    bl_order = 3

    def draw_header(self, context):
        from .panels import pe_icon
        self.layout.label(text="", icon_value=pe_icon())

    @classmethod
    def poll(cls, context):
        return formation_of(context) is not None

    def draw(self, context):
        lay = self.layout
        wm = context.window_manager
        form = formation_of(context)
        arm, skin = rig_objects(form)
        editing = edit_formation(context) is not None
        col = lay.column()
        col.label(text="Skin")
        col.row(align=True).prop(wm, "pe_rig_skin", expand=True)
        painting = skin is not None and context.mode == "PAINT_WEIGHT" and context.active_object == skin
        if arm is not None and skin is not None:
            col.operator("pe.rig_heat", text="Heat Map", icon="WPAINT_HLT", depress=painting)
            if painting:
                _wrap(col, "Ctrl+click a bone for the share of the skin it moves (red all, blue none); "
                           "paint to change it")
        row = lay.row()
        row.scale_y = 1.6
        row.operator("pe.skeleton", text="Skeleton from Modules", icon="ARMATURE_DATA", depress=arm is None)
        if arm is None:
            if not editing:
                lay.operator("pe.rig_back", text="Back to the Formation", icon="EDITMODE_HLT")
            return
        if arm.get(RIG_WID) != form.get("pe_wid"):
            lay.label(text="The formation changed since: make the skeleton again", icon="ERROR")
        stage = stage_of(arm)
        here = not editing and context.active_object == arm
        mode = context.mode
        ctrl = has_controls(arm)
        # 1  the skeleton, edited and typed
        doing = here and mode == "EDIT_ARMATURE" and stage == STAGE_SKELETON
        box = _step(lay, 1, "Skeleton", "BONE_DATA", doing)
        _wrap(box, arm.get(RIG_SUMMARY, ""))
        if doing:
            ts = context.scene.tool_settings
            box.prop(ts, "use_snap", text="Snap Joints to the Modules", toggle=True,
                     icon="SNAP_ON" if ts.use_snap else "SNAP_OFF")
            _wrap(box, "A joint being moved snaps to a module's centre (white) or a seam (gold)")
            ab = _active_bone(arm)
            t = (ab.get(BONE_TYPE, "") or "none") if ab is not None else "none"
            label = next((lab for k, lab, _ in TYPE_ITEMS if k == t), t) if ab is not None else "-"
            row = box.row(align=True)
            row.label(text="Bone Type")
            row.operator_menu_enum("pe.bone_type", "type", text=label, icon="BONE_DATA")
            row = box.row(align=True)
            row.label(text="Root")
            row.operator("pe.make_root", text="Make Root", icon="CON_ARMATURE")
        else:
            box.operator("pe.edit_skeleton", text="Edit Skeleton", icon="EDITMODE_HLT")
        # 2  the control rig, made from it and edited
        doing = here and mode == "EDIT_ARMATURE" and stage == STAGE_CONTROLS
        box = _step(lay, 2, "Control Rig", "CON_KINEMATIC", doing)
        if ctrl:
            _wrap(box, controls_note(arm))
        row = box.row(align=True)
        row.operator("pe.control_rig", text="Control Rig Again" if ctrl else "Make Control Rig",
                     icon="OUTLINER_DATA_ARMATURE", depress=stage == STAGE_SKELETON)
        if ctrl and not doing:
            row.operator("pe.edit_controls", text="Edit Controls", icon="EDITMODE_HLT")
        if doing:
            _wrap(box, "Move or turn the controls (a pole, a foot control, the root); "
                       "what you do to them is kept through a control rig made again")
        # 3  pose it
        doing = here and mode == "POSE"
        box = _step(lay, 3, "Pose", "POSE_HLT", doing)
        if not ctrl:
            _wrap(box, "Make the control rig first")
        elif not doing:
            row = box.row()
            row.operator("pe.rig_pose", text="Pose", icon="POSE_HLT", depress=stage == STAGE_CONTROLS)
        else:
            ab = arm.data.bones.active
            pb = arm.pose.bones.get(ab.name) if ab is not None else None
            if pb is not None and IK_PROP in pb.keys():
                row = box.row(align=True)
                row.label(text="Leg FK / IK")
                row.prop(pb, '["%s"]' % IK_PROP, text="", slider=True)
            else:
                _wrap(box, "Pick a foot control for its leg's FK / IK switch")
        if not editing:
            lay.operator("pe.rig_back", text="Back to the Formation", icon="EDITMODE_HLT")


PROPS = {
    "pe_rig_skin": EnumProperty(name="Skin", items=SKIN_ITEMS, default="modules",
                                description="How the shell follows the bones"),
}

CLASSES = (PE_OT_skeleton, PE_OT_edit_skeleton, PE_OT_make_root, PE_OT_control_rig, PE_OT_edit_controls,
           PE_OT_rig_pose, PE_OT_rig_heat, PE_OT_pose_neutral, PE_OT_bone_type, PE_OT_rig_back, PE_PT_rig)


@persistent
def _after_depsgraph(scene, depsgraph):
    # A control rig's rest changed (its controls edited, by Edit Controls or
    # Blender's own Tab): its knees' poles fitted again once out of Edit Mode,
    # so a leg still rests as it was made.
    try:
        _follow_modes(scene)
    except (ReferenceError, RuntimeError, AttributeError):
        pass
    arms = {u.id.original for u in depsgraph.updates if isinstance(u.id, bpy.types.Armature)}
    if not arms:
        return
    for ob in bpy.data.objects:
        if ob.type == "ARMATURE" and ob.data in arms and ob.mode != "EDIT" and CTRL_SET in ob.keys():
            try:
                fit_controls(ob)
            except (ReferenceError, RuntimeError, AttributeError):
                pass


_draw_handle = None
_BUS = object()


def _listen():
    bpy.msgbus.clear_by_owner(_BUS)
    bpy.msgbus.subscribe_rna(key=(bpy.types.Object, "mode"), owner=_BUS, args=(), notify=_on_mode)


@persistent
def _after_load(*_):
    # A file opened: its objects are new to the mode follower, and the
    # message bus forgets what it was listening to.
    _MODES.clear()
    _listen()


def register():
    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_snaps, (), "WINDOW", "POST_VIEW")
    for lst, fn in ((bpy.app.handlers.depsgraph_update_post, _after_depsgraph),
                    (bpy.app.handlers.load_post, _after_load)):
        if fn not in lst:
            lst.append(fn)
    _listen()


def unregister():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None
    for lst, fn in ((bpy.app.handlers.depsgraph_update_post, _after_depsgraph),
                    (bpy.app.handlers.load_post, _after_load)):
        if fn in lst:
            lst.remove(fn)
    bpy.msgbus.clear_by_owner(_BUS)
