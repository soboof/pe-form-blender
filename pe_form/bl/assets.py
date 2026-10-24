"""Poses and the asset library.

Poses: a pose of the skeleton saved under a name is an Action keyed on every
bone, marked as an asset: what Blender's own Pose Library makes, so its asset
shelf applies it as well as the Poses list here. A pose belongs to the
skeleton it was made on (`pe_skel_id`, kept when the skeleton is made again,
whose bones keep their names), and is named "<formation>: <pose>".

The asset library: Save to Asset Library writes the formation, its skeleton
and skin as one Collection asset (catalog "Pythagoras Engine/Formations"),
and every pose made on its skeleton as pose assets (catalog "Pythagoras
Engine/Poses/<formation>"), into "<library>/PE Formations/<name>.blend";
saving again replaces that file. The catalogs are added to the library's
catalog file. Previews are drawn here (engine/snapshot.py), so they are in
the file when it is written. The formation is in the asset as a formation:
dropped into a scene it opens in Edit Mode like any other, hidden under its
skin while the skeleton is there. Dropped as an instance (Blender's default
for a collection), Make Editable turns the instance into its own objects.
"""

import os
import re
import subprocess
import tempfile
import uuid

import bpy
from bpy.props import StringProperty, EnumProperty

from ..engine.snapshot import snapshot
from .formation import Formation, is_formation
from .rigging import RIG_PROP, SKIN_PROP, SKEL_ID, SNAP_PROP, SEL_WAS, _obj, formation_of, rig_objects

POSE_NAME = "pe_pose"
POSES_PROP = "pe_poses"          # on the skeleton: its poses, so they come with it from the library
ASSET_PROP = "pe_asset"          # on the formation: the collection it is saved as
ASSET_PATH = "pe_asset_path"     # on the formation: where it was last saved
ROOT = "Pythagoras Engine"
FOLDER = "PE Formations"
CATS_FILE = "blender_assets.cats.txt"
PREVIEW = 128


# ── catalogs ─────────────────────────────────────────────────────────────────
def _clean(name):
    return re.sub(r"[:/\\\r\n]+", " ", name).strip() or "Formation"


def catalog_id(path):
    """A catalog's id, the same every time for the same path."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "https://pythagorasengine.com/assets/" + path))


def catalog_paths(name):
    return ROOT + "/Formations", ROOT + "/Poses/" + _clean(name)


def ensure_catalogs(root, paths):
    """Add the catalogs (and their parents) to the library's catalog file."""
    fn = os.path.join(root, CATS_FILE)
    lines = []
    if os.path.exists(fn):
        with open(fn, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    have = set()
    for ln in lines:
        if ln and not ln.startswith("#") and not ln.startswith("VERSION") and ":" in ln:
            parts = ln.split(":")
            if len(parts) >= 2:
                have.add(parts[1])
    want = []
    for p in paths:
        bits = p.split("/")
        for k in range(1, len(bits) + 1):
            q = "/".join(bits[:k])
            if q not in have and q not in want:
                want.append(q)
    if not want and lines:
        return
    if not any(ln.startswith("VERSION") for ln in lines):
        lines = ["# This is an Asset Catalog Definition file for Blender.", "#",
                 "# Empty lines and lines starting with `#` will be ignored.",
                 "# The first non-ignored line should be the version indicator.",
                 '# Other lines are of the format "UUID:catalog/path/for/assets:simple catalog name"',
                 "", "VERSION 1", ""] + lines
    for q in want:
        lines.append("%s:%s:%s" % (catalog_id(q), q, q.replace("/", "-")))
    with open(fn, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ── previews ─────────────────────────────────────────────────────────────────
def _mesh_tris(me):
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    if not n:
        return [], []
    co = [0.0] * (3 * len(me.vertices))
    me.vertices.foreach_get("co", co)
    vi = [0] * (3 * n)
    me.loop_triangles.foreach_get("vertices", vi)
    pts = [(co[3 * i], co[3 * i + 1], co[3 * i + 2]) for i in vi]
    tris = [pts[3 * k:3 * k + 3] for k in range(n)]
    attr = me.color_attributes.get("PE Color")
    cols = []
    if attr is not None and attr.domain == "CORNER":
        li = [0] * (3 * n)
        me.loop_triangles.foreach_get("loops", li)
        for k in range(n):
            c = attr.data[li[3 * k]].color_srgb
            cols.append((c[0], c[1], c[2]))
    else:
        cols = [(0.75, 0.72, 0.66)] * n
    return tris, cols


def object_tris(obj, evaluated=True):
    """The object's triangles in its own space, posed when it is skinned."""
    if obj.mode == "EDIT":
        obj.update_from_editmode()
        if not evaluated:
            # In Edit Mode a mesh's attributes live in the edit mesh, and the
            # object's own read back empty (its colours: index out of range),
            # so the edit mesh is read through a copy.
            import bmesh
            tmp = bpy.data.meshes.new("pe_preview_tmp")
            try:
                bmesh.from_edit_mesh(obj.data).to_mesh(tmp)
                return _mesh_tris(tmp)
            finally:
                bpy.data.meshes.remove(tmp)
    if not evaluated:
        return _mesh_tris(obj.data)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        return _mesh_tris(me)
    finally:
        ev.to_mesh_clear()


def set_preview(idb, tris, cols):
    """Draw the preview; returns its pixels (bottom row first), or None."""
    if not tris:
        return None
    img = snapshot(tris, cols, size=PREVIEW)
    px = img[::-1].ravel().tolist()      # bottom row first
    pv = idb.preview_ensure()
    pv.image_size = (PREVIEW, PREVIEW)
    pv.image_pixels_float = px
    return px


# Blender's partial write (bpy.data.libraries.write) keeps an action's or an
# object's preview but drops a collection's, so the written file is opened
# once in a background Blender, given the collection's preview, and saved.
_PATCH = """
import sys, bpy
path, name, raw = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.open_mainfile(filepath=path)
c = bpy.data.collections.get(name)
if c is not None:
    with open(raw, encoding="utf-8") as fh:
        px = [float(x) for x in fh.read().split()]
    pv = c.preview_ensure()
    pv.image_size = (%d, %d)
    pv.image_pixels_float = px
    bpy.ops.wm.save_mainfile(filepath=path)
""" % (PREVIEW, PREVIEW)


def _write_collection_preview(path, name, px):
    if not px or not bpy.app.binary_path:
        return False
    fd, raw = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(" ".join("%.4f" % v for v in px))
        r = subprocess.run([bpy.app.binary_path, "-b", "--factory-startup", "--python-expr", _PATCH,
                            "--", path, name, raw], capture_output=True, timeout=120)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        try:
            os.remove(raw)
        except OSError:
            pass
        for leftover in (path + "1",):
            if os.path.exists(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass


# ── poses ────────────────────────────────────────────────────────────────────
def poses_of(arm):
    sid = arm.get(SKEL_ID) if arm is not None else None
    if not sid:
        return []
    return sorted((a for a in bpy.data.actions if a.get(SKEL_ID) == sid), key=lambda a: a.get(POSE_NAME, a.name).lower())


def _channels(pb):
    m = pb.rotation_mode
    rot = ("rotation_quaternion", 4) if m == "QUATERNION" else (
        ("rotation_axis_angle", 4) if m == "AXIS_ANGLE" else ("rotation_euler", 3))
    return (("location", 3), rot, ("scale", 3))


def _skeleton(context):
    form = formation_of(context)
    arm, skin = rig_objects(form)
    return form, arm, skin


def apply_pose(arm, act):
    for fc in act.fcurves:
        m = re.match(r'pose\.bones\["(.+)"\]\.(\w+)$', fc.data_path)
        if not m:
            continue
        pb = arm.pose.bones.get(m.group(1).replace('\\"', '"'))
        if pb is None:
            continue
        prop = m.group(2)
        if prop == "rotation_quaternion":
            pb.rotation_mode = "QUATERNION"
        elif prop == "rotation_axis_angle":
            pb.rotation_mode = "AXIS_ANGLE"
        elif prop == "rotation_euler" and pb.rotation_mode in ("QUATERNION", "AXIS_ANGLE"):
            pb.rotation_mode = "XYZ"
        getattr(pb, prop)[fc.array_index] = fc.evaluate(1)


class PE_OT_pose_save(bpy.types.Operator):
    """Save the skeleton's pose under a name, as a pose asset (Blender's Pose Library applies it too). A name already saved is updated"""
    bl_idname = "pe.pose_save"
    bl_label = "Save Pose"
    bl_options = {"REGISTER", "UNDO"}

    name: StringProperty(name="Name", default="", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _skeleton(context)[1] is not None

    def execute(self, context):
        form, arm, skin = _skeleton(context)
        name = _clean(self.name or context.window_manager.pe_pose_name or "")
        if not self.name and not context.window_manager.pe_pose_name:
            taken = {a.get(POSE_NAME) for a in poses_of(arm)}
            k = 1
            while "Pose %d" % k in taken:
                k += 1
            name = "Pose %d" % k
        act = next((a for a in poses_of(arm) if a.get(POSE_NAME, "").lower() == name.lower()), None)
        if act is None:
            act = bpy.data.actions.new("%s: %s" % (form.name, name))
        else:
            while act.fcurves:
                act.fcurves.remove(act.fcurves[0])
        act[SKEL_ID] = arm[SKEL_ID]
        act[POSE_NAME] = name
        for pb in arm.pose.bones:
            for path, n in _channels(pb):
                val = getattr(pb, path)
                for i in range(n):
                    fc = act.fcurves.new('pose.bones["%s"].%s' % (pb.name.replace('"', '\\"'), path), index=i,
                                         action_group=pb.name)
                    fc.keyframe_points.insert(1, val[i], options={"FAST"})
        act.use_fake_user = True
        if act.asset_data is None:
            act.asset_mark()
        act.asset_data.description = "A pose of the formation %s (Pythagoras Engine)" % form.name
        act.asset_data.catalog_id = catalog_id(catalog_paths(form.name)[1])
        if skin is not None:
            set_preview(act, *object_tris(skin))
        context.window_manager.pe_pose_name = ""
        self.report({"INFO"}, "Pose saved: " + name)
        return {"FINISHED"}


class PE_OT_pose_apply(bpy.types.Operator):
    """Put this pose on the skeleton"""
    bl_idname = "pe.pose_apply"
    bl_label = "Apply Pose"
    bl_options = {"REGISTER", "UNDO"}

    action: StringProperty(options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return _skeleton(context)[1] is not None

    def execute(self, context):
        form, arm, skin = _skeleton(context)
        act = bpy.data.actions.get(self.action)
        if act is None:
            return {"CANCELLED"}
        apply_pose(arm, act)
        arm.update_tag()
        return {"FINISHED"}


class PE_OT_pose_rest(bpy.types.Operator):
    """Put the skeleton back in its rest pose"""
    bl_idname = "pe.pose_rest"
    bl_label = "Rest Pose"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return _skeleton(context)[1] is not None

    def execute(self, context):
        arm = _skeleton(context)[1]
        for pb in arm.pose.bones:
            pb.location = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.rotation_euler = (0, 0, 0)
            pb.rotation_axis_angle = (0, 0, 1, 0)
            pb.scale = (1, 1, 1)
        return {"FINISHED"}


class PE_OT_pose_delete(bpy.types.Operator):
    """Delete this pose"""
    bl_idname = "pe.pose_delete"
    bl_label = "Delete Pose"
    bl_options = {"REGISTER", "UNDO"}

    action: StringProperty(options={"SKIP_SAVE"})

    def execute(self, context):
        act = bpy.data.actions.get(self.action)
        if act is not None:
            bpy.data.actions.remove(act)
        return {"FINISHED"}


# ── the asset library ────────────────────────────────────────────────────────
_LIB_ITEMS = []


def _libraries(self, context):
    global _LIB_ITEMS
    libs = context.preferences.filepaths.asset_libraries
    _LIB_ITEMS = [(lib.name, lib.name, lib.path) for lib in libs] or [("", "No asset library", "")]
    return _LIB_ITEMS


def _library(name):
    libs = bpy.context.preferences.filepaths.asset_libraries
    return libs.get(name) if name else (libs[0] if len(libs) else None)


def save_asset(form, lib_path, name, catalog=None, tags=()):
    """Write the formation (with its skeleton, skin and poses) as assets into
    the library at `lib_path`, in `catalog` (Pythagoras Engine/Formations
    unless said). Returns (file written, collection, poses)."""
    arm, skin = rig_objects(form)
    name = _clean(name or form.name)
    if form.mode == "EDIT":
        form.update_from_editmode()
    coll = form.get(ASSET_PROP)
    if not isinstance(coll, bpy.types.Collection):
        coll = bpy.data.collections.new(name)
        form[ASSET_PROP] = coll
    coll.name = name
    for o in list(coll.objects):
        coll.objects.unlink(o)
    for o in (form, arm, skin):
        if o is not None:
            coll.objects.link(o)
    cat_f, cat_p = catalog_paths(name)
    cat_f = catalog or cat_f
    if coll.asset_data is None:
        coll.asset_mark()
    st = Formation.of(form).state
    poses = poses_of(arm)
    n = st.engine.module_count
    coll.asset_data.description = "A Pythagoras Engine formation: %d module%s%s" % (
        n, "" if n == 1 else "s", (", a skeleton and %d pose%s" % (len(poses), "" if len(poses) == 1 else "s"))
        if arm is not None else "")
    coll.asset_data.catalog_id = catalog_id(cat_f)
    for t in ("Pythagoras Engine", "formation") + (("skeleton",) if arm is not None else ()) + tuple(tags):
        if t not in coll.asset_data.tags:
            coll.asset_data.tags.new(t)
    px = set_preview(coll, *object_tris(form, evaluated=False))
    if arm is not None:
        # The poses ride on the skeleton, so dragging the formation in from
        # the library brings them (they are pose assets of their own too).
        arm[POSES_PROP] = {a.get(POSE_NAME, a.name): a for a in poses}
    for act in poses:
        if act.asset_data is None:
            act.asset_mark()
        act.asset_data.catalog_id = catalog_id(cat_p)
        act.name = "%s: %s" % (name, act.get(POSE_NAME, act.name))
    folder = os.path.join(lib_path, FOLDER)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, re.sub(r'[<>:"/\\|?*]+', "_", name) + ".blend")
    # Under its skin while it has one: what is seen, and posed, is the skin.
    # Written as it is outside Edit Skeleton: the skin selectable as it was,
    # and no snap points (they are made while the skeleton is edited).
    hid = form.hide_viewport
    form.hide_viewport = skin is not None
    snap = form.get(SNAP_PROP)
    if snap is not None:
        del form[SNAP_PROP]
    sel = skin.hide_select if skin is not None else None
    if skin is not None and SEL_WAS in skin.keys():
        skin.hide_select = bool(skin[SEL_WAS])
    try:
        bpy.data.libraries.write(path, {coll, *poses}, path_remap="RELATIVE_ALL", fake_user=True)
    finally:
        form.hide_viewport = hid
        if snap is not None:
            form[SNAP_PROP] = snap
        if skin is not None:
            skin.hide_select = sel
    _write_collection_preview(path, coll.name, px)
    ensure_catalogs(lib_path, (cat_f, cat_p) if poses else (cat_f,))
    form[ASSET_PATH] = path
    return path, coll, poses


# ── the sample formations ───────────────────────────────────────────────────
# The website's published samples, shipped with the add-on as the Form's own
# .json saves (pe_form/samples). Written into an asset library on request,
# each built, previewed and filed in Pythagoras Engine/Samples.
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
SAMPLES_CATALOG = ROOT + "/Samples"


def sample_files():
    try:
        names = sorted(n for n in os.listdir(SAMPLES_DIR) if n.lower().endswith(".json"))
    except OSError:
        return []
    return [os.path.join(SAMPLES_DIR, n) for n in names]


def add_samples(context, lib_path):
    """Every sample formation built and saved into the library at `lib_path`,
    the scene left as it was. Returns the files written."""
    import json
    from ..engine import frame as FR
    from .formation import create, write_mesh
    written = []
    for fn in sample_files():
        with open(fn, encoding="utf-8") as fh:
            doc = json.load(fh)
        name = os.path.splitext(os.path.basename(fn))[0]
        seed = doc.get("seedShape", 3) if isinstance(doc.get("seedShape"), int) else 3
        obj = create(context, seed=seed, name=name)
        try:
            obj.location = (0, 0, 0)
            Formation.of(obj).state.load_doc(doc)
            write_mesh(obj, select=None, fresh=True, frame=FR.from_ground(doc.get("ground")))
            path, coll, _ = save_asset(obj, lib_path, name, catalog=SAMPLES_CATALOG, tags=("sample",))
            written.append(path)
        finally:
            coll = obj.get(ASSET_PROP)
            me = obj.data
            Formation.forget(obj)
            bpy.data.objects.remove(obj, do_unlink=True)
            if isinstance(coll, bpy.types.Collection):
                bpy.data.collections.remove(coll)
            if me is not None and me.users == 0:
                bpy.data.meshes.remove(me)
    return written


class PE_OT_add_samples(bpy.types.Operator):
    """Put the sample formations (the dragon, the Eagle and more) in an asset library: each a formation asset in the Pythagoras Engine/Samples catalog, to drag in and build on"""
    bl_idname = "pe.add_samples"
    bl_label = "Add Sample Formations"
    bl_options = {"REGISTER"}

    library: EnumProperty(name="Library", items=_libraries)

    @classmethod
    def poll(cls, context):
        return bool(sample_files())

    def invoke(self, context, event):
        lib = context.window_manager.pe_asset_lib
        if lib:
            self.library = lib
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        lib = _library(self.library)
        if lib is None or not lib.path:
            self.report({"ERROR"}, "No asset library: add one in Preferences > File Paths > Asset Libraries")
            return {"CANCELLED"}
        vl = context.view_layer
        active = vl.objects.active
        mode = active.mode if active is not None else "OBJECT"
        if context.mode != "OBJECT" and active is not None:
            bpy.ops.object.mode_set(mode="OBJECT")
        selected = list(context.selected_objects)
        try:
            written = add_samples(context, bpy.path.abspath(lib.path))
        except (OSError, RuntimeError) as e:
            self.report({"ERROR"}, "Could not add the samples: %s" % e)
            return {"CANCELLED"}
        finally:
            for o in context.selected_objects:
                o.select_set(False)
            for o in selected:
                if o.name in bpy.data.objects:
                    o.select_set(True)
            if active is not None and active.name in bpy.data.objects:
                vl.objects.active = active
                if mode != "OBJECT":
                    bpy.ops.object.mode_set(mode=mode)
        context.window_manager.pe_asset_lib = lib.name
        self.report({"INFO"}, "%d sample formation%s in %s, catalog %s" % (
            len(written), "" if len(written) == 1 else "s", lib.name, SAMPLES_CATALOG))
        return {"FINISHED"}


class PE_OT_save_asset(bpy.types.Operator):
    """Save the formation, its skeleton and skin, and its poses as assets in an asset library: the formation as a collection, each pose as a pose asset"""
    bl_idname = "pe.save_asset"
    bl_label = "Save to Asset Library"
    bl_options = {"REGISTER"}

    library: EnumProperty(name="Library", items=_libraries)
    name: StringProperty(name="Name", default="", options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return formation_of(context) is not None

    def invoke(self, context, event):
        form = formation_of(context)
        coll = form.get(ASSET_PROP)
        self.name = coll.name if isinstance(coll, bpy.types.Collection) else form.name
        lib = context.window_manager.pe_asset_lib
        if lib:
            self.library = lib
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        form = formation_of(context)
        lib = _library(self.library)
        if lib is None or not lib.path:
            self.report({"ERROR"}, "No asset library: add one in Preferences > File Paths > Asset Libraries")
            return {"CANCELLED"}
        try:
            path, coll, poses = save_asset(form, bpy.path.abspath(lib.path), self.name)
        except (OSError, RuntimeError) as e:
            self.report({"ERROR"}, "Could not save: %s" % e)
            return {"CANCELLED"}
        context.window_manager.pe_asset_lib = lib.name
        self.report({"INFO"}, "Saved %s%s to %s" % (coll.name, (" and %d pose%s" % (
            len(poses), "" if len(poses) == 1 else "s")) if poses else "", path))
        return {"FINISHED"}


# ── a formation dropped from the library ─────────────────────────────────────
def _instanced_formation(obj):
    if obj is None or obj.type != "EMPTY" or obj.instance_type != "COLLECTION" or obj.instance_collection is None:
        return None
    return next((o for o in obj.instance_collection.all_objects if is_formation(o)), None)


def make_real(context, inst):
    """Turn an instance of a formation's collection into objects of its own:
    copies of every object and its data, their links to each other kept."""
    src = inst.instance_collection
    objs = list(src.all_objects)
    mapping = {}
    for o in objs:
        n = o.copy()
        if o.data is not None:
            n.data = o.data.copy()
        mapping[o] = n
    coll = bpy.data.collections.new(src.name)
    for c in (inst.users_collection or [context.scene.collection]):
        c.children.link(coll)
        break
    mw = inst.matrix_world.copy()
    for o, n in mapping.items():
        coll.objects.link(n)
        if o.parent in mapping:
            n.parent = mapping[o.parent]
        else:
            n.matrix_world = mw @ o.matrix_world
        for m in n.modifiers:
            if getattr(m, "object", None) in mapping:
                m.object = mapping[m.object]
        for key in list(n.keys()):
            v = n[key]
            if isinstance(v, bpy.types.Object) and v in mapping:
                n[key] = mapping[v]
        for key in (ASSET_PROP, ASSET_PATH):
            if key in n.keys():
                del n[key]
    bpy.data.objects.remove(inst, do_unlink=True)
    return [mapping[o] for o in objs]


class PE_OT_make_editable(bpy.types.Operator):
    """Make this instance of a formation its own objects: the formation, its skeleton and skin, to build on and pose"""
    bl_idname = "pe.make_editable"
    bl_label = "Make Editable"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and _instanced_formation(context.active_object) is not None

    def execute(self, context):
        made = make_real(context, context.active_object)
        form = next(o for o in made if is_formation(o))
        arm = _obj(form.get(RIG_PROP))
        for o in context.selected_objects:
            o.select_set(False)
        top = arm if arm is not None else form
        top.select_set(True)
        context.view_layer.objects.active = top
        return {"FINISHED"}


# ── the panels ───────────────────────────────────────────────────────────────
class PE_PT_poses(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PE"
    bl_label = "Poses"
    bl_idname = "PE_PT_poses"
    bl_parent_id = "PE_PT_rig"

    @classmethod
    def poll(cls, context):
        # Step 3: shown while the skeleton is posed.
        arm = _skeleton(context)[1]
        return arm is not None and context.active_object == arm and arm.mode == "POSE"

    def draw(self, context):
        lay = self.layout
        form, arm, skin = _skeleton(context)
        poses = poses_of(arm)
        col = lay.column(align=True)
        for act in poses:
            row = col.row(align=True)
            row.operator("pe.pose_apply", text=act.get(POSE_NAME, act.name), icon="POSE_HLT").action = act.name
            row.operator("pe.pose_delete", text="", icon="X").action = act.name
        if not poses:
            col.label(text="No poses yet: pose it, then save the pose")
        row = lay.row(align=True)
        row.prop(context.window_manager, "pe_pose_name", text="")
        row.operator("pe.pose_save", text="Save Pose", icon="ADD")
        row = lay.row(align=True)
        row.operator("pe.pose_rest", text="Rest Pose", icon="ARMATURE_DATA")
        row.operator("pe.pose_neutral", text="Neutral Pose", icon="OUTLINER_OB_ARMATURE")


class PE_PT_asset(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PE"
    bl_label = "Asset Library"
    bl_idname = "PE_PT_asset"
    bl_order = 4

    def draw_header(self, context):
        from .panels import pe_icon
        self.layout.label(text="", icon_value=pe_icon())

    @classmethod
    def poll(cls, context):
        return formation_of(context) is not None or _instanced_formation(context.active_object) is not None

    def draw(self, context):
        lay = self.layout
        if formation_of(context) is None:
            lay.label(text="A formation from the asset library")
            lay.operator("pe.make_editable", icon="OUTLINER_OB_GROUP_INSTANCE")
            return
        form = formation_of(context)
        arm, _ = rig_objects(form)
        wm = context.window_manager
        lay.prop(wm, "pe_asset_lib", text="Library")
        n = len(poses_of(arm))
        row = lay.row()
        row.scale_y = 1.3
        row.operator("pe.save_asset", text="Save to Asset Library", icon="ASSET_MANAGER")
        what = "The formation" + (", its skeleton and %d pose%s" % (n, "" if n == 1 else "s") if arm else "")
        lay.label(text=what)
        path = form.get(ASSET_PATH)
        if path:
            lay.label(text="Saved: " + os.path.basename(path), icon="CHECKMARK")
        lay.operator("pe.add_samples", text="Add Sample Formations", icon="ASSET_MANAGER")


PROPS = {
    "pe_pose_name": StringProperty(name="Pose Name", default="",
                                   description="The name to save the pose under (Pose N when empty)"),
    "pe_asset_lib": EnumProperty(name="Asset Library", items=_libraries,
                                 description="The asset library to save into (Preferences > File Paths)"),
}

CLASSES = (PE_OT_pose_save, PE_OT_pose_apply, PE_OT_pose_rest, PE_OT_pose_delete, PE_OT_save_asset,
           PE_OT_add_samples,
           PE_OT_make_editable, PE_PT_poses, PE_PT_asset)
