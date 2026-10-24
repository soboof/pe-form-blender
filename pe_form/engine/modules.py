"""The built-in module library, read from modules.json.

modules.json is written from the web engine itself (scripts/build-data.mjs
exports pe-web/src/engine/modules.ts): the same 57 solids, the vertices
already centred as buildModData centres them, the same face order, so a face
index means the same face in both. Never edit it by hand.
"""

import json
import os

from .vec import Vec3


class ModuleData:
    __slots__ = ("verts", "faces", "poly_faces", "base_edge")

    def __init__(self, verts, faces, poly_faces, base_edge):
        self.verts = verts
        self.faces = faces
        self.poly_faces = poly_faces
        self.base_edge = base_edge


class ModuleMeta:
    __slots__ = ("id", "name", "sub", "color", "premium", "category", "math_category", "short_code")

    def __init__(self, d):
        self.id = d["id"]
        self.name = d["name"]
        self.sub = d["sub"]
        self.color = d["color"]
        self.premium = d["premium"]
        self.category = d["category"]
        self.math_category = d["mathCategory"]
        self.short_code = d.get("shortCode")


def _load():
    path = os.path.join(os.path.dirname(__file__), "modules.json")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    mods = [ModuleMeta(m) for m in raw["modules"]]
    data = {}
    for mid, md in raw["modData"].items():
        data[mid] = ModuleData(
            [Vec3(v[0], v[1], v[2]) for v in md["verts"]],
            [list(f) for f in md["faces"]],
            [list(p) for p in md["polyFaces"]],
            md["baseEdge"],
        )
    sets = {k: frozenset(v) for k, v in raw["sets"].items()}
    return mods, data, sets


MODULES, MOD_DATA, SETS = _load()
MODULE_BY_ID = {m.id: m for m in MODULES}

# The module's own polygon side counts (POLY_FACE_TYPES): what the badges say.
POLY_FACE_TYPES = {mid: sorted({len(p) for p in md.poly_faces}, key=lambda n: [len(p) for p in md.poly_faces].index(n))
                   for mid, md in MOD_DATA.items()}


def color_of(mid, fallback="#c44444"):
    m = MODULE_BY_ID.get(mid)
    return m.color if m else fallback
