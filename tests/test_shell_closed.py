"""The formation's shell stays closed: no edge of the mesh is in one polygon.

Run from the repository root:  python3 -m unittest discover tests
"""

import collections
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pe_form.engine.builder import AllFace  # noqa: E402
from pe_form.engine.closure import faces_to_close  # noqa: E402
from pe_form.engine.form import FormState, Settings  # noqa: E402
from pe_form.engine.meshplan import mesh_plan  # noqa: E402
from pe_form.engine.vec import Vec3  # noqa: E402

SAMPLES = ROOT / "pe_form" / "samples"

# A disphenocingulum and a decagonal prism, both placed on the first module.
# The prism's side wall meets one of the disphenocingulum's triangles with the
# triangle's corner 0.5 inside the prism (the SAT's 6% shrink lets that
# through), the interior purge reads the triangle as buried, and the three
# edges it closed were left in one polygon each.
PRISM_ON_CINGULUM = {
    "v": 1, "X": 5, "color": "#c44444", "seedShape": 3,
    "steps": [
        {"modId": "tridiminishedrhombicosidodecahedron", "usedIdx": 0, "hexaSub": "cap", "orientation": 0,
         "color": "#64ffda", "connSel": {"face:3": 0}, "stepX": 5},
        {"modId": "disphenocingulum", "usedIdx": 22, "hexaSub": "cap", "orientation": 0,
         "color": "#5c6bc0", "connSel": {}, "stepX": 5},
        {"modId": "decagonalprism", "usedIdx": 2, "hexaSub": "cap", "orientation": 0,
         "color": "#ab47bc", "connSel": {}, "stepX": 5},
    ],
}


def load(doc):
    st = FormState(Settings(None))
    placed, failed = st.load_doc(doc)
    return st, placed, failed


def edge_uses(engine):
    """(edges in one polygon, edges in three or more) of the mesh."""
    _verts, polys, _cols, _opens, _hexes = mesh_plan(engine)
    uses = collections.Counter()
    for p in polys:
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]
            uses[(min(a, b), max(a, b))] += 1
    return sum(1 for n in uses.values() if n == 1), sum(1 for n in uses.values() if n > 2)


class ShellClosed(unittest.TestCase):
    def test_wall_buried_by_overlap_stays(self):
        st, placed, failed = load(PRISM_ON_CINGULUM)
        self.assertEqual((placed, failed), (3, None))
        self.assertEqual(edge_uses(st.engine), (0, 0))

    def test_b22_is_closed(self):
        # Two near-coincident triangles of different modules were cancelled
        # as one wall; they differ at one corner by 0.043 of an edge.
        st, placed, failed = load(json.loads((SAMPLES / "B22.json").read_text()))
        self.assertEqual((placed, failed), (36, None))
        self.assertEqual(edge_uses(st.engine), (0, 0))

    def test_samples_load_whole_and_closed(self):
        for path in sorted(SAMPLES.glob("*.json")):
            doc = json.loads(path.read_text())
            with self.subTest(sample=path.name):
                st, placed, failed = load(doc)
                self.assertIsNone(failed)
                self.assertEqual(placed, len(doc["steps"]))
                self.assertEqual(edge_uses(st.engine), (0, 0))

    def test_sample_mesh_sizes_are_kept(self):
        st, _placed, _failed = load(json.loads((SAMPLES / "Littel dragon.json").read_text()))
        verts, polys, _cols, _opens, _hexes = mesh_plan(st.engine)
        self.assertEqual((len(verts), len(polys)), (219, 367))


def tri(a, b, c):
    return AllFace(Vec3(*a), Vec3(*b), Vec3(*c), "k", "#fff")


class FacesToClose(unittest.TestCase):
    P = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    TETRA = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]

    def faces(self):
        return [tri(*(self.P[i] for i in t)) for t in self.TETRA]

    def test_nothing_taken_out(self):
        self.assertEqual(faces_to_close(self.faces(), [], 1e-6), [])

    def test_the_only_face_on_three_edges_comes_back(self):
        f = self.faces()
        self.assertEqual(faces_to_close(f[1:], [f[0]], 1e-6), [f[0]])

    def test_a_corner_a_hair_off_is_the_same_corner(self):
        f = self.faces()
        g = tri((0, 0, 1e-9), (0, 1, 0), (1, 0, 0))
        self.assertEqual(faces_to_close(f[1:], [g], 1e-6), [g])

    def test_a_cancelled_wall_stays_out(self):
        # Two tetrahedra glued on a face, the wall taken out of both: each
        # edge round it is in two faces still, so nothing is put back.
        a, b, c, d, e = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)
        sides = [tri(a, b, d), tri(b, c, d), tri(c, a, d), tri(a, b, e), tri(b, c, e), tri(c, a, e)]
        wall = [tri(a, b, c), tri(a, c, b)]
        self.assertEqual(faces_to_close(sides, wall, 1e-6), [])

    def test_two_walls_that_are_not_twins_come_back_together(self):
        # B22: two triangles on one edge p-q, one corner of one a tenth of an
        # edge off the other's. The cancellation took out both; each of the
        # four edges round them has a neighbour on it and no face on the other
        # side. Neither alone closes them all, the pair does.
        p, q, c1, c2 = (0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0.5, 1, 0.1)
        shell = [tri(p, c1, (-1, 1, 0)), tri(q, c1, (2, 1, 0)), tri(p, c2, (-1, 1, 1)), tri(q, c2, (2, 1, 1))]
        pool = [tri(p, q, c1), tri(p, q, c2)]
        self.assertEqual(faces_to_close(shell, pool, 1e-6), pool)

    def test_an_interior_wall_that_closes_nothing_stays_out(self):
        f = self.faces()
        far = [tri((9, 9, 9), (10, 9, 9), (9, 10, 9)), tri((9, 9, 9), (9, 10, 9), (10, 9, 9))]
        self.assertEqual(faces_to_close(f, far, 1e-6), [])


if __name__ == "__main__":
    unittest.main()
