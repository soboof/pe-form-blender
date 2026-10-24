"""Port of pe-web/src/engine/snap.ts: snapTransform and pickConnFace."""

import math

from . import jsmath

from .vec import Vec3, Quat
from .geom import cen_n, nor_n
from .modules import MOD_DATA, SETS


class SnapResult:
    __slots__ = ("sv", "Q", "tr")

    def __init__(self, sv, Q, tr):
        self.sv = sv
        self.Q = Q
        self.tr = tr


def snap_transform(md, tf, ori_deg, sc, cfi):
    """Rigid transform mating module face `cfi` onto target face `tf`.

    q1 aligns the connection normal with -target normal, q2 twists the
    reference edges into line, qO is the user's turn; Q = qO*q2*q1 and
    tr = target centroid - Q(connection centroid).
    """
    sv = [v.clone().multiply_scalar(sc) for v in md.verts]
    cf = md.poly_faces[cfi]
    poly = [sv[i] for i in cf]
    conn_norm = nor_n(poly)
    conn_cen = cen_n(poly)
    conn_ref = poly[0].clone().sub(conn_cen).normalize()
    tN = tf.normal3.clone()
    tCen = tf.center3.clone()
    tRef = tf.verts3[0].clone().sub(tCen).normalize()
    q1 = Quat().set_from_unit_vectors(conn_norm.clone(), tN.clone().negate())
    cRR = conn_ref.clone().apply_quaternion(q1)
    pC = cRR.clone().sub(tN.clone().multiply_scalar(cRR.dot(tN)))
    pT = tRef.clone().sub(tN.clone().multiply_scalar(tRef.dot(tN)))
    ang = 0
    if pC.length_sq() > 0.0001 and pT.length_sq() > 0.0001:
        pC.normalize()
        pT.normalize()
        ang = jsmath.atan2(Vec3().cross_vectors(pC, pT).dot(tN), pC.dot(pT))
    q2 = Quat().set_from_axis_angle(tN, ang)
    qO = Quat().set_from_axis_angle(tN, (ori_deg * math.pi) / 180)
    Q = qO.clone().multiply(q2).multiply(q1)
    tr = tCen.clone().sub(conn_cen.clone().apply_quaternion(Q))
    return SnapResult(sv, Q, tr)


def _S(name):
    return SETS[name]


def _best(md, pred, score, init, better, fallback):
    best, bs = -1, init
    for fi in range(len(md.poly_faces)):
        if not pred(fi):
            continue
        s = score(fi)
        if better(s, bs):
            bs, best = s, fi
    return fallback if best == -1 else best


def pick_conn_face(mod_id, hs, target_face_type=3):
    """Which polyFace of a built-in module connects, by its cap/side dial.

    Returns a polyFaces index, or -1 when no face of the target's corner
    count exists.
    """
    md = MOD_DATA.get(mod_id)
    if md is None:
        return -1

    def pn(pfi):
        return nor_n([md.verts[i] for i in md.poly_faces[pfi]])

    T = target_face_type
    gt = lambda s, b: s > b  # noqa: E731
    lt = lambda s, b: s < b  # noqa: E731
    NEG, POS = -math.inf, math.inf

    if mod_id in ("rhombicprism", "goldenrhombicprism") and T == 4:
        st = _S("RPRISM_SIDE") if hs == "side" else _S("RPRISM_CAP")
        return _best(md, lambda fi: fi in st, lambda fi: pn(fi).y, POS, lt, 0)

    def capside(cap_sets, side_sets, fallback, cap2_sets=None, side_score_cap2=True):
        best, bs = -1, NEG
        for fi in range(len(md.poly_faces)):
            n = pn(fi)
            if hs == "cap" and any(fi in s for s in cap_sets) and abs(n.y) > bs:
                bs, best = abs(n.y), fi
            elif cap2_sets is not None and hs == "cap2" and any(fi in s for s in cap2_sets) and \
                    (1 - abs(n.y) if side_score_cap2 else abs(n.y)) > bs:
                bs, best = (1 - abs(n.y) if side_score_cap2 else abs(n.y)), fi
            elif hs == "side" and any(fi in s for s in side_sets) and 1 - abs(n.y) > bs:
                bs, best = 1 - abs(n.y), fi
        return fallback if best == -1 else best

    if mod_id == "anticupola" and T == 3:
        return capside([_S("HEXA_CAP_BOT"), _S("HEXA_CAP_TOP")], [_S("HEXA_SIDE")], 0)
    if mod_id == "gyroelongdipyr" and T == 3:
        return capside([_S("GYRO_CAP")], [_S("GYRO_SIDE")], 0)
    if mod_id == "triaugprism" and T == 3:
        return capside([_S("TRIAUGS_CAP")], [_S("TRIAUGS_SIDE")], 0, cap2_sets=[_S("TRIAUGS_CAP2")], side_score_cap2=True)
    if mod_id == "tridiminishedicosahedron" and T == 3:
        return capside([_S("TRIDIM_CAP_TOP")], [_S("TRIDIM_SIDE")], 0, cap2_sets=[_S("TRIDIM_CAP_BOT")], side_score_cap2=False)
    if mod_id == "triangularcupola" and T == 3 and hs in ("cap", "side"):
        return capside([_S("TRICUPOLA_CAP")], [_S("TRICUPOLA_SIDE")], 4)
    if mod_id == "squarecupola" and T == 4:
        return capside([_S("SQCUPOLA_CAP_SQ")], [_S("SQCUPOLA_SIDE_SQ")], 1)
    if mod_id == "pentagonalrotunda" and T == 5:
        return capside([_S("ROTUNDA_PENTA_CAP")], [_S("ROTUNDA_PENTA_SIDE")], 6)
    if mod_id == "pentagonalrotunda" and T == 3:
        return capside([_S("ROTUNDA_TRI_CAP")], [_S("ROTUNDA_TRI_SIDE")], 1)
    if mod_id == "snubdisphenoid" and T == 3:
        want_side = hs == "side"
        st = _S("SNUB_SIDE") if want_side else _S("SNUB_CAP")
        return _best(md, lambda fi: fi in st,
                     lambda fi: (1 - abs(pn(fi).y)) if want_side else abs(pn(fi).y), NEG, gt, 0)
    if mod_id == "snubdodecahedron" and T == 3:
        st = _S("SNUBDODECA_VERTEX") if hs == "vertex" else _S("SNUBDODECA_EDGE")
        return _best(md, lambda fi: fi in st, lambda fi: pn(fi).y, POS, lt, -1)
    if mod_id == "snubcube" and T == 3:
        st = _S("SNUBCUBE_VERTEX") if hs == "vertex" else _S("SNUBCUBE_EDGE")
        return _best(md, lambda fi: fi in st, lambda fi: pn(fi).y, POS, lt, -1)

    # Default: of the polyFaces with the target's corner count, the one whose
    # normal points lowest.
    return _best(md, lambda fi: len(md.poly_faces[fi]) == T, lambda fi: pn(fi).y, POS, lt, -1)
