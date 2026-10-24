"""Port of the parts of pe-web/src/engine/ground.ts the auto rig needs: the
face a formation was told to stand on in the web Form (a save's `ground`),
and the turn and shift that stand it there (`groundMatrix`). The Theater
rigs a formation standing, so its legs reach down to that floor.

three.js r128's own arithmetic (Matrix4.compose, setFromRotationMatrix,
applyMatrix4), so the standing points are the Theater's to the last bit.
"""

import math

from .vec import Vec3, Quat

_DOWN = Vec3(0, -1, 0)


def _ok(a):
    return isinstance(a, (list, tuple)) and len(a) == 3 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in a)


def clean_ground(raw):
    """A saved ground made sound: {"n", "p", and maybe "x"}, or None."""
    if not isinstance(raw, dict) or not _ok(raw.get("n")) or not _ok(raw.get("p")):
        return None
    n = Vec3(*raw["n"])
    if not n.length_sq() > 1e-12:
        return None
    n.normalize()
    g = {"n": (n.x, n.y, n.z), "p": tuple(raw["p"])}
    if _ok(raw.get("x")):
        x = Vec3(*raw["x"])
        x.add_scaled_vector(n, -x.dot(n))
        if x.length_sq() > 1e-12:
            x.normalize()
            g["x"] = (x.x, x.y, x.z)
    return g


def _conjugate(q):
    return Quat(-q.x, -q.y, -q.z, q.w)


def ground_turn(n):
    u = n.clone().normalize()
    if u.dot(_DOWN) < -1 + 1e-9:
        return Quat().set_from_axis_angle(Vec3(1, 0, 0), math.pi)
    return Quat().set_from_unit_vectors(u, _DOWN)


def ground_axes(g):
    Y = Vec3(*g["n"]).normalize().negate()
    X = Vec3(*g["x"]) if g.get("x") else Vec3(1, 0, 0).apply_quaternion(_conjugate(ground_turn(Vec3(*g["n"]))))
    X.add_scaled_vector(Y, -X.dot(Y)).normalize()
    Z = Vec3().cross_vectors(X, Y)
    return X, Y, Z


def _quat_from_basis(X, Y, Z):
    """Quaternion.setFromRotationMatrix(Matrix4.makeBasis(X, Y, Z))."""
    m11, m12, m13 = X.x, Y.x, Z.x
    m21, m22, m23 = X.y, Y.y, Z.y
    m31, m32, m33 = X.z, Y.z, Z.z
    trace = m11 + m22 + m33
    q = Quat()
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q.w, q.x, q.y, q.z = 0.25 / s, (m32 - m23) * s, (m13 - m31) * s, (m21 - m12) * s
    elif m11 > m22 and m11 > m33:
        s = 2.0 * math.sqrt(1.0 + m11 - m22 - m33)
        q.w, q.x, q.y, q.z = (m32 - m23) / s, 0.25 * s, (m12 + m21) / s, (m13 + m31) / s
    elif m22 > m33:
        s = 2.0 * math.sqrt(1.0 + m22 - m11 - m33)
        q.w, q.x, q.y, q.z = (m13 - m31) / s, (m12 + m21) / s, 0.25 * s, (m23 + m32) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m33 - m11 - m22)
        q.w, q.x, q.y, q.z = (m21 - m12) / s, (m13 + m31) / s, (m23 + m32) / s, 0.25 * s
    return q


def _ground_quat(g):
    if not g.get("x"):
        return ground_turn(Vec3(*g["n"]))
    return _conjugate(_quat_from_basis(*ground_axes(g)))


class Matrix4:
    """A rotation and a shift, as three's Matrix4 stores and applies them."""
    __slots__ = ("e",)

    def __init__(self, q, t):
        x, y, z, w = q.x, q.y, q.z, q.w
        x2, y2, z2 = x + x, y + y, z + z
        xx, xy, xz = x * x2, x * y2, x * z2
        yy, yz, zz = y * y2, y * z2, z * z2
        wx, wy, wz = w * x2, w * y2, w * z2
        self.e = [1 - (yy + zz), xy + wz, xz - wy, 0,
                  xy - wz, 1 - (xx + zz), yz + wx, 0,
                  xz + wy, yz - wx, 1 - (xx + yy), 0,
                  t.x, t.y, t.z, 1]

    def apply(self, v):
        """v.applyMatrix4(m), in place."""
        e = self.e
        x, y, z = v.x, v.y, v.z
        w = 1 / (e[3] * x + e[7] * y + e[11] * z + e[15])
        v.x = (e[0] * x + e[4] * y + e[8] * z + e[12]) * w
        v.y = (e[1] * x + e[5] * y + e[9] * z + e[13]) * w
        v.z = (e[2] * x + e[6] * y + e[10] * z + e[14]) * w
        return v

    def through(self, v):
        return self.apply(v.clone())

    def rows(self):
        """Row-major 4x4, for a Blender Matrix."""
        e = self.e
        return [[e[0], e[4], e[8], e[12]], [e[1], e[5], e[9], e[13]],
                [e[2], e[6], e[10], e[14]], [e[3], e[7], e[11], e[15]]]


def ground_matrix(g):
    """From the formation's own space to standing on its ground: the face's
    normal straight down, its centre at the origin. None for no ground."""
    if not g:
        return None
    q = _ground_quat(g)
    t = Vec3(*g["p"]).apply_quaternion(q).negate()
    return Matrix4(q, t)
