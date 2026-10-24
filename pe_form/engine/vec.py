"""The three.js r128 math the web engine uses, operation for operation.

The Form's geometry keys faces by their centroids printed to 4 decimals and
cancels walls on sums of coordinates, so the port has to round exactly where
the web rounds: every method below is the three.js source order of operations
(normalize multiplies by 1/len, applyQuaternion is quat * v * conj, and so on).
Python floats are the same IEEE doubles as JS numbers, so the same order gives
the same bits. Pure Python, no bpy: the engine runs and is tested outside
Blender.
"""

import math

from . import jsmath

EPSILON = 2.220446049250313e-16  # Number.EPSILON


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return "Vec3(%r, %r, %r)" % (self.x, self.y, self.z)

    def clone(self):
        return Vec3(self.x, self.y, self.z)

    def copy(self, v):
        self.x, self.y, self.z = v.x, v.y, v.z
        return self

    def set(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        return self

    def add(self, v):
        self.x += v.x
        self.y += v.y
        self.z += v.z
        return self

    def sub(self, v):
        self.x -= v.x
        self.y -= v.y
        self.z -= v.z
        return self

    def add_vectors(self, a, b):
        self.x = a.x + b.x
        self.y = a.y + b.y
        self.z = a.z + b.z
        return self

    def sub_vectors(self, a, b):
        self.x = a.x - b.x
        self.y = a.y - b.y
        self.z = a.z - b.z
        return self

    def add_scaled_vector(self, v, s):
        self.x += v.x * s
        self.y += v.y * s
        self.z += v.z * s
        return self

    def multiply_scalar(self, s):
        self.x *= s
        self.y *= s
        self.z *= s
        return self

    def divide_scalar(self, s):
        return self.multiply_scalar(1 / s)

    def negate(self):
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return self

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def length_sq(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self):
        return self.divide_scalar(self.length() or 1)

    def cross_vectors(self, a, b):
        ax, ay, az = a.x, a.y, a.z
        bx, by, bz = b.x, b.y, b.z
        self.x = ay * bz - az * by
        self.y = az * bx - ax * bz
        self.z = ax * by - ay * bx
        return self

    def distance_to_squared(self, v):
        dx = self.x - v.x
        dy = self.y - v.y
        dz = self.z - v.z
        return dx * dx + dy * dy + dz * dz

    def distance_to(self, v):
        return math.sqrt(self.distance_to_squared(v))

    def lerp(self, v, alpha):
        self.x += (v.x - self.x) * alpha
        self.y += (v.y - self.y) * alpha
        self.z += (v.z - self.z) * alpha
        return self

    def apply_quaternion(self, q):
        x, y, z = self.x, self.y, self.z
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        ix = qw * x + qy * z - qz * y
        iy = qw * y + qz * x - qx * z
        iz = qw * z + qx * y - qy * x
        iw = -qx * x - qy * y - qz * z
        self.x = ix * qw + iw * -qx + iy * -qz - iz * -qy
        self.y = iy * qw + iw * -qy + iz * -qx - ix * -qz
        self.z = iz * qw + iw * -qz + ix * -qy - iy * -qx
        return self

    def tuple(self):
        return (self.x, self.y, self.z)


def sub(a, b):
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z)


def cross(a, b):
    return Vec3().cross_vectors(a, b)


class Quat:
    __slots__ = ("x", "y", "z", "w")

    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def clone(self):
        return Quat(self.x, self.y, self.z, self.w)

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

    def normalize(self):
        l = self.length()
        if l == 0:
            self.x = self.y = self.z = 0.0
            self.w = 1.0
        else:
            l = 1 / l
            self.x = self.x * l
            self.y = self.y * l
            self.z = self.z * l
            self.w = self.w * l
        return self

    def set_from_axis_angle(self, axis, angle):
        half = angle / 2
        s = jsmath.sin(half)
        self.x = axis.x * s
        self.y = axis.y * s
        self.z = axis.z * s
        self.w = jsmath.cos(half)
        return self

    def set_from_unit_vectors(self, v_from, v_to):
        r = v_from.dot(v_to) + 1
        if r < EPSILON:
            r = 0
            if abs(v_from.x) > abs(v_from.z):
                self.x = -v_from.y
                self.y = v_from.x
                self.z = 0
                self.w = r
            else:
                self.x = 0
                self.y = -v_from.z
                self.z = v_from.y
                self.w = r
        else:
            self.x = v_from.y * v_to.z - v_from.z * v_to.y
            self.y = v_from.z * v_to.x - v_from.x * v_to.z
            self.z = v_from.x * v_to.y - v_from.y * v_to.x
            self.w = r
        return self.normalize()

    def multiply_quaternions(self, a, b):
        qax, qay, qaz, qaw = a.x, a.y, a.z, a.w
        qbx, qby, qbz, qbw = b.x, b.y, b.z, b.w
        self.x = qax * qbw + qaw * qbx + qay * qbz - qaz * qby
        self.y = qay * qbw + qaw * qby + qaz * qbx - qax * qbz
        self.z = qaz * qbw + qaw * qbz + qax * qby - qay * qbx
        self.w = qaw * qbw - qax * qbx - qay * qby - qaz * qbz
        return self

    def multiply(self, q):
        return self.multiply_quaternions(self, q)


INF = float("inf")


class Box3:
    __slots__ = ("min", "max")

    def __init__(self, mn=None, mx=None):
        self.min = mn if mn is not None else Vec3(INF, INF, INF)
        self.max = mx if mx is not None else Vec3(-INF, -INF, -INF)

    def clone(self):
        return Box3(self.min.clone(), self.max.clone())

    def make_empty(self):
        self.min.set(INF, INF, INF)
        self.max.set(-INF, -INF, -INF)
        return self

    def expand_by_point(self, p):
        mn, mx = self.min, self.max
        mn.x = min(mn.x, p.x)
        mn.y = min(mn.y, p.y)
        mn.z = min(mn.z, p.z)
        mx.x = max(mx.x, p.x)
        mx.y = max(mx.y, p.y)
        mx.z = max(mx.z, p.z)
        return self

    def set_from_points(self, pts):
        self.make_empty()
        for p in pts:
            self.expand_by_point(p)
        return self

    def expand_by_scalar(self, s):
        self.min.x -= s
        self.min.y -= s
        self.min.z -= s
        self.max.x += s
        self.max.y += s
        self.max.z += s
        return self

    def contains_point(self, p):
        mn, mx = self.min, self.max
        return not (p.x < mn.x or p.x > mx.x or p.y < mn.y or p.y > mx.y or p.z < mn.z or p.z > mx.z)

    def intersects_box(self, b):
        mn, mx = self.min, self.max
        return not (b.max.x < mn.x or b.min.x > mx.x or b.max.y < mn.y or b.min.y > mx.y
                    or b.max.z < mn.z or b.min.z > mx.z)

    def is_empty(self):
        return self.max.x < self.min.x or self.max.y < self.min.y or self.max.z < self.min.z


def box_from_points(pts):
    return Box3().set_from_points(pts)
