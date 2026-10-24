"""Math.atan2, Math.sin and Math.cos as V8 computes them (fdlibm).

The C runtime Python uses rounds these differently from V8 in the last bit
(measured: atan2 on 18% of random inputs, sin and cos on 3%). The web engine
snaps every module through an atan2 and two sin/cos pairs, so a last bit lost
there shows up in a face key ("-0.0000" against "0.0000") and could, at a
threshold, decide a placement differently. These are the fdlibm routines V8
ships (src/base/ieee754.cc), transcribed operation for operation; Math.sqrt is
exact everywhere, so math.sqrt is used as is. Checked bit for bit against
Node in tests/test_jsmath.py.
"""

import math
import struct

_pack = struct.Struct("<d").pack
_unpack_q = struct.Struct("<q").unpack
_pack_q = struct.Struct("<q").pack
_unpack_d = struct.Struct("<d").unpack


def _words(x):
    b = _unpack_q(_pack(x))[0]
    hi = (b >> 32) & 0xFFFFFFFF
    if hi >= 0x80000000:
        hi -= 0x100000000
    return hi, b & 0xFFFFFFFF


def _hi(x):
    return _words(x)[0]


def _from_words(hi, lo):
    b = ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)
    if b >= 1 << 63:
        b -= 1 << 64
    return _unpack_d(_pack_q(b))[0]


# ── atan / atan2 ─────────────────────────────────────────────────────────────
_ATANHI = (4.63647609000806093515e-01, 7.85398163397448278999e-01,
           9.82793723247329054082e-01, 1.57079632679489655800e+00)
_ATANLO = (2.26987774529616870924e-17, 3.06161699786838301793e-17,
           1.39033110312309984516e-17, 6.12323399573676603587e-17)
_AT = (3.33333333333329318027e-01, -1.99999999998764832476e-01, 1.42857142725034663711e-01,
       -1.11111104054623557880e-01, 9.09088713343650656196e-02, -7.69187620504482999495e-02,
       6.66107313738753120669e-02, -5.83357013379057348645e-02, 4.97687799461593236017e-02,
       -3.65315727442169155270e-02, 1.62858201153657823623e-02)


def atan(x):
    hx = _hi(x)
    ix = hx & 0x7FFFFFFF
    if ix >= 0x44100000:
        if x != x:
            return x + x
        return _ATANHI[3] + _ATANLO[3] if hx > 0 else -_ATANHI[3] - _ATANLO[3]
    if ix < 0x3FDC0000:
        if ix < 0x3E400000:
            return x
        id_ = -1
    else:
        x = abs(x)
        if ix < 0x3FF30000:
            if ix < 0x3FE60000:
                id_ = 0
                x = (2.0 * x - 1.0) / (2.0 + x)
            else:
                id_ = 1
                x = (x - 1.0) / (x + 1.0)
        else:
            if ix < 0x40038000:
                id_ = 2
                x = (x - 1.5) / (1.0 + 1.5 * x)
            else:
                id_ = 3
                x = -1.0 / x
    z = x * x
    w = z * z
    s1 = z * (_AT[0] + w * (_AT[2] + w * (_AT[4] + w * (_AT[6] + w * (_AT[8] + w * _AT[10])))))
    s2 = w * (_AT[1] + w * (_AT[3] + w * (_AT[5] + w * (_AT[7] + w * _AT[9]))))
    if id_ < 0:
        return x - x * (s1 + s2)
    z = _ATANHI[id_] - ((x * (s1 + s2) - _ATANLO[id_]) - x)
    return -z if hx < 0 else z


_PI_O_2 = 1.5707963267948965580e+00
_PI = 3.1415926535897931160e+00
_PI_LO = 1.2246467991473531772e-16


def atan2(y, x):
    if x != x or y != y:
        return x + y
    hx, lx = _words(x)
    ix = hx & 0x7FFFFFFF
    hy, ly = _words(y)
    iy = hy & 0x7FFFFFFF
    if ((hx - 0x3FF00000) | lx) == 0:
        return atan(y)
    m = ((hy >> 31) & 1) | ((hx >> 30) & 2)
    if (iy | ly) == 0:
        if m in (0, 1):
            return y
        return _PI if m == 2 else -_PI
    if (ix | lx) == 0:
        return -_PI_O_2 if hy < 0 else _PI_O_2
    if ix == 0x7FF00000:
        if iy == 0x7FF00000:
            return (0.7853981633974483, -0.7853981633974483, 3.0 * 0.7853981633974483,
                    -3.0 * 0.7853981633974483)[m]
        return (0.0, -0.0, _PI, -_PI)[m]
    if iy == 0x7FF00000:
        return -_PI_O_2 if hy < 0 else _PI_O_2
    k = (iy - ix) >> 20
    if k > 60:
        z = _PI_O_2 + 0.5 * _PI_LO
        m &= 1
    elif hx < 0 and k < -60:
        z = 0.0
    else:
        z = atan(abs(y / x))
    if m == 0:
        return z
    if m == 1:
        return -z
    if m == 2:
        return _PI - (z - _PI_LO)
    return (z - _PI_LO) - _PI


# ── sin / cos ────────────────────────────────────────────────────────────────
_S1 = -1.66666666666666324348e-01
_S2 = 8.33333333332248946124e-03
_S3 = -1.98412698298579493134e-04
_S4 = 2.75573137070700676789e-06
_S5 = -2.50507602534068634195e-08
_S6 = 1.58969099521155010221e-10
_C1 = 4.16666666666666019037e-02
_C2 = -1.38888888888741095749e-03
_C3 = 2.48015872894767294178e-05
_C4 = -2.75573143513906633035e-07
_C5 = 2.08757232129817482790e-09
_C6 = -1.13596475577881948265e-11


def _k_sin(x, y, iy):
    ix = _hi(x) & 0x7FFFFFFF
    if ix < 0x3E400000:
        if int(x) == 0:
            return x
    z = x * x
    v = z * x
    r = _S2 + z * (_S3 + z * (_S4 + z * (_S5 + z * _S6)))
    if iy == 0:
        return x + v * (_S1 + z * r)
    return x - ((z * (0.5 * y - v * r) - y) - v * _S1)


def _k_cos(x, y):
    ix = _hi(x) & 0x7FFFFFFF
    if ix < 0x3E400000:
        if int(x) == 0:
            return 1.0
    z = x * x
    r = z * (_C1 + z * (_C2 + z * (_C3 + z * (_C4 + z * (_C5 + z * _C6)))))
    if ix < 0x3FD33333:
        return 1.0 - (0.5 * z - (z * r - x * y))
    if ix > 0x3FE90000:
        qx = 0.28125
    else:
        qx = _from_words(ix - 0x00200000, 0)
    iz = 0.5 * z - qx
    a = 1.0 - qx
    return a - (iz - (z * r - x * y))


_INVPIO2 = 6.36619772367581382433e-01
_PIO2_1 = 1.57079632673412561417e+00
_PIO2_1T = 6.07710050650619224932e-11
_PIO2_2 = 6.07710050630396597660e-11
_PIO2_2T = 2.02226624879595063154e-21
_PIO2_3 = 2.02226624871116645580e-21
_PIO2_3T = 8.47842766036889956997e-32
_NPIO2_HW = [_hi((k + 1) * (math.pi / 2)) for k in range(32)]


def _rem_pio2(x):
    """(n, y0, y1): x = n*pi/2 + y0 + y1, for |x| up to 2^19*pi/2."""
    hx = _hi(x)
    ix = hx & 0x7FFFFFFF
    if ix <= 0x3FE921FB:
        return 0, x, 0.0
    if ix < 0x4002D97C:
        if hx > 0:
            z = x - _PIO2_1
            if ix != 0x3FF921FB:
                y0 = z - _PIO2_1T
                return 1, y0, (z - y0) - _PIO2_1T
            z -= _PIO2_2
            y0 = z - _PIO2_2T
            return 1, y0, (z - y0) - _PIO2_2T
        z = x + _PIO2_1
        if ix != 0x3FF921FB:
            y0 = z + _PIO2_1T
            return -1, y0, (z - y0) + _PIO2_1T
        z += _PIO2_2
        y0 = z + _PIO2_2T
        return -1, y0, (z - y0) + _PIO2_2T
    if ix <= 0x413921FB:
        t = abs(x)
        n = int(t * _INVPIO2 + 0.5)
        fn = float(n)
        r = t - fn * _PIO2_1
        w = fn * _PIO2_1T
        if n < 32 and ix != _NPIO2_HW[n - 1]:
            y0 = r - w
        else:
            j = ix >> 20
            y0 = r - w
            i = j - ((_hi(y0) >> 20) & 0x7FF)
            if i > 16:
                t = r
                w = fn * _PIO2_2
                r = t - w
                w = fn * _PIO2_2T - ((t - r) - w)
                y0 = r - w
                i = j - ((_hi(y0) >> 20) & 0x7FF)
                if i > 49:
                    t = r
                    w = fn * _PIO2_3
                    r = t - w
                    w = fn * _PIO2_3T - ((t - r) - w)
                    y0 = r - w
        y1 = (r - y0) - w
        if hx < 0:
            return -n, -y0, -y1
        return n, y0, y1
    return None


def sin(x):
    ix = _hi(x) & 0x7FFFFFFF
    if ix <= 0x3FE921FB:
        return _k_sin(x, 0.0, 0)
    if ix >= 0x7FF00000:
        return x - x
    red = _rem_pio2(x)
    if red is None:
        return math.sin(x)  # beyond 2^19*pi/2: never reached by the Form
    n, y0, y1 = red
    q = n & 3
    if q == 0:
        return _k_sin(y0, y1, 1)
    if q == 1:
        return _k_cos(y0, y1)
    if q == 2:
        return -_k_sin(y0, y1, 1)
    return -_k_cos(y0, y1)


def cos(x):
    ix = _hi(x) & 0x7FFFFFFF
    if ix <= 0x3FE921FB:
        return _k_cos(x, 0.0)
    if ix >= 0x7FF00000:
        return x - x
    red = _rem_pio2(x)
    if red is None:
        return math.cos(x)
    n, y0, y1 = red
    q = n & 3
    if q == 0:
        return _k_cos(y0, y1)
    if q == 1:
        return -_k_sin(y0, y1, 1)
    if q == 2:
        return -_k_cos(y0, y1)
    return _k_sin(y0, y1, 1)
