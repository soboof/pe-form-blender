"""JavaScript number formatting, where the web engine's keys depend on it.

Face keys are centroids printed with Number.prototype.toFixed(4). Python's
"%.4f" rounds a tie to even and JS rounds it away from zero, and JS prints
"-0.0000" for a small negative but "0.0000" for -0: both matter, because a key
that prints differently is a different face. This is toFixed by the spec:
the exact decimal value of the double, rounded half away from zero.
"""

from decimal import Decimal, ROUND_HALF_UP

_Q = {}


def to_fixed(x, digits):
    if x != x:
        return "NaN"
    if x == 0:
        x = 0.0  # -0 prints as 0, as in JS
    q = _Q.get(digits)
    if q is None:
        q = _Q[digits] = Decimal(1).scaleb(-digits)
    s = format(Decimal(x).quantize(q, rounding=ROUND_HALF_UP), "f")
    return s


def js_number_str(x):
    """String(x) for the numbers the engine writes into keys and docs."""
    if x != x:
        return "NaN"
    if x == int(x) and abs(x) < 1e21:
        return str(int(x))
    r = repr(float(x))
    if "e" in r:
        mant, exp = r.split("e")
        e = int(exp)
        return "%se%s%d" % (mant, "+" if e > 0 else "-", abs(e))
    return r
