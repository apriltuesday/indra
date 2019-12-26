from numbers import Number

import numpy as np

from .common import Circle, Line


class MobiusTransformation:

    def __init__(self, a, b=None, c=None, d=None):
        # either initialize with matrix or its entries
        if b is None:
            self.M = a
        else:
            det = a * d - b * c
            if np.isclose(det, 0):
                raise ValueError('Determinant must be non-zero')
            self.M = 1 / det * np.array([[a, b], [c, d]])

    @property
    def a(self): return self.M[0, 0]

    @property
    def b(self): return self.M[0, 1]

    @property
    def c(self): return self.M[1, 0]

    @property
    def d(self): return self.M[1, 1]

    def inv(self):
        # Return inverse transformation
        return MobiusTransformation(self.d, -self.b, -self.c, self.a)

    def fps(self):
        # Return positive and negative fixed points (may be the same)
        denom = 2 * self.c
        if denom == 0:
            return np.inf
        diff = self.a - self.d
        root = np.sqrt(self.M.trace()**2 - 4)
        return (diff + root) / denom, (diff - root) / denom

    def multiplier(self):
        tr = self.M.trace()
        return ((tr + np.sqrt(tr**2 - 4)) / 2)**2

    def sink(self):
        # Return attracting fixed point (or the only one)
        pos_fp, neg_fp = self.fps()
        if pos_fp == neg_fp:
            return pos_fp
        k = self.multiplier()
        return pos_fp if abs(k) > 1 else neg_fp

    def source(self):
        # Return repelling fixed point (or the only one)
        pos_fp, neg_fp = self.fps()
        if pos_fp == neg_fp:
            return pos_fp
        k = self.multiplier()
        return neg_fp if abs(k) > 1 else pos_fp

    def conjugate(self, S):
        """Return conjugation by S, i.e. STS^-1"""
        return S(self(S.inv()))

    def __call__(self, other):
        # composition via matrix multiplication
        if isinstance(other, MobiusTransformation):
            return MobiusTransformation(self.M.dot(other.M))

        # application to points and circles
        if isinstance(other, Number):
            return self._apply_to_point(other)
        if isinstance(other, Circle):
            return self._apply_to_circle(other)
        if isinstance(other, Line):  # technically a special case of circles
            return self._apply_to_line(other)

        raise NotImplementedError('Transformation not supported')

    def _apply_to_point(self, z):
        """Apply Mobius transformation to complex number z"""
        if z == np.inf:
            return self.a / self.c if self.c != 0 else np.inf
        num = self.a * z + self.b
        den = self.c * z + self.d
        return num / den if den != 0 else np.inf

    def _apply_to_circle(self, C):
        """Apply Mobius transformation to circle C"""
        if np.isclose(abs(self.d / self.c + C.center), C.radius):
            return self._circle_to_line(C)

        z = C.center
        if self.c != 0:
            den = (self.d / self.c + C.center).conjugate()
            if den != 0:
                z -= C.radius**2 / den
        new_cen = self._apply_to_point(z)
        new_rad = abs(new_cen - self._apply_to_point(C.center + C.radius))
        D = Circle(center=new_cen, radius=new_rad)
        return D

    def _apply_to_line(self, L):
        """Apply Mobius transformation to line L"""
        # 3 points on L
        denom = np.sin(L.direction) if not L.is_horizontal() else 1
        cosine = np.cos(L.direction)
        x1, y1 = 0, L.offset / denom
        x2, y2 = -1, (L.offset + cosine) / denom
        x3, y3 = 1, (L.offset - cosine) / denom

        # 3 points on T(L)
        z1 = self(complex(x1, y1))
        z2 = self(complex(x2, y2))
        z3 = self(complex(x3, y3))

        # solve for circle
        # TODO: need to tell if this also gives a line...
        w = z3 - z1
        w /= z2 - z1
        c = (z1 - z2) * (w - abs(w) ** 2) / 2j / w.imag - z1
        rad = abs(c + z1)
        return Circle(-c, rad)

    def _circle_to_line(self, C):
        """Get application to circle C, given that we know the result is a line"""
        z1 = self(C.center + C.radius)
        z2 = self(C.center - C.radius)
        x1, y1 = z1.real, z1.imag
        x2, y2 = z2.real, z2.imag

        direction = np.arctan((x2 - x1) / (y1 - y2))
        offset = np.cos(direction) * x1 + np.sin(direction) * y1
        return Line(direction, offset)

    def __eq__(self, other):
        return self.M == other.M

    def __repr__(self):
        return f'Mobius transformation:\n{str(self.M)}'
