import numpy as np

from .common import Circle


class MobiusTransformation:

    def __init__(self, a, b=None, c=None, d=None):
        # either initialize with matrix or its entries
        if b is None:
            self.M = a
        else:
            det = a * d - b * c
            assert det != 0
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

    def __call__(self, other):
        # composition via matrix multiplication
        if isinstance(other, MobiusTransformation):
            return MobiusTransformation(self.M.dot(other.M))

        # application to points and circles
        if isinstance(other, complex):
            return self._mobius_on_point(other)
        if isinstance(other, Circle):
            return self._mobius_on_circle(other)

        raise NotImplementedError('Transformation not supported')

    def _mobius_on_point(self, z):
        """Apply Mobius transformation to complex number z"""
        if z == np.inf:
            return self.a / self.c if self.c != 0 else np.inf
        num = self.a * z + self.b
        den = self.c * z + self.d
        return num / den if den != 0 else np.inf

    def _mobius_on_circle(self, C):
        """Apply Mobius transformation to circle C"""
        # TODO: handle lines (Project 3.7)
        z = C.center
        if self.c != 0:
            den = (self.d / self.c + C.center).conjugate()
            if den != 0:
                z -= C.radius**2 / den
        new_cen = self._mobius_on_point(z)
        new_rad = abs(new_cen - self._mobius_on_point(C.center + C.radius))
        D = Circle(center=new_cen, radius=new_rad)
        return D

    def __repr__(self):
        return f'Mobius transformation:\n{str(self.M)}'
