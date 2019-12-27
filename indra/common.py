from numbers import Number

import matplotlib.pyplot as plt
import numpy as np

NUMERIC_EPS = 1e-12
VISUAL_EPS = 1e-4


class Circle:

    def __init__(self, center, radius, inside_pt=None):
        self.center = complex(center)
        if not np.isreal(radius) or radius <= 0:
            raise ValueError('Radius must be positive real')
        self.radius = radius

        # point that is "inside" the circle – defaults to center
        self.inside_pt = inside_pt
        if inside_pt is None:
            self.inside_pt = self.center

    def plot(self, ax, color='k'):
        circ = plt.Circle((self.center.real, self.center.imag), self.radius, color=color)
        ax.add_artist(circ)
        # TODO: how to fill the outside

    def __eq__(self, other):
        return np.isclose(self.center, other.center) and np.isclose(self.radius, other.radius)

    def __repr__(self):
        return f'Circle(center={self.center}, radius={self.radius})'


class Line:

    # lines as cos(a) * x + sin(a) * y = b,
    # for direction a in [0, 2pi) and offset b >= 0
    def __init__(self, direction, offset, inside_pt=None):
        if not np.isreal(direction) or direction < 0 or direction >= 2*np.pi:
            raise ValueError('Direction must be in [0, 2pi)')
        self.direction = direction
        if not np.isreal(offset) or offset < 0:
            raise ValueError('Offset must be >= 0')
        self.offset = offset

        self.x_coef = np.cos(self.direction)
        self.y_coef = np.sin(self.direction)

        # point that is "inside" the line
        # defaults to (0, y+c) or (x+c, 0) depending on orientation
        self.inside_pt = inside_pt
        if inside_pt is None:
            if not self.is_vertical():
                self.inside_pt = self(0) + 1j
            else:
                self.inside_pt = complex(self.offset + 1, 0)

        # for generality this is useful to have
        self.radius = np.inf

    def is_horizontal(self):
        return np.isclose(self.x_coef, 0)

    def is_vertical(self):
        return np.isclose(self.y_coef, 0)

    def plot(self, ax, color='k'):
        if not self.is_vertical():
            xs = ax.get_xlim()
            ys = [self(x) for x in xs]
            max_y = ax.get_ylim()[1]
            ax.fill_between(xs, [max_y, max_y], ys, color=color)
        else:
            ys = ax.get_ylim()
            xs = [self(y) for y in ys]
            max_x = ax.get_xlim()[1]
            ax.fill_betweenx(ys, [max_x, max_x], xs, color=color)
        # TODO: color the appropriate side

    def __call__(self, other):
        if isinstance(other, Number):
            other = complex(other)
            if not self.is_vertical():
                # we assume this is x and must compute y
                return (self.offset - self.x_coef * other) / self.y_coef
            # otherwise vertical, so assume this is y and return x
            return self.offset
        raise NotImplementedError('Transformation not supported')

    def __eq__(self, other):
        return np.isclose(self.direction, other.direction) and np.isclose(self.offset, other.offset)

    def __repr__(self):
        return f'Line(direction={self.direction}, offset={self.offset})'
