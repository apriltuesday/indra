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
        pass  # TODO filling is tricky

    def __eq__(self, other):
        return np.isclose(self.center, other.center) and np.isclose(self.radius, other.radius)

    def __repr__(self):
        return f'Circle(center={self.center}, radius={self.radius})'


class Line:

    # lines as cos(a) * x + sin(a) * y = b
    # for direction a in [0, 2pi) and offset b >= 0
    def __init__(self, direction, offset, inside_pt=None):
        if not np.isreal(direction) or direction < 0 or direction >= 2*np.pi:
            raise ValueError('Direction must be in [0, 2pi)')
        self.direction = direction
        if not np.isreal(offset) or offset < 0:
            raise ValueError('Offset must be >= 0')
        self.offset = offset

        # point that is "inside" the line – defaults to (0, y+c) or (x+c, 0) depending on orientation
        self.inside_pt = inside_pt
        if inside_pt is None:
            if not self.is_vertical():
                self.inside_pt = complex(0, self.offset / np.sin(self.direction) + 1)
            else:
                self.inside_pt = complex(self.offset + 1, 0)

    def is_horizontal(self):
        return np.isclose(self.direction, np.pi / 2) or np.isclose(self.direction, -np.pi / 2)

    def is_vertical(self):
        return np.isclose(self.direction, 0) or np.isclose(self.direction, np.pi)

    def plot(self, ax, color='k'):
        pass  # TODO filling is tricky

    def __eq__(self, other):
        return np.isclose(self.direction, other.direction) and np.isclose(self.offset, other.offset)

    def __repr__(self):
        return f'Line(direction={self.direction}, offset={self.offset})'
