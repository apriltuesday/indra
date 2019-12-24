from collections import namedtuple

NUMERIC_EPS = 1e-12
VISUAL_EPS = 1e-4

# TODO: make these real classes with some checks on domains, etc.

# center is complex, radius is real
Circle = namedtuple('Circle', ['center', 'radius'])

# lines as cos(a) * x + sin(a) * y = b
# for direction a in [0, 2pi) and offset b >= 0
Line = namedtuple('Line', ['direction', 'offset'])
