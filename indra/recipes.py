from .common import Circle
from .mobius import MobiusTransformation as Mobius

import numpy as np


def transform_pairing_circles(C1, C2, u=complex(1, 0), v=complex(0, 0)):
    """Return mobius transformation that pairs the circles C1 and C2"""
    if abs(u)**2 - abs(v)**2 != 1:
        raise ValueError('|u|^2 - |v|^2 is not 1')

    P, r = C1.center, C1.radius
    Q, s = C2.center, C2.radius
    ubar = u.conjugate()
    vbar = v.conjugate()

    M = np.array([
        [ubar, -ubar * P + r * vbar],
        [u, -u * P + r * v]
    ])
    M = s * M + Q

    return Mobius(M)


def kissing_schottky(y, v):
    """Return generators and circles for symmetric kissing Schottky group"""
    assert np.isreal(y) and np.isreal(v)
    x = np.sqrt(1 + y ** 2)
    u = np.sqrt(1 + v ** 2)
    yv = y * v
    k = 1 / yv - np.sqrt(1 / yv ** 2 - 1)
    assert abs(k) < 1 or np.isclose(abs(k), 1)

    a = Mobius(u, 1j * k * v, -1j * v / k, u)
    b = Mobius(x, y, y, x)
    A = a.inv()
    B = b.inv()

    assert np.isclose(b(a(B(A))).M.trace(), -2)

    C_a = Circle(complex(0, k * u / v), k / v)
    C_A = Circle(complex(0, -k * u / v), k / v)
    C_b = Circle(complex(-x / y, 0), 1 / y)
    C_B = Circle(complex(x / y, 0), 1 / y)

    return [a, b, A, B], [C_a, C_b, C_A, C_B]


def parabolic_commutator(t_a, t_b, use_negative=True):
    """Grandma's special parabolic commutator groups"""
    t_a = complex(t_a)
    t_b = complex(t_b)

    # which solution of the Markov identity to use
    tatb = t_a * t_b
    if use_negative:
        t_ab = (tatb - np.sqrt(tatb ** 2 - 4 * (t_a ** 2 + t_b ** 2))) / 2
    else:
        t_ab = (tatb + np.sqrt(tatb ** 2 - 4 * (t_a ** 2 + t_b ** 2))) / 2

    # fixed point of commutator abAB
    z0 = ((t_ab - 2) * t_b) / (t_b * t_ab - 2 * t_a + 2j * t_ab)

    # generator matrices with the correct traces
    a = Mobius(
        t_a / 2,
        (t_a * t_ab - 2 * t_b + 4j) / ((2 * t_ab + 4) * z0),
        ((t_a * t_ab - 2 * t_b - 4j) * z0) / (2 * t_ab - 4),
        t_a / 2
    )
    b = Mobius(
        (t_b - 2j) / 2,
        t_b / 2,
        t_b / 2,
        (t_b + 2j) / 2
    )

    A = a.inv()
    B = b.inv()

    assert np.isclose(a.M.trace(), t_a)
    assert np.isclose(b.M.trace(), t_b)
    assert np.isclose(a(b).M.trace(), t_ab)

    return [a, b, A, B]
