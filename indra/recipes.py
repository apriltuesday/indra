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


def circles_paired_by_transform(T):
    """Return circles paired by mobius transformation T (or ??? if not circle-pairing)"""
    pass
