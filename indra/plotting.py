from functools import reduce

import matplotlib.pyplot as plt

from .common import VISUAL_EPS


def plot_circle(C, ax, color='k'):
    circ = plt.Circle((C.center.real, C.center.imag), C.radius, color=color)
    ax.add_artist(circ)


def plot_tiles(gens, circs, ax=None, plot_level=None, eps=VISUAL_EPS):
    """
    Plot tiles generated by set of Mobius transformations.

    :param gens: list of generating Mobius transformations
    :param circs: seed circles to start with
    :param ax: optional axis for plotting
    :param plot_level: plot only circles of this level, or None to plot all
    :param eps: minimum radius size to return
    :return: the axis used
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        ax.set_xlim((
            max(C.center.real + C.radius for C in circs),
            min(C.center.real - C.radius for C in circs)
        ))
        ax.set_ylim((
            max(C.center.imag + C.radius for C in circs),
            min(C.center.imag - C.radius for C in circs)
        ))

    colors = plt.cm.get_cmap('viridis', 20)

    # sort by level so that they plot in the correct order
    tiles = sorted(dfs_tiles(gens, circs, eps=eps), key=lambda x: x[1])
    if plot_level is not None:
        tiles = [x for x in tiles if x[1] == plot_level]
    for C, level in tiles:
        plot_circle(C, ax, color=colors(level))

    return ax


def dfs_tiles(gens, circs, eps):
    """
    Iterate through tiles with depth-first search.

    :param gens: list of generating Mobius transformations
    :param circs: seed circles to start with
    :param eps: minimum radius size to return
    :return: circle and corresponding level
    """
    for k in range(len(gens)):
        yield circs[k], 0
        yield from explore_tree_tiles(gens[k], k, circs[k], 1, gens, eps)


def explore_tree_tiles(X, l, C, level, gens, eps):
    n = len(gens)
    for k in range(l - 1, l + 2):
        Y = X(gens[k % n])
        new_circ = Y(C)
        yield new_circ, level
        if new_circ.radius > eps:
            yield from explore_tree_tiles(Y, k, C, level + 1, gens, eps)


def _get_generator_fps(gens):
    return [T.sink() for T in gens]


def _get_commutator_fps(gens):
    n = len(gens)
    # will be useful later, for now just used for starting point
    # beg_pts = [
    #     reduce(lambda S, T: S(T), (gens[(i + j) % n] for j in range(1, n + 1)))
    #     for i in range(n)
    # ]
    # beg_pts = [T.sink() for T in beg_pts]

    end_pts = [
        reduce(lambda S, T: S(T), (gens[(i - j) % n] for j in range(1, n + 1)))
        for i in range(n)
    ]
    end_pts = [T.sink() for T in end_pts]
    return end_pts


def plot_limit_points(gens, circs, ax=None, eps=VISUAL_EPS):
    """
    Plot limit points of generating set of Mobius transformations.

    :param gens: list of generating Mobius transformations
    :param circs: seed circles to start with
    :param ax: optional axis for plotting
    :param eps: minimum radius size to return
    :return: the axis used
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    pts = list(dfs_limit(gens, circs, fps_fn=_get_generator_fps, eps=eps))
    ax.scatter([x.real for x in pts], [x.imag for x in pts], marker='.', s=10)

    return ax


def plot_limit_curve(gens, circs, ax=None, eps=VISUAL_EPS):
    """
    Plot limit curve of generating set of Mobius transformations.

    :param gens: list of generating Mobius transformations
    :param circs: seed circles to start with
    :param ax: optional axis for plotting
    :param eps: minimum radius size to return
    :return: the axis used
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    pts = list(dfs_limit(gens, circs, fps_fn=_get_commutator_fps, eps=eps))
    ax.plot([x.real for x in pts] + [pts[0].real], [x.imag for x in pts] + [pts[0].imag])

    return ax


def dfs_limit(gens, circs, fps_fn, eps):
    """
    Iterate through limit points with depth-first search.

    :param gens: list of generating Mobius transformations
    :param circs: seed circles to start with
    :param fps_fn: function to get fixed points
    :param eps: minimum radius size to return
    :return: point (complex number)
    """
    fps = fps_fn(gens)
    for k in range(len(gens)):
        yield from explore_tree_limit(gens[k], k, circs[k], gens, fps, eps)


def explore_tree_limit(X, l, C, gens, fps, eps):
    n = len(gens)
    for k in range(l - 1, l + 2):
        Y = X(gens[k % n])
        if Y(C).radius < eps:  # TODO fix this termination condition
            yield Y(fps[k % n])
        else:
            yield from explore_tree_limit(Y, k, C, gens, fps, eps)
