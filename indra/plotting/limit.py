from collections import deque
from functools import reduce

import matplotlib.pyplot as plt

from ..common import VISUAL_EPS, MAX_LEVEL


def plot_limit_set(gens, as_curve=True, ax=None, max_level=MAX_LEVEL, eps=VISUAL_EPS, debug=False, **kwargs):
    """
    Plot limit set of a generating set of Mobius transformations.
    The kwargs are passed on to matplotlib.

    :param gens: list of generating Mobius transformations
    :param as_curve: whether to plot as a continuous curve (default) or individual points
    :param ax: optional axis for plotting
    :param max_level: max level to plot
    :param eps: tolerance for termination
    :param debug: debug prints
    :return: the axis used
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    pts = list(dfs(gens, max_level=max_level, eps=eps, debug=debug))
    if as_curve:
        ax.plot([x.real for x in pts] + [pts[0].real], [x.imag for x in pts] + [pts[0].imag], **kwargs)
    else:
        ax.scatter([x.real for x in pts], [x.imag for x in pts], marker='.', s=10, **kwargs)

    return ax


def get_commutator_fps(gens):
    n = len(gens)
    # will be useful later, for now just used for starting point
    beg_pts = [
        reduce(lambda S, T: S(T), (gens[(i + j) % n] for j in range(1, n + 1)))
        for i in range(n)
    ]
    beg_pts = [T.sink() for T in beg_pts]

    end_pts = [
        reduce(lambda S, T: S(T), (gens[(i - j) % n] for j in range(1, n + 1)))
        for i in range(n)
    ]
    end_pts = [T.sink() for T in end_pts]
    return beg_pts[-1], end_pts


def dfs(gens, max_level=MAX_LEVEL, eps=VISUAL_EPS, debug=False):
    """
    Non-recursive DFS for plotting limit set (only for 4 generators).

    :param gens: list of generating Mobius transformations
    :param max_level: max level to plot
    :param eps: tolerance for termination
    :param debug: debug prints
    :return: complex points to plot
    """
    tags = deque([0])
    words = deque([gens[0]])
    old_pt, fps = get_commutator_fps(gens)

    level = 0
    first_time = True  # hack to get it to cycle properly

    while True:
        # go forwards till the end of the branch
        while True:
            old_pt, branch_term = branch_termination(words[-1], fps[tags[-1]], old_pt, eps, level, max_level)
            if branch_term:
                break
            next_tag = right_of(tags[-1])
            next_word = words[-1](gens[next_tag])
            tags.append(next_tag)
            words.append(next_word)
            level += 1

        # we have a result!
        yield old_pt
        if debug:
            print(level)
            print_current_word(tags)

        # go backwards till we have another turn or reach the root
        while True:
            last_tag = tags.pop()
            _ = words.pop()
            level -= 1
            if level == 0 or available_turn(last_tag, tags[-1]):
                break

        # turn and go forwards
        next_tag = left_of(last_tag)
        if level == 0:
            # if we're back to the first generator at the root, we're done!
            if next_tag == 0:
                if not first_time:  # hack part 2
                    break
                first_time = False
            next_word = gens[next_tag]
        else:
            next_word = words[-1](gens[next_tag])
        tags.append(next_tag)
        words.append(next_word)
        level += 1


def available_turn(last_tag, curr_tag):
    """Return true if there's another turn to take from curr_tag"""
    return left_of(last_tag) != inverse_of(curr_tag)


def branch_termination(T, fp, old_pt, eps, level, max_level):
    """Return true if we should terminate branch"""
    new_pt = T(fp)
    if level > max_level or abs(new_pt - old_pt) < eps:
        return new_pt, True
    return old_pt, False


def print_current_word(tags):
    # useful for debugging
    letters = ['a', 'b', 'A', 'B']
    s = ''.join(letters[t] for t in tags)
    print(s)


def right_of(tag):
    return (tag + 1) % 4


def left_of(tag):
    return (tag - 1) % 4


def inverse_of(tag):
    return (tag + 2) % 4
