from collections import deque
from functools import reduce
import itertools

import matplotlib.pyplot as plt

from ..common import VISUAL_EPS, MAX_LEVEL, tags_to_word, word_to_tags


def plot_limit_set(gens, beg_prefix='a', end_prefix='b', as_curve=True, ax=None, max_level=MAX_LEVEL, eps=VISUAL_EPS, debug=False, **kwargs):
    """
    Plot limit set of a generating set of Mobius transformations.
    The kwargs are passed on to matplotlib.

    :param gens: list of generating Mobius transformations
    :param beg_prefix: prefix to start at, as a string (default a)
    :param end_prefix: prefix to end at, as a string (default b)
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

    pts = list(dfs(gens, beg_prefix=beg_prefix, end_prefix=end_prefix, max_level=max_level, eps=eps, debug=debug))
    xs = [x.real for x in pts]
    ys = [x.imag for x in pts]
    if as_curve:
        # connect last to first points, if we're plotting the whole curve
        if beg_prefix == 'a' and end_prefix == 'b':
            xs.append(xs[0])
            ys.append(ys[0])
        ax.plot(xs, ys, **kwargs)
    else:
        ax.scatter(xs, ys, marker='.', s=10, **kwargs)

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


def dfs(gens, beg_prefix='a', end_prefix='b', max_level=MAX_LEVEL, eps=VISUAL_EPS, debug=False):
    """
    Non-recursive DFS for plotting limit set (only for 4 generators).

    :param gens: list of generating Mobius transformations
    :param beg_prefix: prefix to start at, as a string (default a)
    :param end_prefix: prefix to end at, as a string (default b)
    :param max_level: max level to plot
    :param eps: tolerance for termination
    :param debug: debug prints
    :return: complex points to plot
    """
    beg_tags = word_to_tags(beg_prefix)
    end_tags = word_to_tags(end_prefix)
    if not precedes_or_equal(beg_tags, end_tags):
        raise ValueError("beginning prefix must precede end prefix in tree ordering")

    # start with the first word that starts with beg_prefix
    tags = deque([beg_tags[0]])
    words = deque([gens[beg_tags[0]]])
    if len(beg_tags) > 1:
        for t in beg_tags[1:]:
            tags.append(t)
            words.append(words[-1](gens[t]))

    if debug:
        print(tags_to_word(tags))
    level = len(tags)
    old_pt, fps = get_commutator_fps(gens)

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
            print(tags_to_word(tags))

        # stop if we're at the last word starting with end_prefix
        if starts_with(tags, end_tags) and all_lefts_from(tags, len(end_tags) - 1):
            break

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
                break
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


def right_of(tag):
    return (tag + 1) % 4


def left_of(tag):
    return (tag - 1) % 4


def inverse_of(tag):
    return (tag + 2) % 4


def starts_with(tags, prefix_tags):
    """Check whether tags starts with a given prefix"""
    return list(itertools.islice(tags, 0, len(prefix_tags))) == prefix_tags


def all_lefts_from(tags, idx):
    """Check whether tags is all left turns starting from idx"""
    for i in range(idx, len(tags)-1):
        if left_of(tags[i]) != tags[i+1]:
            return False
    return True


def precedes_or_equal(tags_1, tags_2):
    """Check whether tags_1 precedes tags_2 in the tree ordering (or is equal)"""
    # ordering is: a, B, A, b <=> 0, 3, 2, 1
    # to make things easier, we replace 0 with 4
    first = [x if x != 0 else 4 for x in tags_1]
    second = [x if x != 0 else 4 for x in tags_2]
    return recursive_precedes_or_equal(first, second)


def recursive_precedes_or_equal(first, second):
    if first == second or len(first) == 0:
        return True
    if len(second) == 0:
        return False
    return (
        first[0] > second[0]
        or (
            first[0] == second[0]
            and recursive_precedes_or_equal(first[1:], second[1:])
        )
    )
