# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2023

from FIAT import quadrature, reference_element
import numpy

"""
@article{isaac2020recursive,
  title={Recursive, parameter-free, explicitly defined interpolation nodes for simplices},
  author={Isaac, Tobin},
  journal={SIAM Journal on Scientific Computing},
  volume={42},
  number={6},
  pages={A4046--A4062},
  year={2020},
  publisher={SIAM}
}
"""

class NodeFamily:
    """ Family of nodes on the unit interval.  This class essentially is a lazy-evaluate-and-cache dictionary: the user passes a routine to evaluate entries for unknown keys """

    def __init__(self, f):
        self._f = f
        self._cache = {}

    def __getitem__(self, key):
        try:
            return self._cache[key]
        except KeyError:
            value = self._f(key)
            self._cache[key] = value
            return value


def recursive(d, n, alpha, family):
    '''The barycentric d-simplex coordinates for a
    multiindex alpha with length n, based on a 1D node family.'''
    b = numpy.zeros((d,), dtype="d")

    xn = family[n]
    if xn is None:
        return b
    if d == 2:
        b[:] = xn[[alpha[0],alpha[1]]]
        return b
    weight = 0.0
    for i in range(d):
        alpha_noti = alpha[:i] + alpha[i+1:]
        n_noti = n - alpha[i]
        w = xn[n_noti]
        br = recursive(d-1, n_noti, alpha_noti, family)
        b[:i] += w * br[:i]
        b[i+1:] += w * br[i:]
        weight += w
    b /= weight
    return b

def recursive_points(ref_el, order, rule="gll", interior=0):
    if rule == "gll":
        lr = quadrature.GaussLobattoLegendreQuadratureLineRule
    elif rule == "gl":
        lr = quadrature.GaussLegendreQuadratureLineRule
    else:
        raise ValueError("Unsupported quadrature rule %s" % rule)

    line = reference_element.UFCInterval()
    f = lambda n: numpy.array(lr(line, n+1).pts).flatten() if n>=1 else None
    family = NodeFamily(f)

    verts = ref_el.vertices
    tdim = len(verts) - 1
    vs = numpy.array(verts)
    hs = vs[:-1, :] - vs[-1]

    get_point = lambda alpha: tuple(numpy.dot(recursive(tdim, order, alpha, family), hs) + vs[-1])
    alphas = reference_element.lattice_iter(interior, order + 1 - interior, tdim)
    return list(map(get_point, alphas))


if __name__ == "__main__":
    ref_el = reference_element.ufc_simplex(2)
    h = 0.5 * numpy.sqrt(3)
    #ref_el.vertices = [(0, h), (-1.0, -h), (1.0, -h)]

    order = 7
    rule = "gll"
    pts = recursive_points(ref_el, order, rule=rule)
    from matplotlib import pyplot as plt

    x = []
    y = []
    for p in pts:
        x.append(p[0])
        y.append(p[1])
    plt.scatter(x, y)
    plt.show()
