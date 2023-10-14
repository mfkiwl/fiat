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


def multiindex_equal(d, k, interior=0):
    """A generator for :math:`d`-tuple multi-indices whose sum is :math:`k`.
    """
    if d <= 0:
        return
    imin = interior
    imax = k - (d-1) * imin
    if imax < imin:
        return
    for i in range(imin, imax):
        for a in multiindex_equal(d-1, k-i, interior=imin):
            yield (i,) + a
    yield (imax,) + (imin,)*(d-1)


class NodeFamily:
    """Family of nodes on the unit interval.  This class essentially is a
    lazy-evaluate-and-cache dictionary: the user passes a routine to evaluate
    entries for unknown keys """

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


def make_node_family(family):
    line = reference_element.UFCInterval()
    if family == "equispaced":
        f = lambda n: numpy.linspace(0.0, 1.0, n + 1)
    elif family == "dg_equispaced":
        f = lambda n: numpy.linspace(1.0/(n+2.0), (n+1.0)/(n+2.0), n + 1)
    elif family == "gl":
        lr = quadrature.GaussLegendreQuadratureLineRule
        f = lambda n: numpy.array(lr(line, n + 1).pts).flatten()
    elif family == "gll":
        lr = quadrature.GaussLobattoLegendreQuadratureLineRule
        f = lambda n: numpy.array(lr(line, n + 1).pts).flatten() if n else None
    else:
        raise ValueError("Invalid node family %s" % family)
    return NodeFamily(f)


def recursive(alpha, family):
    """The barycentric d-simplex coordinates for a
    multiindex alpha with length n, based on a 1D node family."""
    d = len(alpha)
    n = sum(alpha)
    b = numpy.zeros((d,), dtype="d")
    xn = family[n]
    if xn is None:
        return b
    if d == 2:
        b[:] = xn[list(alpha)]
        return b
    weight = 0.0
    for i in range(d):
        w = xn[n - alpha[i]]
        alpha_noti = alpha[:i] + alpha[i+1:]
        br = recursive(alpha_noti, family)
        b[:i] += w * br[:i]
        b[i+1:] += w * br[i:]
        weight += w
    b /= weight
    return b


def recursive_points(family, vertices, order, interior=0):
    X = numpy.array(vertices)
    get_point = lambda alpha: tuple(numpy.dot(recursive(alpha, family), X))
    return list(map(get_point, multiindex_equal(len(vertices), order, interior=interior)))


def make_points(family, ref_el, dim, entity_id, order):
    """Constructs a lattice of points on the entity_id:th
    facet of dimension dim.  Order indicates how many points to
    include in each direction."""
    if dim == 0:
        return (ref_el.get_vertices()[entity_id], )
    elif 0 < dim < ref_el.get_spatial_dimension():
        entity_verts = \
            ref_el.get_vertices_of_subcomplex(
                ref_el.get_topology()[dim][entity_id])
        return recursive_points(family, entity_verts, order, interior=1)
    elif dim == ref_el.get_spatial_dimension():
        return recursive_points(family, ref_el.get_vertices(), order, interior=1)
    else:
        raise ValueError("illegal dimension")


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    ref_el = reference_element.ufc_simplex(2)
    h = numpy.sqrt(3)
    s = 2*h/3
    ref_el.vertices = [(0, s), (-1.0, s-h), (1.0, s-h)]
    x = numpy.array(ref_el.vertices + ref_el.vertices[:1])
    plt.plot(x[:, 0], x[:, 1], "k")

    order = 7
    rule = "gll"
    dg_rule = "gl"

    # rule = "equispaced"
    # dg_rule = "dg_equispaced"

    family = make_node_family(rule)
    dg_family = make_node_family(dg_rule)

    for d in range(1, 4):
        print(make_points(family, reference_element.ufc_simplex(d), d, 0, d))

    topology = ref_el.get_topology()
    for dim in topology:
        for entity in topology[dim]:
            pts = make_points(family, ref_el, dim, entity, order)
            if len(pts):
                x = numpy.array(pts)
                for r in range(1, 3):
                    th = r * (2*numpy.pi)/3
                    ct = numpy.cos(th)
                    st = numpy.sin(th)
                    Q = numpy.array([[ct, -st], [st, ct]])
                    x = numpy.concatenate([x, numpy.dot(x, Q)])
                plt.scatter(x[:, 0], x[:, 1])

    x0 = 2.0
    h = -h
    s = 2*h/3
    ref_el = reference_element.ufc_simplex(2)
    ref_el.vertices = [(x0, s), (x0-1.0, s-h), (x0+1.0, s-h)]

    x = numpy.array(ref_el.vertices + ref_el.vertices[:1])
    d = len(ref_el.vertices)
    x0 = sum(x[:d])/d
    plt.plot(x[:, 0], x[:, 1], "k")

    pts = recursive_points(dg_family, ref_el.vertices, order)
    x = numpy.array(pts)
    for r in range(1, 3):
        th = r * (2*numpy.pi)/3
        ct = numpy.cos(th)
        st = numpy.sin(th)
        Q = numpy.array([[ct, -st], [st, ct]])
        x = numpy.concatenate([x, numpy.dot(x-x0, Q)+x0])
    plt.scatter(x[:, 0], x[:, 1])

    plt.gca().set_aspect('equal', 'box')
    plt.show()
