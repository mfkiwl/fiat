# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2023

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


def multiindex_equal(d, isum, imin=0):
    """A generator for d-tuple multi-indices whose sum is isum and minimum is imin.
    """
    if d <= 0:
        return
    imax = isum - (d - 1) * imin
    if imax < imin:
        return
    for i in range(imin, imax):
        for a in multiindex_equal(d - 1, isum - i, imin=imin):
            yield a + (i,)
    yield (imin,) * (d - 1) + (imax,)


class RecursivePointSet(object):
    """Class to construct recursive points on simplices based on a family of
    points on the unit interval.  This class essentially is a
    lazy-evaluate-and-cache dictionary: the user passes a routine to evaluate
    entries for unknown keys """

    def __init__(self, family):
        self._family = family
        self._cache = {}

    def interval_points(self, degree):
        try:
            return self._cache[degree]
        except KeyError:
            x = self._family(degree)
            if x is not None:
                if not isinstance(x, numpy.ndarray):
                    x = numpy.array(x)
                x = x.reshape((-1,))
                x.setflags(write=False)
            return self._cache.setdefault(degree, x)

    def _recursive(self, alpha):
        """The barycentric (d-1)-simplex coordinates for a
        multiindex alpha of length d and sum n, based on a 1D node family."""
        d = len(alpha)
        n = sum(alpha)
        b = numpy.zeros((d,), dtype="d")
        xn = self.interval_points(n)
        if xn is None:
            return b
        if d == 2:
            b[:] = xn[list(alpha)]
            return b
        weight = 0.0
        for i in range(d):
            w = xn[n - alpha[i]]
            alpha_noti = alpha[:i] + alpha[i+1:]
            br = self._recursive(alpha_noti)
            b[:i] += w * br[:i]
            b[i+1:] += w * br[i:]
            weight += w
        b /= weight
        return b

    def recursive_points(self, vertices, order, interior=0):
        X = numpy.array(vertices)
        get_point = lambda alpha: tuple(numpy.dot(self._recursive(alpha), X))
        return list(map(get_point, multiindex_equal(len(vertices), order, interior)))

    def make_points(self, ref_el, dim, entity_id, order):
        """Constructs a lattice of points on the entity_id:th
        facet of dimension dim.  Order indicates how many points to
        include in each direction."""
        if dim == 0:
            return (ref_el.get_vertices()[entity_id], )
        elif 0 < dim < ref_el.get_spatial_dimension():
            entity_verts = \
                ref_el.get_vertices_of_subcomplex(
                    ref_el.get_topology()[dim][entity_id])
            return self.recursive_points(entity_verts, order, 1)
        elif dim == ref_el.get_spatial_dimension():
            return self.recursive_points(ref_el.get_vertices(), order, 1)
        else:
            raise ValueError("illegal dimension")
