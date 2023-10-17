# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by David A. Ham (david.ham@imperial.ac.uk), 2015
#
# Modified by Pablo D. Brubeck (brubeck@protonmail.com), 2021

from FIAT import (finite_element, polynomial_set, dual_set, functional,
                  quadrature, recursive_points)
from FIAT.reference_element import LINE, TRIANGLE, TETRAHEDRON, UFCInterval
from FIAT.orientation_utils import make_entity_permutations_simplex
from FIAT.barycentric_interpolation import LagrangePolynomialSet


class GaussLobattoLegendrePointSet(recursive_points.RecursivePointSet):
    """Recursive point set on simplices based on the Gauss-Lobatto points on
    the interval"""
    def __init__(self):
        ref_el = UFCInterval()
        lr = quadrature.GaussLobattoLegendreQuadratureLineRule
        f = lambda n: lr(ref_el, n + 1).get_points() if n else None
        super(GaussLobattoLegendrePointSet, self).__init__(f)


class GaussLobattoLegendreDualSet(dual_set.DualSet):
    """The dual basis for continuous elements with nodes at the
    (recursive) Gauss-Lobatto points."""
    point_set = GaussLobattoLegendrePointSet()

    def __init__(self, ref_el, degree):
        entity_ids = {}
        nodes = []
        entity_permutations = {}

        # make nodes by getting points
        # need to do this dimension-by-dimension, facet-by-facet
        top = ref_el.get_topology()

        cur = 0
        for dim in sorted(top):
            entity_ids[dim] = {}
            entity_permutations[dim] = {}
            perms = {0: [0]} if dim == 0 else make_entity_permutations_simplex(dim, degree - dim)
            for entity in sorted(top[dim]):
                pts_cur = self.point_set.make_points(ref_el, dim, entity, degree)
                nodes_cur = [functional.PointEvaluation(ref_el, x)
                             for x in pts_cur]
                nnodes_cur = len(nodes_cur)
                nodes += nodes_cur
                entity_ids[dim][entity] = list(range(cur, cur + nnodes_cur))
                cur += nnodes_cur
                entity_permutations[dim][entity] = perms

        super(GaussLobattoLegendreDualSet, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class GaussLobattoLegendre(finite_element.CiarletElement):
    """Simplicial continuous element with nodes at the (recursive) Gauss-Lobatto points."""
    def __init__(self, ref_el, degree):
        if ref_el.shape not in {LINE, TRIANGLE, TETRAHEDRON}:
            raise ValueError("Gauss-Lobatto-Legendre elements are only defined on simplices.")
        dual = GaussLobattoLegendreDualSet(ref_el, degree)
        if ref_el.shape == LINE:
            points = []
            for node in dual.nodes:
                # Assert singleton point for each node.
                pt, = node.get_point_dict().keys()
                points.append(pt)
            poly_set = LagrangePolynomialSet(ref_el, points)
        else:
            poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        formdegree = 0  # 0-form
        super(GaussLobattoLegendre, self).__init__(poly_set, dual, degree, formdegree)
