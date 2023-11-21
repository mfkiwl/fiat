# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set, functional
from FIAT.orientation_utils import make_entity_permutations_simplex
from FIAT.barycentric_interpolation import LagrangePolynomialSet
from FIAT.reference_element import LINE


class LagrangeDualSet(dual_set.DualSet):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points."""

    def __init__(self, ref_el, degree, variant="equispaced"):
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
                pts_cur = ref_el.make_points(dim, entity, degree, variant=variant)
                nodes_cur = [functional.PointEvaluation(ref_el, x)
                             for x in pts_cur]
                nnodes_cur = len(nodes_cur)
                nodes += nodes_cur
                entity_ids[dim][entity] = list(range(cur, cur + nnodes_cur))
                cur += nnodes_cur
                entity_permutations[dim][entity] = perms

        super(LagrangeDualSet, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class Lagrange(finite_element.CiarletElement):
    """The Lagrange finite element.  It is what it is."""

    def __init__(self, ref_el, degree, variant="equispaced"):
        dual = LagrangeDualSet(ref_el, degree, variant=variant)
        if ref_el.shape == LINE:
            # In 1D we can use the primal basis as the expansion set,
            # avoiding any round-off coming from a basis transformation
            points = []
            for node in dual.nodes:
                # Assert singleton point for each node.
                pt, = node.get_point_dict().keys()
                points.append(pt)
            poly_set = LagrangePolynomialSet(ref_el, points)
        else:
            poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        formdegree = 0  # 0-form
        super(Lagrange, self).__init__(poly_set, dual, degree, formdegree)
