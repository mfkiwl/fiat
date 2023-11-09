# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by David A. Ham (david.ham@imperial.ac.uk), 2015
#
# Modified by Pablo D. Brubeck (brubeck@protonmail.com), 2021

from FIAT import finite_element, polynomial_set, lagrange
from FIAT.reference_element import LINE, TRIANGLE, TETRAHEDRON
from FIAT.barycentric_interpolation import LagrangePolynomialSet


class GaussLobattoLegendre(finite_element.CiarletElement):
    """Simplicial continuous element with nodes at the (recursive) Gauss-Lobatto points."""
    def __init__(self, ref_el, degree):
        if ref_el.shape not in {LINE, TRIANGLE, TETRAHEDRON}:
            raise ValueError("Gauss-Lobatto-Legendre elements are only defined on simplices.")
        dual = lagrange.LagrangeDualSet(ref_el, degree, variant="gll")
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
        super(GaussLobattoLegendre, self).__init__(poly_set, dual, degree, formdegree)
