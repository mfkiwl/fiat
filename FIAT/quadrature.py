# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Modified by Marie E. Rognes (meg@simula.no), 2012
# Modified by David A. Ham (david.ham@imperial.ac.uk), 2015

import itertools
import numpy
from recursivenodes.quadrature import gaussjacobi, simplexgausslegendre

from FIAT import reference_element


class QuadratureRule(object):
    """General class that models integration over a reference element
    as the weighted sum of a function evaluated at a set of points."""

    def __init__(self, ref_el, pts, wts):
        if len(wts) != len(pts):
            raise ValueError("Have %d weights, but %d points" % (len(wts), len(pts)))

        self.ref_el = ref_el
        self.pts = pts
        self.wts = wts

    def get_points(self):
        return numpy.array(self.pts)

    def get_weights(self):
        return numpy.array(self.wts)

    def integrate(self, f):
        return sum(w * f(x) for x, w in zip(self.pts, self.wts))


class GaussJacobiQuadratureLineRule(QuadratureRule):
    """Gauss-Jacobi quadature rule determined by Jacobi weights a and b
    using m roots of m:th order Jacobi polynomial."""

    def __init__(self, ref_el, m, a=0, b=0):
        # this gives roots on the default (-1,1) reference element
        #        (xs_ref, ws_ref) = gaussjacobi(m, a, b)
        xs_ref, ws_ref = gaussjacobi(m, a, b)
        xs_ref = xs_ref.reshape((-1, 1))

        Ref1 = reference_element.DefaultLine()
        A, b = reference_element.make_affine_mapping(Ref1.get_vertices(),
                                                     ref_el.get_vertices())
        mapping = lambda x: numpy.dot(x, A.T) + b[None, :]
        ws_ref *= numpy.linalg.det(A)
        xs = tuple(map(tuple, mapping(xs_ref)))
        ws = tuple(ws_ref)

        QuadratureRule.__init__(self, ref_el, xs, ws)


class GaussLobattoLegendreQuadratureLineRule(QuadratureRule):
    """Gauss-Lobatto-Legendre quadrature rule on the interval.

    The quadrature rule uses m points for a degree of precision of 2m-3.
    """
    def __init__(self, ref_el, m):
        if m < 2:
            raise ValueError(
                "Gauss-Labotto-Legendre quadrature invalid for fewer than 2 points")

        verts = ref_el.vertices
        volume = ref_el.volume()
        if m > 2:
            # Make the interior points and weights
            rule = GaussJacobiQuadratureLineRule(ref_el, m-2, 1, 1)
            # Remove the bubble weight from the quadrature weights
            x = rule.get_points().reshape((-1,))
            bubble = (2.0 / volume)**2 * (x - verts[0][0]) * (verts[1][0] - x)
            wts = rule.get_weights() / bubble
            pts = rule.pts
        else:
            # Special case for lowest order.
            wts = ()
            pts = ()

        # Get the weight at the endpoints via sum(ws) == volume
        w0 = 0.5*(volume - sum(wts))
        xs = (verts[0], *pts, verts[1])
        ws = (w0, *wts, w0)
        QuadratureRule.__init__(self, ref_el, xs, ws)


class GaussLegendreQuadratureLineRule(GaussJacobiQuadratureLineRule):
    """Gauss--Legendre quadrature rule on the interval.

    The quadrature rule uses m points for a degree of precision of 2m-1.
    """
    def __init__(self, ref_el, m):
        super(GaussLegendreQuadratureLineRule, self).__init__(ref_el, m)


class RadauQuadratureLineRule(QuadratureRule):
    """Gauss--Radau quadrature rule on the interval.

    The quadrature rule uses m points for a degree of precision of 2m-1.
    """
    def __init__(self, ref_el, m, right=True):
        if m < 1:
            raise ValueError(
                "Gauss-Radau quadrature invalid for fewer than 1 points")

        right = int(right)
        x0 = ref_el.vertices[right]
        volume = ref_el.volume()
        if m > 1:
            # Make the interior points and weights
            rule = GaussJacobiQuadratureLineRule(ref_el, m-1, right, 1-right)
            # Remove the hat weight from the quadrature weights
            x = rule.get_points().reshape((-1,))
            hat = (2.0 / volume) * abs(x0[0] - x)
            wts = rule.get_weights() / hat
            pts = rule.pts
        else:
            # Special case for lowest order.
            wts = ()
            pts = ()

        # Get the weight at the endpoint via sum(ws) == volume
        w0 = volume - sum(wts)
        xs = (*pts, x0) if right else (x0, *pts)
        ws = (*wts, w0) if right else (w0, *wts)
        QuadratureRule.__init__(self, ref_el, xs, ws)


class CollapsedQuadratureSimplexRule(QuadratureRule):
    """Implements the collapsed quadrature rules defined in
    Karniadakis & Sherwin by mapping products of Gauss-Jacobi rules
    from the hypercube to the simplex."""

    def __init__(self, ref_el, m):
        dim = ref_el.get_spatial_dimension()
        pts_ref, wts = simplexgausslegendre(dim, m)
        pts_ref = pts_ref.reshape((-1, dim))
        wts = wts.reshape((-1,))

        Ref1 = reference_element.default_simplex(dim)
        A, b = reference_element.make_affine_mapping(Ref1.get_vertices(),
                                                     ref_el.get_vertices())
        mapping = lambda x: numpy.dot(x, A.T) + b[None, :]
        pts = tuple(map(tuple, mapping(pts_ref)))
        wts *= numpy.linalg.det(A)
        QuadratureRule.__init__(self, ref_el, pts, tuple(wts))


class CollapsedQuadratureTriangleRule(CollapsedQuadratureSimplexRule):
    """Implements the collapsed quadrature rules defined in
    Karniadakis & Sherwin by mapping products of Gauss-Jacobi rules
    from the square to the triangle."""
    pass


class CollapsedQuadratureTetrahedronRule(CollapsedQuadratureSimplexRule):
    """Implements the collapsed quadrature rules defined in
    Karniadakis & Sherwin by mapping products of Gauss-Jacobi rules
    from the cube to the tetrahedron."""
    pass


class UFCTetrahedronFaceQuadratureRule(QuadratureRule):
    """Highly specialized quadrature rule for the face of a
    tetrahedron, mapped from a reference triangle, used for higher
    order Nedelecs"""

    def __init__(self, face_number, degree):

        # Create quadrature rule on reference triangle
        reference_triangle = reference_element.UFCTriangle()
        reference_rule = make_quadrature(reference_triangle, degree)
        ref_points = reference_rule.get_points()
        ref_weights = reference_rule.get_weights()

        # Get geometry information about the face of interest
        reference_tet = reference_element.UFCTetrahedron()
        face = reference_tet.get_topology()[2][face_number]
        vertices = reference_tet.get_vertices_of_subcomplex(face)

        # Use tet to map points and weights on the appropriate face
        vertices = [numpy.array(list(vertex)) for vertex in vertices]
        x0 = vertices[0]
        J = numpy.vstack([vertices[1] - x0, vertices[2] - x0]).T
        # This is just a very numpyfied way of writing J*p + x0:
        points = numpy.einsum("ij,kj->ki", J, ref_points) + x0

        # Map weights: multiply reference weights by sqrt(|J^T J|)
        detJTJ = numpy.linalg.det(numpy.dot(J.T, J))
        weights = numpy.sqrt(detJTJ) * ref_weights

        # Initialize super class with new points and weights
        QuadratureRule.__init__(self, reference_tet, points, weights)
        self._reference_rule = reference_rule
        self._J = J

    def reference_rule(self):
        return self._reference_rule

    def jacobian(self):
        return self._J


def make_quadrature(ref_el, m):
    """Returns the collapsed quadrature rule using m points per
    direction on the given reference element. In the tensor product
    case, m is a tuple."""

    if isinstance(m, tuple):
        min_m = min(m)
    else:
        min_m = m

    msg = "Expecting at least one (not %d) quadrature point per direction" % min_m
    assert (min_m > 0), msg

    if ref_el.get_shape() == reference_element.POINT:
        return QuadratureRule(ref_el, [()], [1])
    elif ref_el.get_shape() == reference_element.LINE:
        return GaussJacobiQuadratureLineRule(ref_el, m)
    elif ref_el.get_shape() == reference_element.TRIANGLE:
        return CollapsedQuadratureTriangleRule(ref_el, m)
    elif ref_el.get_shape() == reference_element.TETRAHEDRON:
        return CollapsedQuadratureTetrahedronRule(ref_el, m)
    elif ref_el.get_shape() == reference_element.QUADRILATERAL:
        line_rule = GaussJacobiQuadratureLineRule(ref_el.construct_subelement(1), m)
        return make_tensor_product_quadrature(line_rule, line_rule)
    elif ref_el.get_shape() == reference_element.HEXAHEDRON:
        line_rule = GaussJacobiQuadratureLineRule(ref_el.construct_subelement(1), m)
        return make_tensor_product_quadrature(line_rule, line_rule, line_rule)
    else:
        raise ValueError("Unable to make quadrature for cell: %s" % ref_el)


def make_tensor_product_quadrature(*quad_rules):
    """Returns the quadrature rule for a TensorProduct cell, by combining
    the quadrature rules of the components."""
    ref_el = reference_element.TensorProductCell(*[q.ref_el
                                                   for q in quad_rules])
    # Coordinates are "concatenated", weights are multiplied
    pts = [list(itertools.chain(*pt_tuple))
           for pt_tuple in itertools.product(*[q.pts for q in quad_rules])]
    wts = [numpy.prod(wt_tuple)
           for wt_tuple in itertools.product(*[q.wts for q in quad_rules])]
    return QuadratureRule(ref_el, pts, wts)
