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
from recursivenodes.quadrature import gaussjacobi

from FIAT import reference_element, expansions, orthopoly


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

    def __init__(self, ref_el, m):
        # this gives roots on the default (-1,1) reference element
        #        (xs_ref, ws_ref) = gaussjacobi(m, a, b)
        (xs_ref, ws_ref) = gaussjacobi(m, 0., 0.)

        Ref1 = reference_element.DefaultLine()
        A, b = reference_element.make_affine_mapping(Ref1.get_vertices(),
                                                     ref_el.get_vertices())

        mapping = lambda x: numpy.dot(A, x) + b

        scale = numpy.linalg.det(A)

        xs = tuple([tuple(mapping(x_ref)[0]) for x_ref in xs_ref])
        ws = tuple([scale * w for w in ws_ref])

        QuadratureRule.__init__(self, ref_el, xs, ws)


class GaussLobattoLegendreQuadratureLineRule(QuadratureRule):
    """Implement the Gauss-Lobatto-Legendre quadrature rules on the interval using
    Greg von Winckel's implementation. This facilitates implementing
    spectral elements.

    The quadrature rule uses m points for a degree of precision of 2m-3.
    """
    def __init__(self, ref_el, m):
        if m < 2:
            raise ValueError(
                "Gauss-Labotto-Legendre quadrature invalid for fewer than 2 points")

        Ref1 = reference_element.DefaultLine()
        verts = Ref1.get_vertices()

        if m > 2:
            # Calculate the recursion coefficients.
            alpha, beta = orthopoly.rec_jacobi(m, 0, 0)
            xs_ref, ws_ref = orthopoly.lobatto(alpha, beta, verts[0][0], verts[1][0])
        else:
            # Special case for lowest order.
            xs_ref = [v[0] for v in verts[:]]
            ws_ref = (0.5 * (xs_ref[1] - xs_ref[0]), ) * 2

        A, b = reference_element.make_affine_mapping(Ref1.get_vertices(),
                                                     ref_el.get_vertices())

        mapping = lambda x: numpy.dot(A, x) + b

        scale = numpy.linalg.det(A)

        xs = tuple([tuple(mapping(x_ref)[0]) for x_ref in xs_ref])
        ws = tuple([scale * w for w in ws_ref])

        QuadratureRule.__init__(self, ref_el, xs, ws)


class GaussLegendreQuadratureLineRule(QuadratureRule):
    """Produce the Gauss--Legendre quadrature rules on the interval using
    the implementation in numpy. This facilitates implementing
    discontinuous spectral elements.

    The quadrature rule uses m points for a degree of precision of 2m-1.
    """
    def __init__(self, ref_el, m):
        if m < 1:
            raise ValueError(
                "Gauss-Legendre quadrature invalid for fewer than 2 points")

        xs_ref, ws_ref = numpy.polynomial.legendre.leggauss(m)

        A, b = reference_element.make_affine_mapping(((-1.,), (1.)),
                                                     ref_el.get_vertices())

        mapping = lambda x: numpy.dot(A, x) + b

        scale = numpy.linalg.det(A)

        xs = tuple([tuple(mapping(x_ref)[0]) for x_ref in xs_ref])
        ws = tuple([scale * w for w in ws_ref])

        QuadratureRule.__init__(self, ref_el, xs, ws)


class RadauQuadratureLineRule(QuadratureRule):
    """Produce the Gauss--Radau quadrature rules on the interval using
    an adaptation of Winkel's Matlab code.

    The quadrature rule uses m points for a degree of precision of 2m-1.
    """
    def __init__(self, ref_el, m, right=True):
        assert m >= 1
        N = m - 1
        # Use Chebyshev-Gauss-Radau nodes as initial guess for LGR nodes
        x = -numpy.cos(2 * numpy.pi * numpy.linspace(0, N, m) / (2 * N + 1))

        P = numpy.zeros((N + 1, N + 2))

        xold = 2

        free = numpy.arange(1, N + 1, dtype='int')

        while numpy.max(numpy.abs(x - xold)) > 5e-16:
            xold = x.copy()

            P[0, :] = (-1) ** numpy.arange(0, N + 2)
            P[free, 0] = 1
            P[free, 1] = x[free]

            for k in range(2, N + 2):
                P[free, k] = ((2 * k - 1) * x[free] * P[free, k - 1] - (k - 1) * P[free, k - 2]) / k

            x[free] = xold[free] - ((1 - xold[free]) / (N + 1)) * (P[free, N] + P[free, N + 1]) / (P[free, N] - P[free, N + 1])

        # The Legendre-Gauss-Radau Vandermonde
        P = P[:, :-1]
        # Compute the weights
        w = numpy.zeros(N + 1)
        w[0] = 2 / (N + 1) ** 2
        w[free] = (1 - x[free])/((N + 1) * P[free, -1])**2

        if right:
            x = numpy.flip(-x)
            w = numpy.flip(w)

        xs_ref = x
        ws_ref = w

        A, b = reference_element.make_affine_mapping(((-1.,), (1.)),
                                                     ref_el.get_vertices())

        mapping = lambda x: numpy.dot(A, x) + b

        scale = numpy.linalg.det(A)

        xs = tuple([tuple(mapping(x_ref)[0]) for x_ref in xs_ref])
        ws = tuple([scale * w for w in ws_ref])

        QuadratureRule.__init__(self, ref_el, xs, ws)


class CollapsedQuadratureTriangleRule(QuadratureRule):
    """Implements the collapsed quadrature rules defined in
    Karniadakis & Sherwin by mapping products of Gauss-Jacobi rules
    from the square to the triangle."""

    def __init__(self, ref_el, m):
        ptx, wx = gaussjacobi(m, 0., 0.)
        pty, wy = gaussjacobi(m, 1., 0.)

        # map ptx , pty
        pts_ref = [expansions.xi_triangle((x, y))
                   for x in ptx for y in pty]

        Ref1 = reference_element.DefaultTriangle()
        A, b = reference_element.make_affine_mapping(Ref1.get_vertices(),
                                                     ref_el.get_vertices())
        mapping = lambda x: numpy.dot(A, x) + b

        scale = numpy.linalg.det(A)

        pts = tuple([tuple(mapping(x)) for x in pts_ref])

        wts = [0.5 * scale * w1 * w2 for w1 in wx for w2 in wy]

        QuadratureRule.__init__(self, ref_el, tuple(pts), tuple(wts))


class CollapsedQuadratureTetrahedronRule(QuadratureRule):
    """Implements the collapsed quadrature rules defined in
    Karniadakis & Sherwin by mapping products of Gauss-Jacobi rules
    from the cube to the tetrahedron."""

    def __init__(self, ref_el, m):
        ptx, wx = gaussjacobi(m, 0., 0.)
        pty, wy = gaussjacobi(m, 1., 0.)
        ptz, wz = gaussjacobi(m, 2., 0.)

        # map ptx , pty
        pts_ref = [expansions.xi_tetrahedron((x, y, z))
                   for x in ptx for y in pty for z in ptz]

        Ref1 = reference_element.DefaultTetrahedron()
        A, b = reference_element.make_affine_mapping(Ref1.get_vertices(),
                                                     ref_el.get_vertices())
        mapping = lambda x: numpy.dot(A, x) + b

        scale = numpy.linalg.det(A)

        pts = tuple([tuple(mapping(x)) for x in pts_ref])

        wts = [scale * 0.125 * w1 * w2 * w3
               for w1 in wx for w2 in wy for w3 in wz]

        QuadratureRule.__init__(self, ref_el, tuple(pts), tuple(wts))


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
