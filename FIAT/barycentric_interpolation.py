# Copyright (C) 2021 Pablo D. Brubeck
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2021

import numpy
from FIAT import reference_element, expansions, polynomial_set
from FIAT.functional import index_iterator


def make_dmat(x):
    """Returns Lagrange differentiation matrix and barycentric weights
    associated with x[j]."""
    dmat = numpy.add.outer(-x, x)
    numpy.fill_diagonal(dmat, 1.0)
    wts = numpy.prod(dmat, axis=0)
    numpy.reciprocal(wts, out=wts)
    numpy.divide(numpy.divide.outer(wts, wts), dmat, out=dmat)
    numpy.fill_diagonal(dmat, dmat.diagonal() - numpy.sum(dmat, axis=0))
    return dmat, wts


class LagrangeLineExpansionSet(expansions.LineExpansionSet):
    """Evaluates a Lagrange basis on a line reference element
    via the second barycentric interpolation formula. See Berrut and Trefethen (2004)
    https://doi.org/10.1137/S0036144502417715 Eq. (4.2) & (9.4)
    """
    def __init__(self, ref_el, pts):
        self.points = pts
        self.x = numpy.array(pts).flatten()
        self.dmat, self.weights = make_dmat(self.x)
        super(LagrangeLineExpansionSet, self).__init__(ref_el)

    def get_num_members(self, n):
        return len(self.points)

    def get_points(self):
        return self.points

    def get_dmats(self, degree):
        return [self.dmat.T]

    def tabulate(self, n, pts):
        assert n == len(self.points)-1
        results = numpy.add.outer(-self.x, numpy.array(pts).flatten())
        with numpy.errstate(divide='ignore', invalid='ignore'):
            numpy.reciprocal(results, out=results)
            numpy.multiply(results, self.weights[:, None], out=results)
            numpy.multiply(1.0 / numpy.sum(results, axis=0), results, out=results)

        results[results != results] = 1.0
        if results.dtype == object:
            from sympy import simplify
            results = numpy.array(list(map(simplify, results)))
        return results

    def _tabulate(self, n, pts, order=0):
        results = [self.tabulate(n, pts)]
        for r in range(order):
            results.append(numpy.dot(self.dmat, results[-1]))
        for r in range(order+1):
            shape = results[r].shape
            shape = shape[:1] + (1,)*r + shape[1:]
            results[r] = numpy.reshape(results[r], shape)
        return results


class LagrangePolynomialSet(polynomial_set.PolynomialSet):

    def __init__(self, ref_el, pts, shape=tuple()):
        degree = len(pts) - 1
        if shape == tuple():
            num_components = 1
        else:
            flat_shape = numpy.ravel(shape)
            num_components = numpy.prod(flat_shape)
        num_exp_functions = expansions.polynomial_dimension(ref_el, degree)
        num_members = num_components * num_exp_functions
        embedded_degree = degree
        if ref_el.get_shape() == reference_element.LINE:
            expansion_set = LagrangeLineExpansionSet(ref_el, pts)
        else:
            raise ValueError("Invalid reference element type.")

        # set up coefficients
        if shape == tuple():
            coeffs = numpy.eye(num_members)
        else:
            coeffs_shape = (num_members, *shape, num_exp_functions)
            coeffs = numpy.zeros(coeffs_shape, "d")
            # use functional's index_iterator function
            cur_bf = 0
            for idx in index_iterator(shape):
                n = expansions.polynomial_dimension(ref_el, embedded_degree)
                for exp_bf in range(n):
                    cur_idx = (cur_bf, *idx, exp_bf)
                    coeffs[cur_idx] = 1.0
                    cur_bf += 1

        super(LagrangePolynomialSet, self).__init__(ref_el, degree, embedded_degree,
                                                    expansion_set, coeffs)
