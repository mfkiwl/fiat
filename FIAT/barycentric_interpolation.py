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


def barycentric_interpolation(nodes, wts, dmat, pts, order=0):
    """Evaluates a Lagrange basis on a line reference element
    via the second barycentric interpolation formula. See Berrut and Trefethen (2004)
    https://doi.org/10.1137/S0036144502417715 Eq. (4.2) & (9.4)
    """
    if pts.dtype == object:
        from sympy import simplify
        sp_simplify = numpy.vectorize(simplify)
    else:
        sp_simplify = lambda x: x
    phi = numpy.add.outer(-nodes, pts)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        numpy.reciprocal(phi, out=phi)
        numpy.multiply(phi, wts[:, None], out=phi)
        numpy.multiply(1.0 / numpy.sum(phi, axis=0), phi, out=phi)
    phi[phi != phi] = 1.0

    phi = sp_simplify(phi)
    results = [phi]
    for r in range(order):
        phi = sp_simplify(numpy.dot(dmat, phi))
        results.append(phi)
    return results


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
    """Lagrange polynomial expansion set for given points the line."""
    def __init__(self, ref_el, pts):
        self.points = pts
        self.x = numpy.array(pts).flatten()
        self.cell_node_map = expansions.compute_cell_point_map(ref_el, numpy.transpose(pts), unique=False)
        self.dmats = []
        self.weights = []
        for ibfs in self.cell_node_map:
            dmat, wts = make_dmat(self.x[ibfs])
            self.dmats.append(dmat)
            self.weights.append(wts)

        self.degree = max(len(wts) for wts in self.weights)-1
        self.recurrence_order = self.degree + 1
        super(LagrangeLineExpansionSet, self).__init__(ref_el)

    def get_num_members(self, n):
        return len(self.points)

    def get_cell_node_map(self, n):
        return self.cell_node_map

    def get_points(self):
        return self.points

    def get_dmats(self, degree):
        return [dmat.T for dmat in self.dmats]

    def _tabulate(self, n, pts, order=0):
        num_members = self.get_num_members(n)
        cell_node_map = self.get_cell_node_map(n)
        cell_point_map = expansions.compute_cell_point_map(self.ref_el, pts)
        pts = numpy.asarray(pts).flatten()
        results = None
        for ibfs, ipts, wts, dmat in zip(cell_node_map, cell_point_map, self.weights, self.dmats):
            vals = barycentric_interpolation(self.x[ibfs], wts, dmat, pts[ipts], order=order)
            if len(cell_node_map) == 1:
                results = vals
            else:
                if results is None:
                    results = [numpy.zeros((num_members, len(pts)), dtype=vals[0].dtype) for r in range(order+1)]
                indices = numpy.ix_(ibfs, ipts)
                for result, val in zip(results, vals):
                    result[indices] = val

        for r in range(order+1):
            shape = results[r].shape
            shape = shape[:1] + (1,)*r + shape[1:]
            results[r] = numpy.reshape(results[r], shape)
        return tuple(results)


class LagrangePolynomialSet(polynomial_set.PolynomialSet):

    def __init__(self, ref_el, pts, shape=tuple()):
        if ref_el.get_shape() != reference_element.LINE:
            raise ValueError("Invalid reference element type.")

        expansion_set = LagrangeLineExpansionSet(ref_el, pts)
        degree = expansion_set.degree
        if shape == tuple():
            num_components = 1
        else:
            flat_shape = numpy.ravel(shape)
            num_components = numpy.prod(flat_shape)
        num_exp_functions = expansion_set.get_num_members(degree)
        num_members = num_components * num_exp_functions
        embedded_degree = degree

        # set up coefficients
        if shape == tuple():
            coeffs = numpy.eye(num_members)
        else:
            coeffs_shape = (num_members, *shape, num_exp_functions)
            coeffs = numpy.zeros(coeffs_shape, "d")
            # use functional's index_iterator function
            cur_bf = 0
            for idx in index_iterator(shape):
                for exp_bf in range(num_exp_functions):
                    cur_idx = (cur_bf, *idx, exp_bf)
                    coeffs[cur_idx] = 1.0
                    cur_bf += 1

        super(LagrangePolynomialSet, self).__init__(ref_el, degree, embedded_degree,
                                                    expansion_set, coeffs)
