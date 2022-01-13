# Copyright (C) 2015 Imperial College London and others.
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# Written by Pablo D. Brubeck (brubeck@protonmail.com), 2021

import numpy

from FIAT import finite_element, polynomial_set, dual_set, functional, quadrature
from FIAT.reference_element import LINE
from FIAT.barycentric_interpolation import barycentric_interpolation
from FIAT.lagrange import make_entity_permutations
from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre


def sym_eig(A, B):
    """
    A numpy-only implementation of `scipy.linalg.eigh`
    """
    L = numpy.linalg.cholesky(B)
    Linv = numpy.linalg.inv(L)
    C = numpy.dot(Linv, numpy.dot(A, Linv.T))
    Z, W = numpy.linalg.eigh(C)
    V = numpy.dot(Linv.T, W)
    return Z, V


def semhat(elem, rule):
    """
    Construct Laplacian stiffness and mass matrices

    :arg elem: the element
    :arg rule: quadrature rule

    :returns: 5-tuple of
        Ahat: reference stiffness matrix
        Bhat: reference mass matrix
        Jhat: tabulation of the shape functions on the quadrature nodes
        Dhat: tabulation of the first derivative of the shape functions on the quadrature nodes
        xhat: nodes of the element
    """
    basis = elem.tabulate(1, rule.get_points())
    Jhat = basis[(0,)]
    Dhat = basis[(1,)]
    what = rule.get_weights()
    Ahat = numpy.dot(numpy.multiply(Dhat, what), Dhat.T)
    Bhat = numpy.dot(numpy.multiply(Jhat, what), Jhat.T)
    xhat = numpy.array([list(x.get_point_dict().keys())[0][0] for x in elem.dual_basis()])
    return Ahat, Bhat, Jhat, Dhat, xhat


class FDMDualSet(dual_set.DualSet):
    """The dual basis for 1D (dis)continuous elements with FDM shape functions."""
    def __init__(self, ref_el, degree, formdegree):

        elem = GaussLobattoLegendre(ref_el, degree)
        rule = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
        Ahat, Bhat, Jhat, _, xhat = semhat(elem, rule)

        Sfdm = numpy.eye(Ahat.shape[0])
        if Sfdm.shape[0] > 2:
            bdof = (0, -1)
            idof = slice(1, -1)
            _, Sfdm[idof, idof] = sym_eig(Ahat[idof, idof], Bhat[idof, idof])
            Sii = Sfdm[idof, idof]
            Sbb = Sfdm[numpy.ix_(bdof, bdof)]
            Sfdm[idof, bdof] = numpy.dot(Sii, numpy.dot(Sii.T, numpy.dot(Bhat[idof, bdof], -Sbb)))

        self.gll_points = xhat
        self.gll_tabulation = Sfdm
        basis = numpy.dot(Sfdm.T, Jhat)
        nodes = [functional.IntegralMoment(ref_el, rule, phi) for phi in basis]
        nodes[:degree+1:degree] = [functional.PointEvaluation(ref_el, x) for x in ref_el.get_vertices()]

        entity_ids = {}
        entity_permutations = {}
        if formdegree == 0:
            entity_ids[0] = {0: [0], 1: [degree]}
            entity_ids[1] = {0: list(range(1, degree))}
            entity_permutations[0] = {0: {0: [0]}, 1: {0: [0]}}
            entity_permutations[1] = {0: make_entity_permutations(1, degree - 1)}
        else:
            entity_ids[0] = {0: [], 1: []}
            entity_ids[1] = {0: list(range(0, degree+1))}
            entity_permutations[0] = {0: {0: []}, 1: {0: []}}
            entity_permutations[1] = {0: make_entity_permutations(1, degree + 1)}

        super(FDMDualSet, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class FDMElement(finite_element.CiarletElement):
    """1D (dis)continuous element with FDM shape functions."""
    def __init__(self, ref_el, degree, formdegree=0):
        if ref_el.shape != LINE:
            raise ValueError("FDM elements are only defined in one dimension.")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = FDMDualSet(ref_el, degree, formdegree)
        super(FDMElement, self).__init__(poly_set, dual, degree, formdegree)

    def tabulate(self, order, points, entity=None):
        # This overrides the default with a more numerically stable algorithm

        if entity is None:
            entity = (self.ref_el.get_dimension(), 0)

        entity_dim, entity_id = entity
        transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
        xsrc = self.dual.gll_points
        xdst = numpy.array(list(map(transform, points))).flatten()
        tabulation = barycentric_interpolation(xsrc, xdst, order=order)
        for key in tabulation:
            tabulation[key] = numpy.dot(self.dual.gll_tabulation.T, tabulation[key])
        return tabulation
