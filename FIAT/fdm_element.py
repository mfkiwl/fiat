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
from FIAT.P0 import P0Dual


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


class FDMDual(dual_set.DualSet):
    """The dual basis for 1D CG elements with FDM shape functions."""
    def __init__(self, ref_el, degree, order=1):
        bc_nodes = []
        for x in ref_el.get_vertices():
            bc_nodes.append([functional.PointEvaluation(ref_el, x),
                             *[functional.PointDerivative(ref_el, x, [alpha]) for alpha in range(1, order)]])
        bc_nodes[1].reverse()
        k = len(bc_nodes[0])
        idof = slice(k, -k)
        bdof = list(range(-k, k))
        bdof = bdof[k:] + bdof[:k]

        # Define the generalized eigenproblem on a GLL element
        gll = GaussLobattoLegendre(ref_el, degree)
        xhat = numpy.array([list(x.get_point_dict().keys())[0][0] for x in gll.dual_basis()])

        # Tabulate the BC nodes
        constraints = gll.tabulate(order-1, ref_el.get_vertices())
        C = numpy.column_stack(list(constraints.values()))
        perm = list(range(len(bdof)))
        perm = perm[::2] + perm[-1::-2]
        C = C[:, perm].T
        # Tabulate the basis that splits the DOFs into interior and bcs
        E = numpy.eye(degree+1)
        E[bdof, idof] = -C[:, idof]
        E[bdof, :] = numpy.dot(numpy.linalg.inv(C[:, bdof]), E[bdof, :])

        # Assemble the constrained Galerkin matrices on the reference cell
        rule = quadrature.GaussLegendreQuadratureLineRule(ref_el, degree+1)
        phi = gll.tabulate(order, rule.get_points())
        E0 = numpy.dot(E, phi[(0, )].T)
        Ek = numpy.dot(E, phi[(order, )].T)
        B = numpy.dot(numpy.multiply(E0.T, rule.get_weights()), E0)
        A = numpy.dot(numpy.multiply(Ek.T, rule.get_weights()), Ek)

        # Eigenfunctions in the constrained basis
        S = numpy.eye(A.shape[0])
        if S.shape[0] > len(bdof):
            _, Sii = sym_eig(A[idof, idof], B[idof, idof])
            S[idof, idof] = Sii
            S[idof, bdof] = numpy.dot(Sii, numpy.dot(Sii.T, -B[idof, bdof]))

        # Eigenfunctions in the Lagrange basis
        S = numpy.dot(E, S)
        self.gll_points = xhat
        self.gll_tabulation = S.T
        # Interpolate eigenfunctions onto the quadrature points
        basis = numpy.dot(S.T, phi[(0, )])
        nodes = bc_nodes[0] + [functional.IntegralMoment(ref_el, rule, phi) for phi in basis[idof]] + bc_nodes[1]

        entity_ids = {0: {0: [0], 1: [degree]},
                      1: {0: list(range(1, degree))}}
        entity_permutations = {}
        entity_permutations[0] = {0: {0: [0]}, 1: {0: [0]}}
        entity_permutations[1] = {0: make_entity_permutations(1, degree - 1)}
        super(FDMDual, self).__init__(nodes, ref_el, entity_ids, entity_permutations)


class FDMLagrange(finite_element.CiarletElement):
    """1D CG element with interior shape functions that diagonalize the Laplacian."""
    _order = 1

    def __init__(self, ref_el, degree):
        if ref_el.shape != LINE:
            raise ValueError("FDM elements are only defined in one dimension.")
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        if degree == 0:
            dual = P0Dual(ref_el)
        else:
            dual = FDMDual(ref_el, degree, order=self._order)
        formdegree = 0
        super(FDMLagrange, self).__init__(poly_set, dual, degree, formdegree)

    def tabulate(self, order, points, entity=None):
        # This overrides the default with a more numerically stable algorithm
        if hasattr(self.dual, "gll_points"):
            if entity is None:
                entity = (self.ref_el.get_dimension(), 0)

            entity_dim, entity_id = entity
            transform = self.ref_el.get_entity_transform(entity_dim, entity_id)
            xsrc = self.dual.gll_points
            xdst = numpy.array(list(map(transform, points))).flatten()
            tabulation = barycentric_interpolation(xsrc, xdst, order=order)
            for key in tabulation:
                tabulation[key] = numpy.dot(self.dual.gll_tabulation, tabulation[key])
            return tabulation
        else:
            return super(FDMLagrange, self).tabulate(order, points, entity)


class FDMHermite(FDMLagrange):
    """1D CG element with interior shape functions that diagonalize the biharmonic operator."""
    _order = 2
