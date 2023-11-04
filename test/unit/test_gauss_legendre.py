# Copyright (C) 2016 Imperial College London and others
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FIAT. If not, see <http://www.gnu.org/licenses/>.
#
# Authors:
#
# David Ham

import pytest
import numpy as np


def symmetric_simplex(dim):
    from FIAT.reference_element import ufc_simplex
    s = ufc_simplex(dim)
    if dim == 1:
        s.vertices = [(-1.,), (1.,)]
    elif dim == 2:
        h = 3.**0.5 / dim
        s.vertices = [(0., 1.), (-h, -0.5), (h, -0.5)]
    elif dim == 3:
        h = 3.**0.5 / dim
        s.vertices = [(-h, h, h), (h, -h, h), (h, h, -h), (h, h, h)]
    return s


@pytest.mark.parametrize("degree", range(0, 8))
@pytest.mark.parametrize("dim", (1, 2, 3))
def test_gl_basis_values(dim, degree):
    """Ensure that integrating a simple monomial produces the expected results."""
    from FIAT import GaussLegendre, make_quadrature

    s = symmetric_simplex(dim)
    q = make_quadrature(s, degree + 1)
    fe = GaussLegendre(s, degree)
    tab = fe.tabulate(0, q.pts)[(0,)*dim]

    for test_degree in range(degree + 1):
        v = lambda x: sum(x)**test_degree
        coefs = [n(v) for n in fe.dual.nodes]
        integral = np.dot(coefs, np.dot(tab, q.wts))
        reference = np.dot([v(x) for x in q.pts], q.wts)
        assert np.allclose(integral, reference, rtol=1e-14)


@pytest.mark.parametrize("dim, degree", [(1, 4), (2, 4), (3, 4)])
def test_symmetry(dim, degree):
    """ Ensure the dual basis has the right symmetry."""
    from FIAT import GaussLegendre, quadrature, expansions

    s = symmetric_simplex(dim)
    fe = GaussLegendre(s, degree)
    ndof = fe.space_dimension()
    assert ndof == expansions.polynomial_dimension(s, degree)

    points = np.zeros((ndof, dim), "d")
    for i, node in enumerate(fe.dual_basis()):
        points[i, :], = node.get_point_dict().keys()

    # Test that edge DOFs are located at the GL quadrature points
    line = s if dim == 1 else s.construct_subelement(1)
    lr = quadrature.GaussLegendreQuadratureLineRule(line, degree + 1)
    quadrature_points = lr.pts

    entity_dofs = fe.entity_dofs()
    edge_dofs = entity_dofs[1]
    for entity in edge_dofs:
        if len(edge_dofs[entity]) > 0:
            transform = s.get_entity_transform(1, entity)
            assert np.allclose(points[edge_dofs[entity]], np.array(list(map(transform, quadrature_points))))


@pytest.mark.parametrize("dim, degree", [(1, 128), (2, 64), (3, 16)])
def test_interpolation(dim, degree):
    from FIAT import GaussLobattoLegendre, quadrature
    from FIAT.polynomial_set import mis

    a = (1. + 0.5)
    a = 0.5 * a**2
    r2 = lambda x: 0.5 * np.linalg.norm(x, axis=-1)**2
    f = lambda x: np.exp(a / (r2(x) - a))
    df = lambda x: f(x) * (-a*(r2(x) - a)**-2)

    s = symmetric_simplex(dim)
    rule = quadrature.make_quadrature(s, degree + 1)
    points = rule.get_points()
    weights = rule.get_weights()

    f_at_pts = {}
    f_at_pts[(0,) * dim] = f(points)
    df_at_pts = df(points) * points.T
    alphas = mis(dim, 1)
    for alpha in alphas:
        i = next(j for j, aj in enumerate(alpha) if aj > 0)
        f_at_pts[alpha] = df_at_pts[i]

    print()
    scaleL2 = 1 / np.sqrt(np.dot(weights, f(points)**2))
    scaleH1 = 1 / np.sqrt(np.dot(weights, sum(f_at_pts[alpha]**2 for alpha in f_at_pts)))

    k = 1
    while k <= degree:
        fe = GaussLobattoLegendre(s, k)
        tab = fe.tabulate(1, points)
        coefficients = np.array([v(f) for v in fe.dual_basis()])

        alpha = (0,) * dim
        err = f_at_pts[alpha] - np.dot(coefficients, tab[alpha])
        errorL2 = scaleL2 * np.sqrt(np.dot(weights, err ** 2))

        err2 = sum((f_at_pts[alpha] - np.dot(coefficients, tab[alpha])) ** 2 for alpha in tab)
        errorH1 = scaleH1 * np.sqrt(np.dot(weights, err2))
        print("dim = %d, degree = %2d, L2-error = %.4E, H1-error = %.4E" % (dim, k, errorL2, errorH1))
        k = min(k * 2, k + 16)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
