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
    from FIAT import ufc_simplex
    s = ufc_simplex(dim)
    r = lambda x: x ** 0.5
    if dim == 2:
        s.vertices = [(0.0, 0.0), (-1.0, -r(3.0)), (1.0, -r(3.0))]
    elif dim == 3:
        s.vertices = [(r(3.0)/3, 0.0, 0.0), (-r(3.0)/6, 0.5, 0.0),
                      (-r(3.0)/6, -0.5, 0.0), (0.0, 0.0, r(6.0)/3)]
    return s


@pytest.mark.parametrize("dim, degree", sum(([(d, p) for p in range(0, 8-d)] for d in range(1, 4)), []))
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
    from FIAT import GaussLegendre, quadrature, expansions, ufc_simplex

    s = symmetric_simplex(dim)
    fe = GaussLegendre(s, degree)
    ndof = fe.space_dimension()
    assert ndof == expansions.polynomial_dimension(s, degree)

    points = np.zeros((ndof, dim), "d")
    for i, node in enumerate(fe.dual_basis()):
        points[i, :], = node.get_point_dict().keys()

    # Test that edge DOFs are located at the GL quadrature points
    lr = quadrature.GaussLegendreQuadratureLineRule(ufc_simplex(1), degree + 1)
    quadrature_points = lr.pts

    entity_dofs = fe.entity_dofs()
    edge_dofs = entity_dofs[1]
    for entity in edge_dofs:
        if len(edge_dofs[entity]) > 0:
            transform = s.get_entity_transform(1, entity)
            assert np.allclose(points[edge_dofs[entity]], np.array(list(map(transform, quadrature_points))))

    # TODO add rotational symmetry tests


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))
