import numpy
import pytest
from FIAT.reference_element import ufc_simplex, symmetric_simplex
from FIAT.macro import AlfeldSplit, MacroQuadratureRule
from FIAT.quadrature_schemes import create_quadrature


@pytest.mark.parametrize("sdim", (2, 3))
def test_split_entity_transform(sdim):
    T = ufc_simplex(sdim)
    TA = AlfeldSplit(T)
    TAT = TA.topology

    for i in range(1, sdim+1):
        TX = TA.construct_subelement(i)
        b = numpy.average(TX.get_vertices(), axis=0)
        for entity in TAT[i]:
            mapped_bary = TA.get_entity_transform(i, entity)(b)
            computed_bary = numpy.average(TA.get_vertices_of_subcomplex(TAT[i][entity]), axis=0)
            assert numpy.allclose(mapped_bary, computed_bary)


@pytest.mark.parametrize("sdim", (2, 3))
@pytest.mark.parametrize("degree", (4,))
@pytest.mark.parametrize("variant", ("gll", "equispaced"))
def test_split_make_points(sdim, degree, variant):
    T = ufc_simplex(sdim)
    TA = AlfeldSplit(T)
    TAT = TA.topology

    for i in range(1, sdim+1):
        TX = TA.construct_subelement(i)
        pts_ref = TX.make_points(i, 0, degree, variant=variant)
        for entity in TAT[i]:
            pts_entity = TA.make_points(i, entity, degree, variant=variant)
            mapping = TA.get_entity_transform(i, entity)
            mapped_pts = list(map(mapping, pts_ref))
            assert numpy.allclose(mapped_pts, pts_entity)


@pytest.mark.parametrize("sdim", (2, 3))
def test_macro_quadrature(sdim):
    T = symmetric_simplex(sdim)
    TA = AlfeldSplit(T)

    degree = 6
    Q_ref = create_quadrature(T, degree)
    Q = MacroQuadratureRule(TA, Q_ref)
    Q.get_points()
    # import matplotlib.pyplot as plt
    # plt.scatter(*Q.get_points().T)
    # plt.show()
