import numpy
import pytest
from FIAT.reference_element import ufc_simplex
from FIAT.macro import AlfeldSplit, UniformSplit, MacroQuadratureRule
from FIAT.quadrature_schemes import create_quadrature


@pytest.fixture(params=("T", "S"))
def cell(request):
    return {"T": ufc_simplex(2),
            "S": ufc_simplex(3)}[request.param]


@pytest.mark.parametrize("split", (AlfeldSplit, UniformSplit))
def test_split_entity_transform(split, cell):
    split_cell = split(cell)
    top = split_cell.get_topology()
    sdim = cell.get_spatial_dimension()
    for dim in range(1, sdim+1):
        ref_el = split_cell.construct_subelement(dim)
        b = numpy.average(ref_el.get_vertices(), axis=0)
        for entity in top[dim]:
            mapped_bary = split_cell.get_entity_transform(dim, entity)(b)
            computed_bary = numpy.average(split_cell.get_vertices_of_subcomplex(top[dim][entity]), axis=0)
            assert numpy.allclose(mapped_bary, computed_bary)


@pytest.mark.parametrize("degree", (4,))
@pytest.mark.parametrize("variant", ("gll", "equispaced"))
@pytest.mark.parametrize("split", (AlfeldSplit, UniformSplit))
def test_split_make_points(split, cell, degree, variant):
    split_cell = split(cell)
    top = split_cell.get_topology()
    sdim = cell.get_spatial_dimension()
    for i in range(1, sdim+1):
        ref_el = split_cell.construct_subelement(i)
        pts_ref = ref_el.make_points(i, 0, degree, variant=variant)
        for entity in top[i]:
            pts_entity = split_cell.make_points(i, entity, degree, variant=variant)
            mapping = split_cell.get_entity_transform(i, entity)
            mapped_pts = list(map(mapping, pts_ref))
            assert numpy.allclose(mapped_pts, pts_entity)


@pytest.mark.parametrize("split", (AlfeldSplit, UniformSplit))
def test_split_child_to_parent(split, cell):
    split_cell = split(cell)
    mapping = split_cell.get_child_to_parent()
    print("")
    for dim in mapping:
        print(mapping[dim])


@pytest.mark.parametrize("split", (AlfeldSplit, UniformSplit))
def test_macro_quadrature(split, cell):
    split_cell = split(cell)

    degree = 12
    Q_ref = create_quadrature(cell.construct_subelement(1), degree)
    Q = MacroQuadratureRule(split_cell, Q_ref)
    Q.get_points()

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # sdim = cell.get_spatial_dimension()
    # if sdim == 3:
    #     ax = fig.add_subplot(projection='3d')
    # else:
    #     ax = fig.add_subplot()
    # for i, vert in enumerate(split_cell.vertices):
    #     ax.text(*vert, str(i))
    #
    # ax.scatter(*Q.get_points().T)
    # ax.axis("equal")
    # plt.show()
