import numpy
import pytest
from FIAT.reference_element import ufc_simplex
from FIAT.macro import AlfeldSplit, IsoSplit
from FIAT.quadrature_schemes import create_quadrature
from FIAT.lagrange import Lagrange
from FIAT.hierarchical import Legendre


@pytest.fixture(params=("I", "T", "S"))
def cell(request):
    dim = {"I": 1, "T": 2, "S": 3}[request.param]
    return ufc_simplex(dim)


@pytest.mark.parametrize("split", (AlfeldSplit, IsoSplit))
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
@pytest.mark.parametrize("split", (AlfeldSplit, IsoSplit))
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


def test_split_child_to_parent(cell):
    split_cell = IsoSplit(cell)

    dim = cell.get_spatial_dimension()
    degree = 2 if dim == 3 else 4
    parent_degree = 2*degree

    top = cell.get_topology()
    parent_pts = {dim: {} for dim in top}
    for dim in top:
        for entity in top[dim]:
            parent_pts[dim][entity] = cell.make_points(dim, entity, parent_degree)

    top = split_cell.get_topology()
    child_pts = {dim: {} for dim in top}
    for dim in top:
        for entity in top[dim]:
            child_pts[dim][entity] = split_cell.make_points(dim, entity, degree)

    child_to_parent = split_cell.get_child_to_parent()
    for dim in top:
        for entity in top[dim]:
            parent_dim, parent_entity = child_to_parent[dim][entity]
            assert set(child_pts[dim][entity]) <= set(parent_pts[parent_dim][parent_entity])


@pytest.mark.parametrize("split", (AlfeldSplit, IsoSplit))
def test_macro_quadrature(split, cell):
    ref_el = split(cell)
    sd = ref_el.get_spatial_dimension()

    degree = 3
    Q = create_quadrature(ref_el, 2*degree)
    pts, wts = Q.get_points(), Q.get_weights()

    # Test that the mass matrix for an orthogonal basis is diagonal
    fe = Legendre(ref_el, degree)
    phis = fe.tabulate(0, pts)[(0,)*sd]
    M = numpy.dot(numpy.multiply(phis, wts), phis.T)
    M = M - numpy.diag(M.diagonal())
    assert numpy.allclose(M, 0)


@pytest.mark.parametrize("degree", range(1, 5))
@pytest.mark.parametrize("variant", ("equispaced", "gll"))
@pytest.mark.parametrize("split", (AlfeldSplit, IsoSplit))
def test_macro_lagrange(variant, degree, split, cell):
    ref_el = split(cell)

    fe = Lagrange(ref_el, degree, variant=variant)
    poly_set = fe.get_nodal_basis()

    # Test that the polynomial set is defined on the split and not on the parent cell
    assert poly_set.get_reference_element() is ref_el

    # Test that the finite element is defined on the parent cell and not on the split
    assert fe.get_reference_element() is cell

    # Test that parent entities are the ones exposed
    entity_ids = fe.entity_dofs()
    parent_top = ref_el.get_parent().get_topology()
    for dim in parent_top:
        assert len(entity_ids[dim]) == len(parent_top[dim])

    # TODO more thorough checks on entity_ids

    # Test that tabulation onto lattice points gives the identity
    sd = ref_el.get_spatial_dimension()
    top = ref_el.get_topology()
    pts = []
    for dim in sorted(top):
        for entity in sorted(top[dim]):
            pts.extend(ref_el.make_points(dim, entity, degree, variant=variant))

    phis = fe.tabulate(0, pts)[(0,)*sd]
    assert numpy.allclose(phis, numpy.eye(fe.space_dimension()))

    # Test that we can reproduce the Vandermonde matrix by tabulating the expansion set
    U = poly_set.get_expansion_set()
    V = U.tabulate(degree, pts).T
    assert numpy.allclose(fe.V, V)


@pytest.mark.parametrize("variant", ("gll", "Alfeld,equispaced", "gll,iso"))
def test_is_macro(variant):
    is_macro = "alfeld" in variant.lower() or "iso" in variant.lower()

    fe = Lagrange(ufc_simplex(2), 2, variant)
    assert not fe.get_reference_element().is_macrocell()
    assert fe.is_macroelement() == is_macro
    assert fe.get_reference_complex().is_macrocell() == is_macro
    assert fe.get_nodal_basis().get_reference_element().is_macrocell() == is_macro
