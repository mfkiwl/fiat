import math
import numpy
import pytest
from FIAT import DiscontinuousLagrange, Lagrange, Legendre, P0
from FIAT.macro import AlfeldSplit, IsoSplit
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import ufc_simplex
from FIAT.expansions import polynomial_entity_ids, polynomial_cell_node_map
from FIAT.polynomial_set import make_bubbles, PolynomialSet, ONPolynomialSet


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

    # Test that tabulation onto lattice points gives the identity
    sd = ref_el.get_spatial_dimension()
    top = ref_el.get_topology()
    pts = []
    for dim in sorted(top):
        for entity in sorted(top[dim]):
            pts.extend(ref_el.make_points(dim, entity, degree, variant=variant))

    phis = fe.tabulate(2, pts)
    assert numpy.allclose(phis[(0,)*sd], numpy.eye(fe.space_dimension()))

    # Test that we can reproduce the Vandermonde matrix by tabulating the expansion set
    U = poly_set.get_expansion_set()
    V = U.tabulate(degree, pts).T
    assert numpy.allclose(fe.V, V)


@pytest.mark.parametrize("degree", (1, 2,))
def test_lagrange_iso_duals(cell, degree):
    iso = Lagrange(IsoSplit(cell), degree, variant="equispaced")
    P2 = Lagrange(cell, 2*degree, variant="equispaced")

    def get_points(fe):
        points = []
        for node in fe.dual_basis():
            pt, = node.get_point_dict()
            points.append(pt)
        return points

    assert numpy.allclose(get_points(iso), get_points(P2))

    iso_ids = iso.entity_dofs()
    P2_ids = P2.entity_dofs()
    for dim in iso_ids:
        for entity in iso_ids[dim]:
            assert iso_ids[dim][entity] == P2_ids[dim][entity]


@pytest.mark.parametrize("variant", ("gll", "Alfeld,equispaced", "gll,iso"))
def test_is_macro_lagrange(variant):
    is_macro = "alfeld" in variant.lower() or "iso" in variant.lower()

    fe = Lagrange(ufc_simplex(2), 2, variant)
    assert not fe.get_reference_element().is_macrocell()
    assert fe.is_macroelement() == is_macro
    assert fe.get_reference_complex().is_macrocell() == is_macro
    assert fe.get_nodal_basis().get_reference_element().is_macrocell() == is_macro


@pytest.mark.parametrize("variant", ("gl", "Alfeld,equispaced_interior", "chebyshev,iso"))
@pytest.mark.parametrize("degree", (0, 2))
def test_is_macro_discontinuous_lagrange(degree, variant):
    is_macro = "alfeld" in variant.lower() or "iso" in variant.lower()

    fe = DiscontinuousLagrange(ufc_simplex(2), degree, variant)
    if degree == 0 and not is_macro:
        assert isinstance(fe, P0)
    assert not fe.get_reference_element().is_macrocell()
    assert fe.is_macroelement() == is_macro
    assert fe.get_reference_complex().is_macrocell() == is_macro
    assert fe.get_nodal_basis().get_reference_element().is_macrocell() == is_macro


@pytest.mark.parametrize('split', [None, AlfeldSplit])
@pytest.mark.parametrize('codim', range(3))
def test_make_bubbles(cell, split, codim):
    sd = cell.get_spatial_dimension()
    if codim > sd:
        return
    degree = 5
    if split is not None:
        cell = split(cell)
    B = make_bubbles(cell, degree, codim=codim)

    # basic tests
    assert isinstance(B, PolynomialSet)
    assert B.degree == degree
    num_members = B.get_num_members()
    top = cell.get_topology()
    assert num_members == math.comb(degree-1, sd-codim) * len(top[sd - codim])

    # tabulate onto a lattice
    points = []
    for dim in range(sd+1-codim):
        for entity in sorted(top[dim]):
            points.extend(cell.make_points(dim, entity, degree))
    values = B.tabulate(points)[(0,) * sd]

    # test that bubbles vanish on the boundary
    num_pts_on_facet = len(points) - num_members
    facet_values = values[:, :num_pts_on_facet]
    assert numpy.allclose(facet_values, 0, atol=1E-12)

    # test linear independence
    interior_values = values[:, num_pts_on_facet:]
    assert numpy.linalg.matrix_rank(interior_values.T, tol=1E-12) == num_members

    # test trace similarity
    dim = sd - codim
    nfacets = len(top[dim])
    if nfacets > 1 and dim > 0:
        ref_facet = cell.construct_subelement(dim)
        ref_bubbles = make_bubbles(ref_facet, degree)
        ref_points = ref_facet.make_points(dim, 0, degree)
        ref_values = ref_bubbles.tabulate(ref_points)[(0,) * dim]

        bubbles_per_entity = ref_bubbles.get_num_members()
        cur = 0
        for entity in sorted(top[dim]):
            indices = list(range(cur, cur + bubbles_per_entity))
            cur_values = interior_values[numpy.ix_(indices, indices)]
            scale = numpy.max(abs(cur_values)) / numpy.max(abs(ref_values))
            assert numpy.allclose(ref_values * scale, cur_values)
            cur += bubbles_per_entity

    # test that bubbles do not have components in span(P_{degree+2} \ P_{degree})
    Pkdim = math.comb(degree + sd, sd)
    entity_ids = polynomial_entity_ids(cell, degree + 2)
    indices = []
    for entity in top[sd]:
        indices.extend(entity_ids[sd][entity][Pkdim:])
    P = ONPolynomialSet(cell, degree + 2)
    P = P.take(indices)

    Q = create_quadrature(cell, P.degree + B.degree)
    qpts, qwts = Q.get_points(), Q.get_weights()
    P_at_qpts = P.tabulate(qpts)[(0,) * sd]
    B_at_qpts = B.tabulate(qpts)[(0,) * sd]
    assert numpy.allclose(numpy.dot(numpy.multiply(P_at_qpts, qwts), B_at_qpts.T), 0.0)


@pytest.mark.parametrize("degree", (4,))
@pytest.mark.parametrize("variant", (None, "bubble"))
@pytest.mark.parametrize("split", (AlfeldSplit, IsoSplit))
def test_macro_expansion(cell, split, variant, degree):
    ref_complex = split(cell)
    top = ref_complex.get_topology()
    sd = ref_complex.get_spatial_dimension()
    P = ONPolynomialSet(ref_complex, degree, variant=variant, scale=1)

    npoints = degree + sd + 1
    cell_point_map = []
    pts = []
    for cell in top[sd]:
        cur = len(pts)
        pts.extend(ref_complex.make_points(sd, cell, npoints))
        cell_point_map.append(list(range(cur, len(pts))))

    order = 2
    values = P.tabulate(pts, order)
    cell_node_map = polynomial_cell_node_map(ref_complex, degree, continuity=P.expansion_set.continuity)
    for cell in top[sd]:
        sub_el = ref_complex.construct_subelement(sd)
        sub_el.vertices = ref_complex.get_vertices_of_subcomplex(top[sd][cell])
        Pcell = ONPolynomialSet(sub_el, degree, variant=variant, scale=1)

        cell_pts = sub_el.make_points(sd, 0, npoints)
        cell_values = Pcell.tabulate(cell_pts, order)

        ibfs = cell_node_map[cell]
        ipts = cell_point_map[cell]
        indices = numpy.ix_(ibfs, ipts)
        for alpha in values:
            assert numpy.allclose(cell_values[alpha], values[alpha][indices])
