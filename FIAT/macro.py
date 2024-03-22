import copy
import numpy
from FIAT.reference_element import SimplicialComplex
from FIAT.finite_element import FiniteElement
from FIAT.quadrature import QuadratureRule, FacetQuadratureRule


def bary_to_xy(verts, bary, result=None):
    # verts is (sdim + 1) x sdim so verts[i, :] is i:th vertex
    # bary is [npts, sdim + 1]
    # result is [npts, sdim]

    if result is None:
        return bary @ verts
    else:
        numpy.dot(bary, verts, out=result)
        return result


def xy_to_bary(verts, pts, result=None):
    # verts is (sdim + 1) x sdim so verts[i, :] is i:th vertex
    # result is [npts, sdim]
    # bary is [npts, sdim + 1]
    npts = pts.shape[0]
    sdim = verts.shape[1]

    mat = numpy.vstack((verts.T, numpy.ones((1, sdim+1))))

    b = numpy.vstack((pts.T, numpy.ones((1, npts))))

    foo = numpy.linalg.solve(mat, b)

    if result is None:
        return numpy.copy(foo.T)
    else:
        result[:, :] = foo[:, :].T
    return result


class AlfeldSplit(SimplicialComplex):

    def __init__(self, ref_el):
        self.parent = ref_el
        sd = ref_el.get_spatial_dimension()
        old_verts = ref_el.get_vertices()

        b = numpy.average(old_verts, axis=0)
        new_verts = old_verts + (tuple(b),)

        new_topology = copy.deepcopy(ref_el.topology)

        new_vert_id = len(ref_el.topology[0])
        new_topology[0] = {i: (i,) for i in range(new_vert_id + 1)}
        new_topology[sd] = {}

        for dim in range(1, sd + 1):
            offset = len(new_topology[dim])
            for entity, ids in ref_el.topology[dim-1].items():
                new_topology[dim][offset+entity] = ids + (new_vert_id,)

        super(AlfeldSplit, self).__init__(ref_el.shape, new_verts, new_topology)

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return self.parent.construct_subelement(dimension)


class UniformSplit(SimplicialComplex):

    def __init__(self, ref_el):
        self.parent = ref_el
        sd = ref_el.get_spatial_dimension()
        old_verts = ref_el.get_vertices()

        new_verts = old_verts + tuple(tuple(numpy.average(old_verts[list(ids)], axis=0))
                                      for ids in ref_el.topology[1].values())

        new_topology = {}
        new_topology[0] = {i: (i,) for i in range(len(new_verts))}
        new_topology[1] = {}

        # Split each edge
        offset = len(old_verts)
        for entity, verts in ref_el.topology[1].items():
            midpoint = offset + entity
            new_topology[1][2*entity] = (verts[0], midpoint)
            new_topology[1][2*entity+1] = (verts[1], midpoint)

        # Add edges connecting midpoints
        num_old_edges = len(ref_el.topology[1])
        cur = len(new_topology[1])
        for j in range(num_old_edges):
            for i in range(j+1, num_old_edges):
                new_topology[1][cur] = (offset+j, offset+i)
                cur = cur + 1

        # TODO add higher dimensional entites
        for dim in range(2, sd+1):
            new_topology[dim] = {}


        super(UniformSplit, self).__init__(ref_el.shape, new_verts, new_topology)


    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return self.parent.construct_subelement(dimension)


class MacroElement(FiniteElement):
    """
    A macro element built from a base finite element on a split of the reference cell
    """

    def __init__(self, element, split):
        ref_el = element.get_reference_element()
        dual = None
        order = element.get_order()
        formdegree = element.get_formdegree()
        mapping = element._mapping
        self.element = element
        self.cell_complex = split(ref_el)
        super(MacroElement, self).__init__(ref_el, dual, order, formdegree=formdegree, mapping=mapping)

    def tabulate(self, order, points, entity=None):
        raise NotImplementedError("Wait for it")
        # tabulate the reference element on each sub-cell and scatter with the local to global mapping


class MacroQuadratureRule(QuadratureRule):

    def __init__(self, cell_complex, Q_ref):
        pts = []
        wts = []
        sd = cell_complex.get_spatial_dimension()
        ref_el = cell_complex.construct_subelement(sd)
        t = cell_complex.get_topology()
        dim = Q_ref.ref_el.get_spatial_dimension()
        for entity in t[dim]:
            Q_cur = FacetQuadratureRule(cell_complex, dim, entity, Q_ref)
            pts.extend(Q_cur.pts)
            wts.extend(Q_cur.wts)

        pts = tuple(pts)
        wts = tuple(wts)
        super(MacroQuadratureRule, self).__init__(ref_el, pts, wts)
