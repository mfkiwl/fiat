import copy
import numpy
from FIAT.reference_element import make_lattice, lattice_iter, SimplicialComplex
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


class SplitSimplicialComplex(SimplicialComplex):
    """Abstract class to implement a split on a Simplex
    """

    def __init__(self, ref_el):
        self.parent = ref_el
        vertices, topology = self.split_topology(ref_el)
        super(SplitSimplicialComplex, self).__init__(ref_el.shape, vertices, topology)

    def split_topology(self, ref_el):
        raise NotImplementedError

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return self.parent.construct_subelement(dimension)


class AlfeldSplit(SplitSimplicialComplex):

    def split_topology(self, ref_el):
        sd = ref_el.get_spatial_dimension()
        new_topology = copy.deepcopy(ref_el.topology)
        new_topology[sd] = {}

        barycenter = ref_el.make_points(sd, 0, sd+1)
        new_verts = ref_el.vertices + tuple(barycenter)
        new_vert_id = len(ref_el.vertices)

        new_topology[0][new_vert_id] = (new_vert_id,)
        for dim in range(1, sd + 1):
            offset = len(new_topology[dim])
            for entity, ids in ref_el.topology[dim-1].items():
                new_topology[dim][offset+entity] = ids + (new_vert_id,)
        return new_verts, new_topology


class UniformSplit(SplitSimplicialComplex):

    def split_topology(self, ref_el):
        sd = ref_el.get_spatial_dimension()
        old_verts = ref_el.get_vertices()
        new_verts = make_lattice(old_verts, 2)

        new_topology = {}
        new_topology[0] = {i: (i,) for i in range(len(new_verts))}
        new_topology[1] = {}

        # Loop through vertex pairs
        # Edges are oriented from low vertex id to high vertex id to avoid duplicates
        # Place a new edge when the two lattice multiindices are at Manhattan distance < 3,
        # this connects the midpoints of edges within a face
        # Only include diagonal edges that are parallel to the simplex edges,
        # we take the diagonal that goes through vertices of the same "polynomial degree"
        cur = 0
        distance = lambda x, y: sum(abs(b-a) for a, b in zip(x, y))
        for j, v1 in enumerate(lattice_iter(0, 3, sd)):
            for i, v0 in enumerate(lattice_iter(0, 3, sd)):
                if i < j and distance(v0, v1) < 3 and sum(v1) - sum(v0) <= 1:
                    new_topology[1][cur] = (i, j)
                    cur = cur + 1
        if sd == 3:
            # Cut the octahedron
            # FIXME do this more generalically
            new_topology[1][cur] = (1, 8)

        # Get an adjacency list for each vertex
        adjacency = {}
        for v in new_topology[0]:
            cur_neigh = []
            for entity in new_topology[1]:
                if v in new_topology[1][entity]:
                    cur_neigh.extend(new_topology[1][entity])
            adjacency[v] = set(cur_neigh)

        # Complete the higher dimensional facets by appending a vertex
        # adjacent to the vertices of co-dimension 1 facets
        for dim in range(2, sd+1):
            entities = []
            for entity in new_topology[dim-1]:
                facet = new_topology[dim-1][entity]
                for v in range(min(facet)):
                    if set(facet) < adjacency[v]:
                        entities.append((v,) + facet)

            new_topology[dim] = dict(enumerate(entities))
        return new_verts, new_topology


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
