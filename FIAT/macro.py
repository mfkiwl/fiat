import copy
import numpy
from itertools import chain
from FIAT.reference_element import make_lattice, lattice_iter, SimplicialComplex
from FIAT.quadrature import QuadratureRule, FacetQuadratureRule


def bary_to_xy(verts, bary, result=None):
    # verts is (sdim + 1) x sdim so verts[i, :] is i:th vertex
    # bary is [npts, sdim + 1]
    # result is [npts, sdim]
    return numpy.dot(bary, verts, out=result)


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


def facet_support(facet_coords, tol=1.e-12):
    # facet_coords is an iterable of tuples (barycentric coordinates)
    # return vertex ids where some x is nonzero
    return tuple(sorted(set(i for x in facet_coords for (i, xi) in enumerate(x) if abs(xi) > tol)))


def invert_cell_topology(T):
    return {dim: {T[dim][entity]: entity for entity in T[dim]} for dim in T}


class SplitSimplicialComplex(SimplicialComplex):
    """Abstract class to implement a split on a Simplex
    """

    def __init__(self, ref_el, splits=1):
        self.parent = ref_el
        vertices, topology = self.split_topology(ref_el, splits=splits)
        super(SplitSimplicialComplex, self).__init__(ref_el.shape, vertices, topology)

    def split_topology(self, ref_el):
        raise NotImplementedError

    def get_child_to_parent(self):
        bary = xy_to_bary(numpy.asarray(self.parent.get_vertices()),
                          numpy.asarray(self.get_vertices()))
        top = self.get_topology()
        parent_inv_top = invert_cell_topology(self.parent.get_topology())
        child_to_parent = {}
        for dim in top:
            child_to_parent[dim] = {}
            for entity in top[dim]:
                facet_ids = top[dim][entity]
                facet_coords = bary[list(facet_ids), :]
                parent_verts = facet_support(facet_coords)
                parent_dim = len(parent_verts) - 1
                parent_entity = parent_inv_top[parent_dim][parent_verts]
                child_to_parent[dim][entity] = (parent_dim, parent_entity)
        return child_to_parent

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return self.parent.construct_subelement(dimension)

    def get_entity_transform(self, dim, entity):
        # This is to trick FiniteElement.tabulate
        return self.parent.get_entity_transform(dim, entity)


class AlfeldSplit(SplitSimplicialComplex):

    def split_topology(self, ref_el, splits=1):
        assert splits == 1
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        new_topology = copy.deepcopy(top)
        new_topology[sd] = {}

        barycenter = ref_el.make_points(sd, 0, sd+1)
        old_verts = ref_el.get_vertices()
        new_verts = old_verts + tuple(barycenter)
        new_vert_id = len(old_verts)

        new_topology[0][new_vert_id] = (new_vert_id,)
        for dim in range(1, sd + 1):
            offset = len(new_topology[dim])
            for entity, ids in top[dim-1].items():
                new_topology[dim][offset+entity] = ids + (new_vert_id,)
        return new_verts, new_topology


class UniformSplit(SplitSimplicialComplex):

    def split_topology(self, ref_el, splits=1):
        depth = splits + 1
        sd = ref_el.get_spatial_dimension()
        old_verts = ref_el.get_vertices()
        new_verts = make_lattice(old_verts, depth)

        new_topology = {}
        new_topology[0] = {i: (i,) for i in range(len(new_verts))}
        new_topology[1] = {}

        # Loop through vertex pairs
        # Edges are oriented from low vertex id to high vertex id to avoid duplicates
        # Place a new edge when the two lattice multiindices are at Manhattan distance < 3,
        # this connects the midpoints of edges within a face
        # Only include diagonal edges that are parallel to the simplex edges,
        # we take the diagonal that goes through vertices at the same depth
        cur = 0
        distance = lambda x, y: sum(abs(b-a) for a, b in zip(x, y))
        for j, v1 in enumerate(lattice_iter(0, depth+1, sd)):
            for i, v0 in enumerate(lattice_iter(0, depth+1, sd)):
                if i < j and distance(v0, v1) < 3 and sum(v1) - sum(v0) <= 1:
                    new_topology[1][cur] = (i, j)
                    cur = cur + 1
        if sd == 3:
            # Cut the octahedron
            # FIXME do this more generically
            assert splits == 1
            new_topology[1][cur] = (1, 8)

        # Get an adjacency list for each vertex
        edges = new_topology[1].values()
        adjacency = {v: set(chain.from_iterable(verts for verts in edges if v in verts))
                     for v in new_topology[0]}

        # Complete the higher dimensional facets by appending a vertex
        # adjacent to the vertices of codimension 1 facets
        for dim in range(2, sd+1):
            entities = []
            for entity in new_topology[dim-1]:
                facet = new_topology[dim-1][entity]
                for v in range(min(facet)):
                    if set(facet) < adjacency[v]:
                        entities.append((v,) + facet)
            new_topology[dim] = dict(enumerate(entities))
        return new_verts, new_topology


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


class MacroElement():

    def __init__(self, finite_element, splitting):
        ref_el = finite_element.ref_el
        sdim = ref_el.get_spatial_dimension()
        sc = splitting(ref_el)

        ref_eids = finite_element.entity_dofs()
        ndofs_per_dim = {d: len(ref_eids[d][0]) for d in ref_eids}

        sc_facet_to_dofs = {}

        sc_t = sc.topology

        # Enumerate dofs, attach to complex facets
        dof_cur = 0
        for dim in sc_t:
            for facet_id in sc_t[dim]:
                facet = sc_t[dim][facet_id]
                ndof_cur = ndofs_per_dim[dim]
                sc_facet_to_dofs[facet] = (dof_cur, dof_cur + ndof_cur)
                dof_cur += ndof_cur

        # cell_node_map
        num_cells = len(sc_t[dim])
        dofs_per_cell = finite_element.space_dimension()

        # This is used in evaluation.
        cell_node_map = numpy.zeros((num_cells, dofs_per_cell), int)
        conn = sc.connectivity
        for cid, in conn[(sdim, sdim)]:
            for dim in range(sdim+1):
                for ref_fid, fid in enumerate(conn[(sdim, dim)][cid]):
                    facet = sc_t[dim][fid]
                    dofs = list(range(*sc_facet_to_dofs[facet]))
                    ref_dofs = ref_eids[dim][ref_fid]
                    cell_node_map[cid, ref_dofs] = dofs

        # collect dofs from complex onto facets of main cell
        # This needs to go into a dual something or other somewhere
        c2p = sc.get_child_to_parent()
        ref_t = ref_el.topology
        entity_ids = {d: {f: [] for f in ref_t[d]} for d in ref_t}
        for dim in c2p:
            for fid in c2p[dim]:
                (parent_dim, parent_id) = c2p[dim][fid]
                dofs_cur = list(range(*sc_facet_to_dofs[sc_t[dim][fid]]))
                entity_ids[parent_dim][parent_id].extend(dofs_cur)

        print(f"Cell node map:\n{cell_node_map}")
        print(f"DOFs per facet in reference cell:\n{entity_ids}")


if __name__ == "__main__":
    from reference_element import ufc_simplex
    from lagrange import Lagrange
    K = ufc_simplex(2)
    L = Lagrange(K, 3)
    ML = MacroElement(L, AlfeldSplit)
