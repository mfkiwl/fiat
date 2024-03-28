import copy
from itertools import chain

import numpy

from FIAT.quadrature import FacetQuadratureRule, QuadratureRule
from FIAT.reference_element import (SimplicialComplex, lattice_iter,
                                    make_lattice)


def bary_to_xy(verts, bary, result=None):
    """Maps barycentric coordinates to physical points.

    :arg verts: A tuple of points.
    :arg bary: A row-stacked numpy array of barycentric coordinates.
    :arg result: A row-stacked numpy array of physical points.
    :returns: result
    """
    return numpy.dot(bary, verts, out=result)


def xy_to_bary(verts, pts, result=None):
    """Maps physical points to barycentric coordinates.

    :arg verts: A tuple of points.
    :arg ots: A row-stacked numpy array of physical points.
    :arg result: A row-stacked numpy array of barycentric coordinates.
    :returns: result
    """
    verts = numpy.asarray(verts)
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
    """Returns the support of a facet.

    :arg facet_coords: An iterable of tuples (barycentric coordinates) describing the facet.
    :returns: A tuple of vertex ids where some coordinate is nonzero.
    """
    return tuple(sorted(set(i for x in facet_coords for (i, xi) in enumerate(x) if abs(xi) > tol)))


def invert_cell_topology(T):
    """Returns a dict of dicts mapping dimension x vertices to entity id."""
    return {dim: {T[dim][entity]: entity for entity in T[dim]} for dim in T}


class SplitSimplicialComplex(SimplicialComplex):
    """Abstract class to implement a split on a Simplex.

    :arg parent: The parent Simplex to split.
    :arg vertices: The vertices of the simplicial complex.
    :arg topology: The topology of the simplicial complex.
    """
    def __init__(self, parent, vertices, topology):
        self._parent = parent

        bary = xy_to_bary(numpy.asarray(parent.get_vertices()), numpy.asarray(vertices))
        parent_top = parent.get_topology()
        parent_inv_top = invert_cell_topology(parent_top)

        # dict mapping child facets to their parent facet
        child_to_parent = {}
        # dict mapping parent facets to their children
        parent_to_children = {dim: {entity: [] for entity in parent_top[dim]} for dim in parent_top}
        for dim in topology:
            child_to_parent[dim] = {}
            for entity in topology[dim]:
                facet_ids = topology[dim][entity]
                facet_coords = bary[list(facet_ids), :]
                parent_verts = facet_support(facet_coords)
                parent_dim = len(parent_verts) - 1
                parent_entity = parent_inv_top[parent_dim][parent_verts]
                child_to_parent[dim][entity] = (parent_dim, parent_entity)
                parent_to_children[parent_dim][parent_entity].append((dim, entity))

        self._child_to_parent = child_to_parent
        self._parent_to_children = parent_to_children

        sd = parent.get_spatial_dimension()
        inv_top = invert_cell_topology(topology)

        # dict mapping cells to their boundary facets for each dimension,
        # while respecting the ordering on the parent simplex
        connectivity = {cell: {dim: [] for dim in topology} for cell in topology[sd]}
        for cell in topology[sd]:
            cell_verts = topology[sd][cell]
            for dim in parent_top:
                for entity in parent_top[dim]:
                    ref_verts = parent_top[dim][entity]
                    global_verts = tuple(cell_verts[v] for v in ref_verts)
                    connectivity[cell][dim].append(inv_top[dim][global_verts])
        self._cell_connectivity = connectivity

        super(SplitSimplicialComplex, self).__init__(parent.shape, vertices, topology)

    def get_child_to_parent(self):
        """Maps split complex facet tuple to its parent entity tuple."""
        return self._child_to_parent

    def get_parent_to_children(self):
        """Maps parent facet tuple to a list of tuples of entites in the split complex."""
        return self._parent_to_children

    def get_cell_connectivity(self):
        """Connectitivity from cell in a complex to global facet ids and
        respects the entity numbering on the reference cell.

        N.B. cell_connectivity[cell][dim] has the same contents as
        self.connectivity[(sd, dim)][cell], except those are sorted.
        """
        return self._cell_connectivity

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return self._parent.construct_subelement(dimension)

    def is_macrocell(self):
        return True

    def get_parent(self):
        return self._parent


class AlfeldSplit(SplitSimplicialComplex):
    """Splits a simplex into the simplicial complex obtained by
    connecting vertices to barycenter.
    """
    def __init__(self, ref_el):
        sd = ref_el.get_spatial_dimension()
        top = ref_el.get_topology()
        # Keep old facets, respecting the old numbering
        new_topology = copy.deepcopy(top)
        # Discard the cell interior
        new_topology[sd] = {}

        # Append the barycenter as the new vertex
        barycenter = ref_el.make_points(sd, 0, sd+1)
        old_verts = ref_el.get_vertices()
        new_verts = old_verts + tuple(barycenter)
        new_vert_id = len(old_verts)
        new_topology[0][new_vert_id] = (new_vert_id,)

        # Append new facets by adding the barycenter to old facets
        for dim in range(1, sd + 1):
            offset = len(new_topology[dim])
            for entity, ids in top[dim-1].items():
                new_topology[dim][offset+entity] = ids + (new_vert_id,)
        super(AlfeldSplit, self).__init__(ref_el, new_verts, new_topology)


class IsoSplit(SplitSimplicialComplex):
    """Splits simplex into the simplicial complex obtained by
    connecting points on a regular lattice.

    :arg ref_el: The parent Simplex to split.
    :kwarg depth: The number of subdivisions along each edge of the simplex.
    :kwarg variant: The point distribution variant.
    """
    def __init__(self, ref_el, depth=2, variant=None):
        sd = ref_el.get_spatial_dimension()
        old_verts = ref_el.get_vertices()
        new_verts = make_lattice(old_verts, depth, variant=variant)

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
            assert depth == 2
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
        super(IsoSplit, self).__init__(ref_el, new_verts, new_topology)


class MacroQuadratureRule(QuadratureRule):
    """Composite quadrature rule on parent facets that respects the splitting.

    :arg ref_el: A simplicial complex.
    :arg Q_ref: A QuadratureRule on the reference simplex.
    :args parent_facets: An iterable of facets of the same dimension as Q_ref,
                         defaults to all facets.

    :returns: A quadrature rule on the sub entities of the simplicial complex.
    """
    def __init__(self, ref_el, Q_ref, parent_facets=None):
        parent_dim = Q_ref.ref_el.get_spatial_dimension()
        if parent_facets is not None:
            parent_cell = ref_el.parent
            parent_to_children = parent_cell.get_parent_to_children()
            facets = []
            for parent_entity in parent_facets:
                children = parent_to_children[parent_dim][parent_entity]
                facets.extend(entity for dim, entity in children if dim == parent_dim)
        else:
            top = ref_el.get_topology()
            facets = top[parent_dim]

        pts = []
        wts = []
        for entity in facets:
            Q_cur = FacetQuadratureRule(ref_el, parent_dim, entity, Q_ref)
            pts.extend(Q_cur.pts)
            wts.extend(Q_cur.wts)
        pts = tuple(pts)
        wts = tuple(wts)
        super(MacroQuadratureRule, self).__init__(ref_el, pts, wts)
