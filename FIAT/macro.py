import copy

import numpy

from FIAT.reference_element import SimplicialComplex, ufc_simplex


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
    def __init__(self, T):
        self.parent = T
        sdim = T.get_spatial_dimension()
        old_vs = T.get_vertices()

        b = numpy.average(old_vs, axis=0)

        new_verts = old_vs + (tuple(b),)

        new_topology = copy.deepcopy(T.topology)

        new_vert_id = len(T.topology[0])
        new_topology[0] = {i: (i,) for i in range(new_vert_id+1)}
        new_topology[sdim] = {}

        for dim_cur in range(1, sdim + 1):
            start = len(new_topology[dim_cur])
            for eid, vs in T.topology[dim_cur-1].items():
                new_topology[dim_cur][start+eid] = vs + (new_vert_id,)

        super(AlfeldSplit, self).__init__(T.shape, new_verts, new_topology)

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return self.parent.construct_subelement(dimension)


# Does a uniform split
class UniformSplit(SimplicialComplex):
    def __init__(self, T):
        self.parent = T
        sdim = T.get_spatial_dimension()
        old_vs = T.get_vertices()
