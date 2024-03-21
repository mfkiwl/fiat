import copy

import numpy

from FIAT import Lagrange
from FIAT.reference_element import SimplicialComplex, ufc_simplex


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


if __name__ == "__main__":
    sdim = 3

    T = ufc_simplex(sdim)

    TA = AlfeldSplit(T)

    TAT = TA.topology

    # degree = sdim+1
    # print(T.vertices)
    # for i in range(4):
    #     print("subcell", i, TA.get_vertices_of_subcomplex(TAT[3][i]))
    #     print("points", TA.make_points(sdim, i, degree))
    # print(T.connectivity)
    # print(TA.connectivity)
