import copy

import numpy

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

    def construct_subelement(self, dimension):
        """Constructs the reference element of a cell subentity
        specified by subelement dimension.

        :arg dimension: subentity dimension (integer)
        """
        return self.parent.construct_subelement(dimension)


if __name__ == "__main__":
    sdim = 2

    T = ufc_simplex(sdim)

    TA = AlfeldSplit(T)

    TAT = TA.topology

    for i in range(1, sdim+1):
        TX = TA.construct_subelement(i)
        b = numpy.average(TX.get_vertices(), axis=0)
        for entity in TAT[i].keys():
            mapped_bary = TA.get_entity_transform(i, entity)(b)
            computed_bary = numpy.average(TA.get_vertices_of_subcomplex(TAT[i][entity]), axis=0)
            assert numpy.allclose(mapped_bary, computed_bary)
