from macro import AlfeldSplit
import dual_set
import functional
from finite_element import CiarletElement


# TODO: Need to do constraints with jump functionals,
# null space, etc to get a new polynomial set
def C1subspace(simplicial_complex, deg):
    pass


class HCTDual(dual_set.DualSet):

    def __init__(self, simplical_complex):
        entity_ids = {
            0: {0: [0, 1, 2],
                1: [3, 4, 5],
                2: [6, 7, 8]},
            1: {0: [9], 1: [10], 2: [11]},
            2: {0: []}}
        T = simplicial_complex.parent
        verts = T.get_vertices()
        for vid in sorted(T[0]):
            v = verts[vid]
            nodes.extend(
                [functional.PointEvaluation(v),
                 functional.PointDerivative(v, (1, 0)),
                 functional.PointDerivative(v, (0, 1))])
        # FIXME: Should be integral moments 
        # for better transformation theory
        for eid in sorted(T[1]):
            pt = T.make_points(1, eid, 2)[0]
            nodes.append(
                functional.PointNormalDerivative(T, eid, pt))
        super(HCTDual, self).__init__(nodes, T, entity_ids)
                 
    

class HCT(CiarletElement):

    def __init__(self, ref_el):
        TA = AlfeldSplit(ref_el)

        P = C1Subspace(TA, 3)

        D = HCTDualSet(TA)

        super(HCT, self).__init__(P, D, 3)
        
        
