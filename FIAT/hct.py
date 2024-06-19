from FIAT.functional import (PointEvaluation, PointDerivative,
                             IntegralMoment,
                             IntegralMomentOfNormalDerivative)
from FIAT import finite_element, dual_set, macro, polynomial_set
from FIAT.reference_element import TRIANGLE, ufc_simplex
from FIAT.quadrature import FacetQuadratureRule
from FIAT.quadrature_schemes import create_quadrature
from FIAT.jacobi import eval_jacobi


class HCTDualSet(dual_set.DualSet):
    def __init__(self, ref_complex, degree, reduced=False):
        if degree < 3:
            raise ValueError("HCT only defined for degree >= 3")
        ref_el = ref_complex.get_parent()
        if ref_el.get_shape() != TRIANGLE:
            raise ValueError("HCT only defined on triangles")
        top = ref_el.get_topology()
        verts = ref_el.get_vertices()
        sd = ref_el.get_spatial_dimension()
        entity_ids = {dim: {entity: [] for entity in sorted(top[dim])} for dim in sorted(top)}

        # get first order jet at each vertex
        alphas = polynomial_set.mis(sd, 1)
        nodes = []
        for v in sorted(top[0]):
            pt = verts[v]
            cur = len(nodes)
            nodes.append(PointEvaluation(ref_el, pt))
            nodes.extend(PointDerivative(ref_el, pt, alpha) for alpha in alphas)
            entity_ids[0][v].extend(range(cur, len(nodes)))

        rline = ufc_simplex(1)
        k = 2 if reduced else degree - 3
        Q = create_quadrature(rline, degree-1+k)
        qpts = Q.get_points()
        if degree == 3:
            # lowest order case
            x, = qpts.T
            f_at_qpts = eval_jacobi(0, 0, k, 2.0*x - 1)
            for e in sorted(top[1]):
                cur = len(nodes)
                nodes.append(IntegralMomentOfNormalDerivative(ref_el, e, Q, f_at_qpts))
                entity_ids[1][e].extend(range(cur, len(nodes)))
        else:
            Pk = polynomial_set.ONPolynomialSet(rline, k)
            phis = Pk.tabulate(qpts)[(0,)]
            for e in sorted(top[1]):
                Q_mapped = FacetQuadratureRule(ref_el, 1, e, Q)
                cur = len(nodes)
                nodes.extend(IntegralMomentOfNormalDerivative(ref_el, e, Q, phi) for phi in phis)
                nodes.extend(IntegralMoment(ref_el, Q_mapped, phi) for phi in phis[:-1])
                entity_ids[1][e].extend(range(cur, len(nodes)))

            q = degree - 4
            Q = create_quadrature(ref_complex, degree + q)
            Pq = polynomial_set.ONPolynomialSet(ref_el, q)
            phis = Pq.tabulate(Q.get_points())[(0,) * sd]
            cur = len(nodes)
            nodes.extend(IntegralMoment(ref_el, Q, phi) for phi in phis)
            entity_ids[sd][0] = list(range(cur, len(nodes)))

        super(HCTDualSet, self).__init__(nodes, ref_el, entity_ids)


class HsiehCloughTocher(finite_element.CiarletElement):
    """The HCT finite element."""

    def __init__(self, ref_el, degree=3, reduced=False):
        ref_complex = macro.AlfeldSplit(ref_el)
        dual = HCTDualSet(ref_complex, degree, reduced=reduced)

        poly_set = macro.CkPolynomialSet(ref_complex, degree, order=1, variant="bubble")

        print("entity_ids", dual.entity_ids)
        print("num bfs", poly_set.get_num_members())
        print("num dofs", len(dual.nodes))

        super(HsiehCloughTocher, self).__init__(poly_set, dual, degree)


if __name__ == "__main__":

    degree = 4
    HsiehCloughTocher(ufc_simplex(2), degree)
