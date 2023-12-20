import numpy as np

from FIAT.quadrature import QuadratureRule, map_quadrature
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import Cell, ufc_simplex


def bary_to_xy(verts, bary, result=None):
    # verts is (sdim + 1) x sdim so verts[i, :] is i:th vertex
    # bary is [npts, sdim + 1]
    # result is [npts, sdim]

    if result is None:
        return bary @ verts
    else:
        np.dot(bary, verts, out=result)
        return result


def xy_to_bary(verts, pts, result=None):
    # verts is (sdim + 1) x sdim so verts[i, :] is i:th vertex
    # result is [npts, sdim]
    # bary is [npts, sdim + 1]
    npts = pts.shape[0]
    sdim = verts.shape[1]

    mat = np.vstack((verts.T, np.ones((1, sdim+1))))

    b = np.vstack((pts.T, np.ones((1, npts))))

    foo = np.linalg.solve(mat, b)

    if result is None:
        return np.copy(foo.T)
    else:
        result[:, :] = foo[:, :].T
    return result


def barycentric_split(ref_el):
    d = ref_el.get_dimension()
    vs = np.asarray(T.get_vertices())
    # existing vertices plus the barycenter
    newvs = np.vstack((vs, np.average(vs, axis=0)))
    # cells comprising each face plus the barycenter
    subcell2vert = np.asarray(
        [[j for j in range(d+1) if j != i] + [d+1] for i in range(d+1)])
    return newvs, subcell2vert


def split_to_cells(ref_el, splitting):
    newvs, subcell2vert = splitting(ref_el)
    top = ref_el.get_topology()
    shape = ref_el.shape
    ncells = subcell2vert.shape[0]
    return [Cell(shape, newvs[subcell2vert[i, :]], top)
            for i in range(ncells)]


class MacroQuadratureRule(QuadratureRule):
    def __init__(self, rule, splitting):
        ref_el = rule.ref_el
        pts = np.asarray(rule.pts)
        wts = np.asarray(rule.wts)
        new_els = split_to_cells(ref_el, splitting)
        new_rules = [map_quadrature(pts, wts, ref_el, new_el)
                     for new_el in new_els]
        super(MacroQuadratureRule, self).__init__(
            ref_el,
            np.vstack([np.asarray(new_rule[0]) for new_rule in new_rules]),
            np.hstack([np.asarray(new_rule[1]) for new_rule in new_rules]))


T = ufc_simplex(2)
Q = create_quadrature(T, 2)
macro_Q = MacroQuadratureRule(Q, barycentric_split)
print(macro_Q.pts)
print(macro_Q.wts)
