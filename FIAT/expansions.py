# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Principal orthogonal expansion functions as defined by Karniadakis
and Sherwin.  These are parametrized over a reference element so as
to allow users to get coordinates that they want."""

import numpy
import math
import sympy
from FIAT import reference_element
from FIAT import jacobi
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLegendreQuadratureLineRule
from FIAT.recursive_points import RecursivePointSet


def morton_index2(i, j=0):
    return (i + j) * (i + j + 1) // 2 + j


def morton_index3(p, q=0, r=0):
    return (p + q + r)*(p + q + r + 1)*(p + q + r + 2)//6 + (q + r)*(q + r + 1)//2 + r


def jrc(a, b, n):
    an = (2*n+1+a+b)*(2*n+2+a+b) / (2*(n+1)*(n+1+a+b))
    bn = (a*a-b*b) * (2*n+1+a+b) / (2*(n+1)*(2*n+a+b)*(n+1+a+b))
    cn = (n+a)*(n+b)*(2*n+2+a+b) / ((n+1)*(n+1+a+b)*(2*n+a+b))
    return an, bn, cn


def recurrence(dim, n, factors, phi, dfactors=None, dphi=None):
    skip_derivs = dphi is None
    if dim == 0:
        return
    elif dim == 1:
        idx = lambda p: p
    elif dim == 2:
        idx = morton_index2
    elif dim == 3:
        idx = morton_index3
    else:
        raise ValueError("Invalid number of spatial dimensions")

    f1, f2, f3, f4 = factors
    f5 = f4 ** 2
    if dfactors is not None:
        df1, df2, df3, df4 = dfactors
        df5 = 2 * f4 * df4

    # p = 1
    phi[idx(1)] = f1
    if not skip_derivs:
        dphi[idx(1)] = df1

    # general p by recurrence
    for p in range(1, n):
        icur = idx(p)
        inext = idx(p + 1)
        iprev = idx(p - 1)
        a = (2. * p + 1.) / (1. + p)
        b = p / (1. + p)
        phi[inext] = a * f1 * phi[icur] - b * f2 * phi[iprev]
        if skip_derivs:
            continue
        dphi[inext] = (a * f1 * dphi[icur] - b * f2 * dphi[iprev] +
                       a * phi[icur] * df1 - b * phi[iprev] * df2)
    if dim < 2:
        return

    # q = 1
    for p in range(n):
        icur = idx(p, 0)
        inext = idx(p, 1)
        g = (p + 1.5) * f3 - f4
        phi[inext] = g * phi[icur]
        if skip_derivs:
            continue
        dg = (p + 1.5) * df3 - df4
        dphi[inext] = g * dphi[icur] + phi[icur] * dg

    # general q by recurrence
    for p in range(n - 1):
        for q in range(1, n - p):
            icur = idx(p, q)
            inext = idx(p, q + 1)
            iprev = idx(p, q - 1)
            aq, bq, cq = jrc(2 * p + 1, 0, q)
            g = aq * f3 + (bq - aq) * f4
            h = cq * f5
            phi[inext] = g * phi[icur] - h * phi[iprev]
            if skip_derivs:
                continue
            dg = aq * df3 + (bq - aq) * df4
            dh = cq * df5
            dphi[inext] = g * dphi[icur] + phi[icur] * dg - h * dphi[iprev] - phi[iprev] * dh
    if dim < 3:
        return

    z = 1 - 2 * f4
    if dfactors:
        dz = -2 * df4
    # r = 1
    for p in range(n):
        for q in range(n - p):
            icur = idx(p, q, 0)
            inext = idx(p, q, 1)
            a = 2.0 + p + q
            b = 1.0 + p + q
            g = a * z + b
            phi[inext] = g * phi[icur]
            if skip_derivs:
                continue
            dg = a * dz
            dphi[inext] = g * dphi[icur] + phi[icur] * dg

    # general r by recurrence
    for p in range(n - 1):
        for q in range(0, n - p - 1):
            for r in range(1, n - p - q):
                icur = idx(p, q, r)
                inext = idx(p, q, r + 1)
                iprev = idx(p, q, r - 1)
                ar, br, cr = jrc(2 * p + 2 * q + 2, 0, r)
                phi[inext] = (ar * z + br) * phi[icur] - cr * phi[iprev]
                if skip_derivs:
                    continue
                dphi[inext] = (ar * z + br) * dphi[icur] + ar * phi[icur] * dz - cr * dphi[iprev]


def _tabulate_dpts(tabulator, D, n, order, pts):
    X = sympy.DeferredVector('x')

    def form_derivative(F):
        '''Forms the derivative recursively, i.e.,
        F               -> [F_x, F_y, F_z],
        [F_x, F_y, F_z] -> [[F_xx, F_xy, F_xz],
                            [F_yx, F_yy, F_yz],
                            [F_zx, F_zy, F_zz]]
        and so forth.
        '''
        out = []
        try:
            out = [sympy.diff(F, X[j]) for j in range(D)]
        except (AttributeError, ValueError):
            # Intercept errors like
            #  AttributeError: 'list' object has no attribute
            #  'free_symbols'
            for f in F:
                out.append(form_derivative(f))
        return out

    def numpy_lambdify(X, F):
        '''Unfortunately, SymPy's own lambdify() doesn't work well with
        NumPy in that simple functions like
            lambda x: 1.0,
        when evaluated with NumPy arrays, return just "1.0" instead of
        an array of 1s with the same shape as x. This function does that.
        '''
        try:
            lambda_x = [numpy_lambdify(X, f) for f in F]
        except TypeError:  # 'function' object is not iterable
            # SymPy's lambdify also works on functions that return arrays.
            # However, use it componentwise here so we can add 0*x to each
            # component individually. This is necessary to maintain shapes
            # if evaluated with NumPy arrays.
            lmbd_tmp = sympy.lambdify(X, F)
            lambda_x = lambda x: lmbd_tmp(x) + 0 * x[0]
        return lambda_x

    def evaluate_lambda(lmbd, x):
        '''Properly evaluate lambda expressions recursively for iterables.
        '''
        try:
            values = [evaluate_lambda(l, x) for l in lmbd]
        except TypeError:  # 'function' object is not iterable
            values = lmbd(x)
        return values

    # Tabulate symbolically
    symbolic_tab = tabulator(n, X)
    # Make sure that the entries of symbolic_tab are lists so we can
    # append derivatives
    symbolic_tab = [[phi] for phi in symbolic_tab]
    #
    data = (order + 1) * [None]
    for r in range(order + 1):
        shape = [len(symbolic_tab), len(pts)] + r * [D]
        data[r] = numpy.empty(shape)
        for i, phi in enumerate(symbolic_tab):
            # Evaluate the function numerically using lambda expressions
            deriv_lambda = numpy_lambdify(X, phi[r])
            data[r][i] = \
                numpy.array(evaluate_lambda(deriv_lambda, pts.T)).T
            # Symbolically compute the next derivative.
            # This actually happens once too many here; never mind for
            # now.
            phi.append(form_derivative(phi[-1]))
    return data


def xi_triangle(eta):
    """Maps from [-1,1]^2 to the (-1,1) reference triangle."""
    eta1, eta2 = eta
    xi1 = 0.5 * (1.0 + eta1) * (1.0 - eta2) - 1.0
    xi2 = eta2
    return (xi1, xi2)


def xi_tetrahedron(eta):
    """Maps from [-1,1]^3 to the -1/1 reference tetrahedron."""
    eta1, eta2, eta3 = eta
    xi1 = 0.25 * (1. + eta1) * (1. - eta2) * (1. - eta3) - 1.
    xi2 = 0.5 * (1. + eta2) * (1. - eta3) - 1.
    xi3 = eta3
    return xi1, xi2, xi3


class ExpansionSet(object):
    point_set = RecursivePointSet(lambda n: GaussLegendreQuadratureLineRule(UFCInterval(), n + 1).get_points())

    def __new__(cls, ref_el, *args, **kwargs):
        """Returns an ExpansionSet instance appopriate for the given
        reference element."""
        if cls is not ExpansionSet:
            return super(ExpansionSet, cls).__new__(cls)
        if ref_el.get_shape() == reference_element.POINT:
            return PointExpansionSet(ref_el)
        elif ref_el.get_shape() == reference_element.LINE:
            return LineExpansionSet(ref_el)
        elif ref_el.get_shape() == reference_element.TRIANGLE:
            return TriangleExpansionSet(ref_el)
        elif ref_el.get_shape() == reference_element.TETRAHEDRON:
            return TetrahedronExpansionSet(ref_el)
        else:
            raise ValueError("Invalid reference element type.")

    def __init__(self, ref_el):
        self.ref_el = ref_el
        dim = ref_el.get_spatial_dimension()
        self.base_ref_el = reference_element.default_simplex(dim)
        v1 = ref_el.get_vertices()
        v2 = self.base_ref_el.get_vertices()
        self.A, self.b = reference_element.make_affine_mapping(v1, v2)
        self.mapping = lambda x: numpy.dot(self.A, x) + self.b
        self.scale = numpy.sqrt(numpy.linalg.det(self.A))
        self._dmats_cache = {}

    def _tabulate(self, n, pts):
        '''A version of tabulate() that also works for a single point.
        '''
        dim = self.ref_el.get_spatial_dimension()
        results = [None] * self.get_num_members(n)
        results[0] = sum((pts[i] - pts[i] for i in range(dim)), 1.)
        if n == 0:
            return results
        m1, m2 = self.A.shape
        ref_pts = [sum((self.A[i][j] * pts[j] for j in range(m2)), self.b[i])
                   for i in range(m1)]
        recurrence(dim, n, self._make_factors(ref_pts), results)
        self._normalize(n, results)
        return results

    def _tabulate_derivatives(self, n, pts):
        '''A version of tabulate_derivatives() that also works for a single point.
        '''
        dim = self.ref_el.get_spatial_dimension()
        phi = [None] * self.get_num_members(n)
        dphi = [None] * self.get_num_members(n)
        phi[0] = sum((pts[i] - pts[i] for i in range(dim)), 1.)
        dphi[0] = pts - pts
        if n == 0:
            return phi, dphi
        m1, m2 = self.A.shape
        ref_pts = [sum((self.A[i][j] * pts[j] for j in range(m2)), self.b[i])
                   for i in range(m1)]

        ref_pts = numpy.array(ref_pts)
        factors = self._make_factors(ref_pts)
        dfactors = self._make_dfactors(ref_pts)
        recurrence(dim, n, factors, phi, dfactors=dfactors, dphi=dphi)
        self._normalize(n, phi)
        self._normalize(n, dphi)
        return phi, dphi

    def make_dmats(self, degree):
        """Returns a numpy array with the expansion coefficients dmat[k, j, i]
        of the gradient of each member of the expansion set:
            d/dx_k phi_j = sum_i dmat[k, j, i] phi_i.
        """
        cache = self._dmats_cache
        key = degree
        try:
            return cache[key]
        except KeyError:
            pass
        if degree == 0:
            return cache.setdefault(key, numpy.zeros((self.ref_el.get_spatial_dimension(), 1, 1), "d"))
        pts = self.point_set.recursive_points(self.ref_el.get_vertices(), degree)

        v, dv = self._tabulate_derivatives(degree, numpy.transpose(pts))
        dv = numpy.array(dv).transpose((1, 2, 0))
        dmats = numpy.linalg.solve(numpy.transpose(v), dv)
        return cache.setdefault(key, dmats)

    def _tabulate_jet(self, degree, pts, order=0):
        from FIAT.polynomial_set import mis
        result = {}
        base_vals = self.tabulate(degree, pts)
        dmats = self.make_dmats(degree) if order > 0 else []
        for i in range(order + 1):
            alphas = mis(self.ref_el.get_spatial_dimension(), i)
            for alpha in alphas:
                beta = next((beta for beta in sorted(result, reverse=True)
                             if all(bj <= aj for bj, aj in zip(beta, alpha))), (0,) * len(alpha))
                vals = base_vals if sum(beta) == 0 else result[beta]
                for dmat, start, end in zip(dmats, beta, alpha):
                    for j in range(start, end):
                        vals = numpy.dot(dmat.T, vals)
                result[alpha] = vals
        return result


class PointExpansionSet(ExpansionSet):
    """Evaluates the point basis on a point reference element."""
    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 0:
            raise ValueError("Must have a point")
        super(PointExpansionSet, self).__init__(ref_el)

    def get_num_members(self, n):
        return 1

    def tabulate(self, n, pts):
        """Returns a numpy array A[i,j] = phi_i(pts[j]) = 1.0."""
        assert n == 0
        return numpy.ones((1, len(pts)))

    def tabulate_derivatives(self, n, pts):
        """Returns a numpy array of size A where A[i,j] = phi_i(pts[j])
        but where each element is an empty tuple (). This maintains
        compatibility with the interfaces of the interval, triangle and
        tetrahedron expansions."""
        deriv_vals = numpy.empty_like(self.tabulate(n, pts), dtype=tuple)
        deriv_vals.fill(())

        return deriv_vals


class LineExpansionSet(ExpansionSet):
    """Evaluates the Legendre basis on a line reference element."""
    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 1:
            raise Exception("Must have a line")
        super(LineExpansionSet, self).__init__(ref_el)

    def get_num_members(self, n):
        return n + 1

    def _make_factors(self, ref_pts):
        return [ref_pts[0], 1., 0., 0.]

    def _make_dfactors(self, ref_pts):
        dx = ref_pts - ref_pts + self.A[:, 0][:, None]
        return [dx, 0.*dx, 0.*dx, 0.*dx]

    def _normalize(self, n, phi):
        for p in range(n + 1):
            phi[p] *= math.sqrt(p + 0.5)

    def tabulate(self, n, pts):
        """Returns a numpy array A[i,j] = phi_i(pts[j])"""
        if len(pts) > 0:
            ref_pts = numpy.array([self.mapping(pt) for pt in pts])
            psitilde_as = jacobi.eval_jacobi_batch(0, 0, n, ref_pts)

            results = numpy.zeros((n + 1, len(pts)), type(pts[0][0]))
            for k in range(n + 1):
                results[k, :] = psitilde_as[k, :] * math.sqrt(k + 0.5)

            return results
        else:
            return []

    def tabulate_derivatives(self, n, pts):
        """Returns a tuple of length one (A,) such that
        A[i,j] = D phi_i(pts[j]).  The tuple is returned for
        compatibility with the interfaces of the triangle and
        tetrahedron expansions."""
        ref_pts = numpy.array([self.mapping(pt) for pt in pts])
        psitilde_as_derivs = jacobi.eval_jacobi_deriv_batch(0, 0, n, ref_pts)

        # Jacobi polynomials defined on [-1, 1], first derivatives need scaling
        psitilde_as_derivs *= 2.0 / self.ref_el.volume()

        results = numpy.zeros((n + 1, len(pts)), "d")
        for k in range(0, n + 1):
            results[k, :] = psitilde_as_derivs[k, :] * numpy.sqrt(k + 0.5)

        vals = self.tabulate(n, pts)
        deriv_vals = (results,)

        # Create the ordinary data structure.
        dv = []
        for i in range(vals.shape[0]):
            dv.append([])
            for j in range(vals.shape[1]):
                dv[-1].append((vals[i][j], [deriv_vals[0][i][j]]))

        return dv


class TriangleExpansionSet(ExpansionSet):
    """Evaluates the orthonormal Dubiner basis on a triangular
    reference element."""
    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 2:
            raise Exception("Must have a triangle")
        super(TriangleExpansionSet, self).__init__(ref_el)

    def get_num_members(self, n):
        return (n + 1) * (n + 2) // 2

    def tabulate(self, n, pts):
        if len(pts) == 0:
            return numpy.array([])
        else:
            return numpy.array(self._tabulate(n, numpy.array(pts).T))

    def _make_factors(self, ref_pts):
        x = ref_pts[0]
        y = ref_pts[1]
        return [0.5 * (1. + 2. * x + y),
                (0.5 * (1. - y)) ** 2,
                1. + y,
                1.]

    def _make_dfactors(self, ref_pts):
        y = ref_pts[1]
        dx = ref_pts - ref_pts + self.A[:, 0][:, None]
        dy = ref_pts - ref_pts + self.A[:, 1][:, None]
        return [dx + 0.5 * dy,
                -0.5 * (1. - y) * dy,
                dy,
                0 * dx]

    def _normalize(self, n, phi):
        idx = morton_index2
        for p in range(n + 1):
            for q in range(n - p + 1):
                phi[idx(p, q)] *= math.sqrt((p + 0.5) * (p + q + 1.0))

    def tabulate_derivatives(self, n, pts):
        order = 1
        data = _tabulate_dpts(self._tabulate, 2, n, order, numpy.array(pts))
        # Put data in the required data structure, i.e.,
        # k-tuples which contain the value, and the k-1 derivatives
        # (gradient, Hessian, ...)
        m = data[0].shape[0]
        n = data[0].shape[1]
        data2 = [[tuple([data[r][i][j] for r in range(order+1)])
                  for j in range(n)]
                 for i in range(m)]
        return data2

    def tabulate_jet(self, n, pts, order=1):
        return _tabulate_dpts(self._tabulate, 2, n, order, numpy.array(pts))


class TetrahedronExpansionSet(ExpansionSet):
    """Collapsed orthonormal polynomial expanion on a tetrahedron."""
    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 3:
            raise Exception("Must be a tetrahedron")
        super(TetrahedronExpansionSet, self).__init__(ref_el)

    def get_num_members(self, n):
        return (n + 1) * (n + 2) * (n + 3) // 6

    def tabulate(self, n, pts):
        if len(pts) == 0:
            return numpy.array([])
        else:
            return numpy.array(self._tabulate(n, numpy.array(pts).T))

    def _make_factors(self, ref_pts):
        x = ref_pts[0]
        y = ref_pts[1]
        z = ref_pts[2]
        return [0.5 * (2. + 2. * x + y + z),
                (0.5 * (y + z))**2,
                1. + y,
                0.5 * (1. - z)]

    def _make_dfactors(self, ref_pts):
        y = ref_pts[1]
        z = ref_pts[2]
        dx = ref_pts - ref_pts + self.A[:, 0][:, None]
        dy = ref_pts - ref_pts + self.A[:, 1][:, None]
        dz = ref_pts - ref_pts + self.A[:, 2][:, None]
        return [dx + 0.5 * dy + 0.5 * dz,
                0.5 * (y + z) * (dy + dz),
                dy,
                -0.5 * dz]

    def _normalize(self, n, phi):
        idx = morton_index3
        for p in range(n + 1):
            for q in range(n - p + 1):
                for r in range(n - p - q + 1):
                    phi[idx(p, q, r)] *= math.sqrt((p + 0.5) * (p + q + 1.0) * (p + q + r + 1.5))

    def tabulate_derivatives(self, n, pts):
        order = 1
        D = 3
        data = _tabulate_dpts(self._tabulate, D, n, order, numpy.array(pts))
        # Put data in the required data structure, i.e.,
        # k-tuples which contain the value, and the k-1 derivatives
        # (gradient, Hessian, ...)
        m = data[0].shape[0]
        n = data[0].shape[1]
        data2 = [[tuple([data[r][i][j] for r in range(order + 1)])
                  for j in range(n)]
                 for i in range(m)]
        return data2

    def tabulate_jet(self, n, pts, order=1):
        return _tabulate_dpts(self._tabulate, 3, n, order, numpy.array(pts))


def polynomial_dimension(ref_el, degree):
    """Returns the dimension of the space of polynomials of degree no
    greater than degree on the reference element."""
    if ref_el.get_shape() == reference_element.POINT:
        if degree > 0:
            raise ValueError("Only degree zero polynomials supported on point elements.")
        return 1
    elif ref_el.get_shape() == reference_element.LINE:
        return max(0, degree + 1)
    elif ref_el.get_shape() == reference_element.TRIANGLE:
        return max((degree + 1) * (degree + 2) // 2, 0)
    elif ref_el.get_shape() == reference_element.TETRAHEDRON:
        return max(0, (degree + 1) * (degree + 2) * (degree + 3) // 6)
    else:
        raise ValueError("Unknown reference element type.")
