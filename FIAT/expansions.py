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


def morton_index2(p, q=0):
    return (p + q) * (p + q + 1) // 2 + q


def morton_index3(p, q=0, r=0):
    return (p + q + r)*(p + q + r + 1)*(p + q + r + 2)//6 + (q + r)*(q + r + 1)//2 + r


def jrc(a, b, n):
    """Jacobi recurrence coefficients"""
    an = (2*n+1+a+b)*(2*n+2+a+b) / (2*(n+1)*(n+1+a+b))
    bn = (a*a-b*b) * (2*n+1+a+b) / (2*(n+1)*(2*n+a+b)*(n+1+a+b))
    cn = (n+a)*(n+b)*(2*n+2+a+b) / ((n+1)*(n+1+a+b)*(2*n+a+b))
    return an, bn, cn


def pad_coordinates(ref_pts, embedded_dim):
    """Pad reference coordinates by appending -1.0."""
    return tuple(ref_pts) + (-1.0, )*(embedded_dim - len(ref_pts))


def pad_jacobian(A, embedded_dim):
    """Pad coordinate mapping Jacobian by appending zero rows."""
    A = numpy.pad(A, [(0, embedded_dim - A.shape[0]), (0, 0)])
    return tuple(row[:, None] for row in A)


def jacobi_factors(x, y, z, dx, dy, dz):
    fb = 0.5 * (y + z)
    fa = x + fb + 1.0
    fc = fb ** 2
    dfa = dfb = dfc = None
    if dx is not None:
        dfb = 0.5 * (dy + dz)
        dfa = dx + dfb
        dfc = 2 * fb * dfb
    return fa, fb, fc, dfa, dfb, dfc


def dubiner_recurrence(dim, n, ref_pts, phi, jacobian=None, dphi=None):
    """Dubiner recurrence from (Kirby 2010)"""
    skip_derivs = dphi is None
    phi[0] = sum((ref_pts[i] - ref_pts[i] for i in range(dim)), 1.)
    if not skip_derivs:
        dphi[0] = ref_pts - ref_pts
    if dim == 0 or n == 0:
        return
    if dim > 3 or dim < 0:
        raise ValueError("Invalid number of spatial dimensions")

    idx = (lambda p: p, morton_index2, morton_index3)[dim-1]
    pad_dim = dim + 2
    X = pad_coordinates(ref_pts, pad_dim)
    if jacobian is None:
        dX = (None,) * pad_dim
    else:
        dX = pad_jacobian(jacobian, pad_dim)

    for codim in range(dim):
        # Extend the basis from codim to codim + 1
        fa, fb, fc, dfa, dfb, dfc = jacobi_factors(*X[codim:codim+3], *dX[codim:codim+3])
        for sub_index in reference_element.lattice_iter(0, n, codim):
            # handle i = 1
            icur = idx(*sub_index, 0)
            inext = idx(*sub_index, 1)
            alpha = 2 * sum(sub_index) + len(sub_index)
            b = 0.5 * alpha
            a = b + 1.0
            factor = a * fa - b * fb
            phi[inext] = factor * phi[icur]
            if not skip_derivs:
                dfactor = a * dfa - b * dfb
                dphi[inext] = factor * dphi[icur] + phi[icur] * dfactor
            # general i by recurrence
            for i in range(1, n - sum(sub_index)):
                iprev, icur, inext = icur, inext, idx(*sub_index, i + 1)
                a, b, c = jrc(alpha, 0, i)
                factor = a * fa - b * fb
                phi[inext] = factor * phi[icur] - c * (fc * phi[iprev])
                if skip_derivs:
                    continue
                dfactor = a * dfa - b * dfb
                dphi[inext] = (factor * dphi[icur] + phi[icur] * dfactor -
                               c * (fc * dphi[iprev] + phi[iprev] * dfc))
        # normalize
        for alpha in reference_element.lattice_iter(0, n+1, codim+1):
            scale = math.sqrt(sum(alpha) + 0.5 * len(alpha))
            phi[idx(*alpha)] *= scale
            if skip_derivs:
                continue
            dphi[idx(*alpha)] *= scale


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

    def get_num_members(self, n):
        D = self.ref_el.get_spatial_dimension()
        return math.comb(n + D, D)

    def _mapping(self, pts):
        if isinstance(pts, numpy.ndarray) and len(pts.shape) == 2:
            return numpy.dot(self.A, pts) + self.b[:, None]
        else:
            m1, m2 = self.A.shape
            return [sum((self.A[i, j] * pts[j] for j in range(m2)), self.b[i])
                    for i in range(m1)]

    def _tabulate(self, n, pts):
        """A version of tabulate() that also works for a single point.
        """
        D = self.ref_el.get_spatial_dimension()
        results = [None] * self.get_num_members(n)
        dubiner_recurrence(D, n, self._mapping(pts), results)
        return results

    def _tabulate_derivatives(self, n, pts):
        """A version of tabulate_derivatives() that also works for a single point.
        """
        D = self.ref_el.get_spatial_dimension()
        num_members = self.get_num_members(n)
        phi = [None] * num_members
        dphi = [None] * num_members
        dubiner_recurrence(D, n, self._mapping(pts), phi, jacobian=self.A, dphi=dphi)
        return phi, dphi

    def tabulate(self, n, pts):
        if len(pts) == 0:
            return numpy.array([])
        return numpy.array(self._tabulate(n, numpy.transpose(pts)))

    def tabulate_derivatives(self, n, pts):
        D = self.ref_el.get_spatial_dimension()
        vals, deriv_vals = self._tabulate_derivatives(n, numpy.transpose(pts))
        # Create the ordinary data structure.
        data = [[(vals[i][j], [deriv_vals[i][r][j] for r in range(D)])
                 for j in range(len(vals[0]))]
                for i in range(len(vals))]
        return data

    def tabulate_jet(self, n, pts, order=1):
        D = self.ref_el.get_spatial_dimension()
        return _tabulate_dpts(self._tabulate, D, n, order, numpy.array(pts))

    def get_dmats(self, degree):
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

        pts = reference_element.make_lattice(self.ref_el.get_vertices(), degree, variant="gl")
        v, dv = self._tabulate_derivatives(degree, numpy.transpose(pts))
        dv = numpy.array(dv).transpose((1, 2, 0))
        dmats = numpy.linalg.solve(numpy.transpose(v), dv)
        return cache.setdefault(key, dmats)

    def _tabulate_jet(self, degree, pts, order=0):
        from FIAT.polynomial_set import mis
        result = {}
        D = self.ref_el.get_spatial_dimension()
        if order == 0:
            base_vals = self.tabulate(degree, pts)
        else:
            v, dv = self._tabulate_derivatives(degree, numpy.transpose(pts))
            base_vals = numpy.array(v)
            dtildes = numpy.array(dv).transpose((1, 0, 2))
            alphas = mis(D, 1)
            for alpha in alphas:
                result[alpha] = next(dv for dv, ai in zip(dtildes, alpha) if ai > 0)

        def distance(alpha, beta):
            return sum(ai != bi for ai, bi in zip(alpha, beta))

        # Only use dmats if order > 1
        dmats = self.get_dmats(degree) if order > 1 else []
        base_alpha = (0,) * D
        for i in range(order + 1):
            alphas = mis(D, i)
            for alpha in alphas:
                if alpha in result:
                    continue
                if len(result) > 0 and i > 0:
                    base_alpha = next(a for a in result if sum(a) == i-1 and distance(alpha, a) == 1)
                    base_vals = result[base_alpha]
                vals = base_vals
                for dmat, start, end in zip(dmats, base_alpha, alpha):
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

    def _tabulate(self, n, pts):
        """Returns a numpy array A[i,j] = phi_i(pts[j])"""
        if len(pts) > 0:
            ref_pts = self._mapping(pts).T
            results = jacobi.eval_jacobi_batch(0, 0, n, ref_pts)
            for p in range(n + 1):
                results[p] *= math.sqrt(p + 0.5)
            return results
        else:
            return []

    def _tabulate_derivatives(self, n, pts):
        """Returns a tuple of (vals, derivs) such that
        vals[i,j] = phi_i(pts[j]), derivs[i,j] = D vals[i,j]."""
        ref_pts = self._mapping(pts).T
        derivs = jacobi.eval_jacobi_deriv_batch(0, 0, n, ref_pts)

        # Jacobi polynomials defined on [-1, 1], first derivatives need scaling
        derivs *= 2.0 / self.ref_el.volume()
        for p in range(n + 1):
            derivs[p] *= math.sqrt(p + 0.5)
        return self._tabulate(n, pts), derivs[:, None, :]


class TriangleExpansionSet(ExpansionSet):
    """Evaluates the orthonormal Dubiner basis on a triangular
    reference element."""
    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 2:
            raise Exception("Must have a triangle")
        super(TriangleExpansionSet, self).__init__(ref_el)


class TetrahedronExpansionSet(ExpansionSet):
    """Collapsed orthonormal polynomial expansion on a tetrahedron."""
    def __init__(self, ref_el):
        if ref_el.get_spatial_dimension() != 3:
            raise Exception("Must be a tetrahedron")
        super(TetrahedronExpansionSet, self).__init__(ref_el)


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
