import numpy as np
import gem


class UndefinedError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class FiniteElementBase(object):

    def __init__(self):
        pass

    @property
    def cell(self):
        '''The reference cell on which the element is defined.
        '''

        return self._cell

    @property
    def degree(self):
        '''The degree of the embedding polynomial space.

        In the tensor case this is a tuple.
        '''

        return self._degree

    @property
    def entity_dofs(self):
        '''The map of topological entities to degrees of
        freedom for the finite element.

        Note that entity numbering needs to take into account the tensor case.
        '''

        raise NotImplementedError

    @property
    def entity_closure_dofs(self):
        '''The map of topological entities to degrees of
        freedom on the closure of those entities for the finite element.'''

        raise NotImplementedError

    @property
    def index_shape(self):
        '''A tuple indicating the number of degrees of freedom in the
        element. For example a scalar quadratic Lagrange element on a triangle
        would return (6,) while a vector valued version of the same element
        would return (6, 2)'''

        raise NotImplementedError

    @property
    def value_shape(self):
        '''A tuple indicating the shape of the element.'''

        raise NotImplementedError

    def get_indices():
        '''A tuple of GEM :class:`Index` of the correct extents to loop over
        the basis functions of this element.'''

        raise NotImplementedError

    def basis_evaluation(self, q, entity=None, derivative=None):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param index: the basis function index.
        :param q: the quadrature rule.
        :param q_index: the quadrature index.
        :param entity: the cell entity on which to tabulate.
        :param derivative: the derivative to take of the basis functions.
        '''

        raise NotImplementedError

    def dual_evaluation(self, kernel_data):
        '''Return code for evaluating an expression at the dual set.

        Note: what does the expression need to look like?
        '''

        raise NotImplementedError

    def __hash__(self):
        """Elements are equal if they have the same class, degree, and cell."""

        return hash((type(self), self._cell, self._degree))

    def __eq__(self, other):
        """Elements are equal if they have the same class, degree, and cell."""

        return type(self) == type(other) and self._cell == other._cell and\
            self._degree == other._degree


class FiatElementBase(FiniteElementBase):
    """Base class for finite elements for which the tabulation is provided
    by FIAT."""
    def __init__(self, cell, degree):
        super(FiatElementBase, self).__init__()

        self._cell = cell
        self._degree = degree

    def get_indices():
        '''A tuple of GEM :class:`Index` of the correct extents to loop over
        the basis functions of this element.'''

        return (gem.Index(self_fiat_element.get_spatial_dimension()),)

    def basis_evaluation(self, q, entity=None, derivative=0):
        '''Return code for evaluating the element at known points on the
        reference element.

        :param q: the quadrature rule.
        :param entity: the cell entity on which to tabulate.
        :param derivative: the derivative to take of the basis functions.
        '''

        assert entity == None

        dim = self.cell.get_spatial_dimension()

        i = self.get_indices()
        qi = q.get_indices()
        di = tuple(gem.Index() for i in range(dim)) 

        fiat_tab = self._fiat_element.tabulate(derivative, q.points)

        def tabtensor(pre_indices=()):
            if len(pre_indices) < dim:
                return gem.ListTensor([tabtensor(pre_indices + (i,))
                                       for i in range(derivative + 1)])
            else:
                return gem.ListTensor([gem.Literal(fiat_tab.get(pre_indices + (i,)).T, None)
                                       for i in range(derivative + 1)])

        return ComponentTensor(Indexed(tabtensor(), di + qi + i), qi + i + di)

    @property
    def entity_dofs(self):
        '''The map of topological entities to degrees of
        freedom for the finite element.

        Note that entity numbering needs to take into account the tensor case.
        '''

        return self._fiat_element.entity_dofs()

    @property
    def entity_closure_dofs(self):
        '''The map of topological entities to degrees of
        freedom on the closure of those entities for the finite element.'''

        return self._fiat_element.entity_closure_dofs()

    @property
    def facet_support_dofs(self):
        '''The map of facet id to the degrees of freedom for which the
        corresponding basis functions take non-zero values.'''

        return self._fiat_element.entity_support_dofs()
