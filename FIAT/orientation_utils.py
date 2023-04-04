import itertools
import numpy as np


def make_entity_permutations_simplex(dim, npoints):
    r"""Make orientation-permutation map for the given
    simplex dimension, dim, and the number of points along
    each axis

    As an example, we first compute the orientation of a
    triangular cell:

       +                    +
       | \                  | \
       1   0               47   42
       |     \              |     \
       +--2---+             +--43--+
    FIAT canonical     Mapped example physical cell

    Suppose that the facets of the physical cell
    are canonically ordered as:

    C = [43, 42, 47]

    FIAT facet to Physical facet map is given by:

    M = [42, 47, 43]

    Then the orientation of the cell is computed as:

    C.index(M[0]) = 1; C.remove(M[0])
    C.index(M[1]) = 1; C.remove(M[1])
    C.index(M[2]) = 0; C.remove(M[2])

    o = (1 * 2!) + (1 * 1!) + (0 * 0!) = 3

    For npoints = 3, there are 6 DoFs:

        5                   0
        3 4                 1 3
        0 1 2               2 4 5
    FIAT canonical     Physical cell canonical

    The permutation associated with o = 3 then is:

    [2, 4, 5, 1, 3, 0]

    The output of this function contains one such permutation
    for each orientation for the given simplex dimension and
    the number of points along each axis.
    """
    from FIAT.polynomial_set import mis

    if npoints <= 0:
        return {o: [] for o in range(np.math.factorial(dim + 1))}
    a = np.array(sorted(mis(dim + 1, npoints - 1)), dtype=int)[:, ::-1]
    index_perms = sorted(itertools.permutations(range(dim + 1)))
    perms = {}
    for o, index_perm in enumerate(index_perms):
        perm = np.lexsort(np.transpose(a[:, index_perm]))
        perms[o] = perm.tolist()
    return perms


def make_entity_permutations_tensorproduct(cells, dim, o_p_maps):
    """Make orientation-permutation map for an entity of a tensor product cell.

    :arg cells: List of cells composing the tensor product cell.
    :arg dim: List of (sub)dimensions of the component cells that makes the tensor product (sub)cell.
    :arg o_p_maps: List of orientation-permutation maps of the component (sub)cells.
    :returns: The orientation-permutation map of the tensor product (sub)cell.

    Example
    -------

    .. code-block:: python3

        from FIAT.reference_element import UFCInterval
        from FIAT.orientation_utils import make_entity_permutations_tensorproduct


        cells = [UFCInterval(), UFCInterval()]
        m = make_entity_permutations_tensorproduct(cells, [1, 0], [{0: [0, 1], 1: [1, 0]}, {0: [0]}])
        print(m)
        # prints:
        # {(0, 0, 0): [0, 1],
        #  (0, 1, 0): [1, 0]}
        m = make_entity_permutations_tensorproduct(cells, [1, 1], [{0: [0, 1], 1: [1, 0]}, {0: [0, 1], 1: [1, 0]}])
        print(m)
        # prints:
        # {(0, 0, 0): [0, 1, 2, 3],
        #  (0, 0, 1): [1, 0, 3, 2],
        #  (0, 1, 0): [2, 3, 0, 1],
        #  (0, 1, 1): [3, 2, 1, 0],
        #  (1, 0, 0): [0, 2, 1, 3],
        #  (1, 0, 1): [2, 0, 3, 1],
        #  (1, 1, 0): [1, 3, 0, 2],
        #  (1, 1, 1): [3, 1, 2, 0]}

    """
    from FIAT.reference_element import UFCInterval

    # Handle extrinsic orientations.
    # This is complex and we need to think to make this function more general.
    # One interesting case is pyramid x pyramid. There are two types of facets
    # in a pyramid cell, quad and triangle, and two types of intervals, ones
    # attached to quad (Iq) and ones attached to triangles (It). When we take
    # a tensor product of two pyramid cells, there are different kinds of tensor
    # product of intervals, i.e., Iq x Iq, Iq x It, It x Iq, It x It, and we
    # need a careful thought on how many possible extrinsic orientations we need
    # to consider for each.
    # For now we restrict ourselves to specific cases.
    nprod = len(o_p_maps)
    assert len(cells) == nprod
    assert len(dim) == nprod
    if len(set(cells)) == nprod:
        # All components have different cells.
        # Example: triangle x interval.
        #          dim == (2, 1) ->
        #          triangle x interval (1 possible extrinsic orientation).
        axis_perms = (tuple(range(nprod)), )  # Identity: no permutations
    elif len(set(cells)) == 1 and isinstance(cells[0], UFCInterval):
        # Tensor product of intervals.
        # Example: interval x interval x interval x interval
        #          dim == (0, 1, 1, 1) ->
        #          point x interval x interval x interval  (1! * 3! possible extrinsic orientations).
        axis_perms = sorted(itertools.permutations(range(nprod)))
        for idim, d in enumerate(dim):
            if d == 0:
                # idim-th component does not contribute to the extrinsic orientation.
                axis_perms = [ap for ap in axis_perms if ap[idim] == idim]
    else:
        # More general tensor product cells.
        # Example: triangle x quad x triangle x triangle x interval x interval
        #          dim == (2, 2, 2, 2, 1, 1) ->
        #          triangle x quad x triangle x triangle x interval x interval (3! * 1! * 2! possible extrinsic orientations).
        raise NotImplementedError(f"Unable to compose permutations for {' x '.join([str(cell) for cell in cells])}")
    o_tuple_perm_map = {}
    for eo, ap in enumerate(axis_perms):
        for o_tuple in itertools.product(*[o_p_map.keys() for o_p_map in o_p_maps]):
            ps = [o_p_map[o] for o_p_map, o in zip(o_p_maps, o_tuple)]
            shape = [len(p) for p in ps]
            for idim in range(len(ap)):
                shape[ap[idim]] = len(ps[idim])
            size = np.prod(shape)
            if size == 0:
                o_tuple_perm_map[(eo, ) + o_tuple] = []
            else:
                a = np.arange(size).reshape(shape)
                # Tensorproduct elements on a tensorproduct cell of intervals:
                # When we map the reference element to the physical element, we fisrt apply
                # the extrinsic orientation and then the intrinsic orientation.
                # Thus, to make the a.reshape(-1) trick work in the below,
                # we apply the inverse operation on a; we first apply the inverse of the
                # intrinsic orientation and then the inverse of the extrinsic orienataion.
                for idim, p in enumerate(ps):
                    # Note that p inverse = p for interval elements.
                    # Do not use p inverse (just use p) for elements on simplices
                    # as p already does what we want by construction.
                    a = a.swapaxes(0, ap[idim])[p, :].swapaxes(0, ap[idim])
                apinv = list(range(nprod))
                for idim in range(len(ap)):
                    apinv[ap[idim]] = idim
                a = np.moveaxis(a, range(nprod), apinv)
                o_tuple_perm_map[(eo, ) + o_tuple] = a.reshape(-1).tolist()
    return o_tuple_perm_map
