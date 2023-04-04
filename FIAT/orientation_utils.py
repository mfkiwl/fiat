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
