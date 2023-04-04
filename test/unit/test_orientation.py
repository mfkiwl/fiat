from FIAT.reference_element import UFCInterval
from FIAT.orientation_utils import make_entity_permutations_tensorproduct


def test_orientation_make_entity_permutations_tensorproduct():
    cells = [UFCInterval(), UFCInterval()]
    m = make_entity_permutations_tensorproduct(cells, [1, 0], [{0: [0, 1],
                                                                1: [1, 0]},
                                                               {0: [0]}])
    assert m == {(0, 0, 0): [0, 1],
                 (0, 1, 0): [1, 0]}
    m = make_entity_permutations_tensorproduct(cells, [1, 1], [{0: [0, 1],
                                                                1: [1, 0]},
                                                               {0: [0, 1],
                                                                1: [1, 0]}])
    assert m == {(0, 0, 0): [0, 1, 2, 3],
                 (0, 0, 1): [1, 0, 3, 2],
                 (0, 1, 0): [2, 3, 0, 1],
                 (0, 1, 1): [3, 2, 1, 0],
                 (1, 0, 0): [0, 2, 1, 3],
                 (1, 0, 1): [2, 0, 3, 1],
                 (1, 1, 0): [1, 3, 0, 2],
                 (1, 1, 1): [3, 1, 2, 0]}
