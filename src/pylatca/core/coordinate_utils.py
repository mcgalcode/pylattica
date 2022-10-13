import itertools
from typing import Iterable

# TO DELETE
# def get_in_range(upper, val):
#     if val < 0:
#         dist = -val - 1
#         rem = math.remainder(dist, upper)
#         return int(upper - rem)
#     elif val > upper:
#         dist = val - upper - 1
#         rem = math.remainder(dist, upper)
#         return int(rem)
#     else:
#         return int(val)

# def get_coords_in_box(size, coords):
#     highest_val = size - 1
#     return tuple([get_in_range(highest_val, c) for c in coords])

def get_points_in_box(lbs: Iterable[int], ubs: Iterable[int]) -> list[list[int]]:
    """Using a Cartesian product, returns a list of integer points in some region.
    
    Parameters
    ----------
    lbs : Iterable[Number]
        _description_
    ubs : Iterable[Number]
        _description_

    Returns
    -------
    list[list[Number]]
        _description_

    Examples
    --------
    >>> get_points_in_box([0,1], [1,2])
    [[0, 1], [0, 2], [1, 1], [1, 2]]
    """    
    args = [list(range(lb, ub)) for lb, ub in zip(lbs, ubs)]
    return list(itertools.product(*args))
