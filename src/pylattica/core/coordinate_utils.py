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

def get_points_in_cube(lb: int, ub: int, dim: int) -> list[list[int]]:
    """Returns the list of all integer separated points in a box of dimension
    dim with lower bound and upper bounds in each direction specified by
    lb and ub. 

    Parameters
    ----------
    lb : int
        the lower bound of the cube
    ub : int
        the uppwer bound of the cube

    Returns
    -------
    list[list[int]]
        A list of points in the cube
    """    
    return get_points_in_box([lb for _ in range(dim)], [ub for _ in range(dim)])

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
