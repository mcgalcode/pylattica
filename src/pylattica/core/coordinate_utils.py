import itertools
from typing import Iterable, List


def get_points_in_cube(lb: int, ub: int, dim: int) -> List[List[int]]:
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
    List[List[int]]
        A list of points in the cube
    """
    return get_points_in_box([lb for _ in range(dim)], [ub for _ in range(dim)])


def get_points_in_box(lbs: Iterable[int], ubs: Iterable[int]) -> List[List[int]]:
    """Using a Cartesian product, returns a list of integer points in some region.

    Parameters
    ----------
    lbs : Iterable[Number]
        _description_
    ubs : Iterable[Number]
        _description_

    Returns
    -------
    List[List[Number]]
        _description_

    Examples
    --------
    >>> get_points_in_box([0,1], [1,2])
    [[0, 1], [0, 2], [1, 1], [1, 2]]
    """
    args = [list(range(lb, ub)) for lb, ub in zip(lbs, ubs)]
    return list(itertools.product(*args))
