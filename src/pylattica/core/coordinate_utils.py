import itertools
from typing import Iterable
import numpy as np

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

def periodic_distance(x0: np.ndarray, x1: np.ndarray, dimensions: np.ndarray) -> float:
    """Returns the distance between two points under periodic boundary
    conditions. Can handle many points at once if x0 and x1 are more than 1 dimensional.

    https://stackoverflow.com/questions/11108869/optimizing-python-distance-calculation-while-accounting-for-periodic-boundary-co

    There are some other options for this here:

    https://pymatgen.org/pymatgen.core.lattice.html#pymatgen.core.lattice.Lattice.get_all_distances
    https://pymatgen.org/pymatgen.core.lattice.html#pymatgen.core.lattice.Lattice.get_distance_and_image

    Parameters
    ----------
    x0 : np.ndarray
        The first point, or array of points
    x1 : np.ndarray
        The second point, or array of points
    dimensions : np.ndarray
        The periodic bounds

    Returns
    -------
    float
        The distance between two points
    """    
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))