from numbers import Number
from typing import List, Tuple, Dict
import numpy as np


def distance(arr1: np.array, arr2: np.array) -> float:
    """Given two 2D or 3D coordinate tuples, return the Euclidean distance
    between them. This implementation is simpler than the scipy.distance
    one, but it is much more efficient for small array.

    Parameters
    ----------
    arr1 : np.array
        The first point
    arr2 : np.array
        The second point

    Returns
    -------
    float
        The Euclidean distance between the points.
    """
    return np.sqrt(np.square(arr1 - arr2).sum())


class DistanceMap:
    """
    The DistanceMap is a dictionary containing the distance from a
    center point to each point in a list of neighbor relative locations.

    This exists to avoid repeated recomputation of neighbor distances.

    For instance, if a neighborhood included one neighbor that was one
    unit in the positive x direction, and one that was offset by
    one unit in both the x and y directions, this map would be summarized
    by the following:

    {
        (1, 0): 1,
        (1, 1): math.sqrt(2)
    }
    """

    def __init__(self, relative_neighbor_locs: List[Tuple]):
        """Intializes a DistanceMap.

        Parameters
        ----------
        relative_neighbor_locs : List[Tuple]
            The relative neighbor locations to calculate and store distances
            for.
        """
        self.distances: np.array = self._find_distances(relative_neighbor_locs)

    def _find_distances(
        self, relative_neighbor_locs: List[Tuple]
    ) -> Dict[Tuple, float]:
        """Generates a map of relative neighbor locations to their distances away.

        Parameters
        ----------
        relative_neighbor_locs : List[Tuple]
            _description_

        Returns
        -------
        Dict[Tuple, float]
            A map of each location to it's distance away.
        """
        distances = {}
        for loc in relative_neighbor_locs:
            distances[loc] = self._distance(np.zeros(len(loc)), np.array(loc))

        return distances

    def get_dist(self, relative_loc: Tuple[Number]) -> float:
        """Given a relative location, returns the stored distance for that location.

        Parameters
        ----------
        relative_loc : Tuple[Number]
            The location of the neighbor to calculate a distance for.

        Returns
        -------
        float
            The distance of that neighbor.
        """
        return self.distances.get(relative_loc)


class EuclideanDistanceMap(DistanceMap):
    """A distance map for storing Euclidean distances."""

    def _distance(self, p1, p2):
        return round(distance(p1, p2), 2)


class ManhattanDistanceMap(DistanceMap):
    """A distance map that calculates and stores Manhattan distances"""

    def _distance(self, p1, p2):
        return np.abs(p1 - p2).sum()
