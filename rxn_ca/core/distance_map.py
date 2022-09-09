from typing import List, Tuple
import numpy as np

def distance(arr1: np.array, arr2: np.array) -> float:
    """Given two 2D or 3D coordinate tuples, return the euclidean distance between them. This implementation
    is simpler than the scipy.distance one, but it is much more efficient for small array.s

    Args:
        arr1 (np.array): The first point
        arr2 (np.array): The second point

    Returns:
        float: The Euclidean distance between the points.
    """
    return np.sqrt(np.square(arr1 - arr2).sum())

class DistanceMap():
    """
    The DistanceMap is a dictionary containing the distance from the center point to every
    other point in a list of relative neighbor locations.
    """

    def __init__(self, relative_neighbor_locs: List[Tuple]):
        """Intializes a DistanceMap.

        Args:
            side_length (int): The size of the distance map to create
        """

        self.distances: np.array = self.find_distances(relative_neighbor_locs)

    def find_distances(self, relative_neighbor_locs: List[Tuple]) -> dict[Tuple, float]:
        """Generates a map of relative locations to their distance from the center cell

        Args:
            relative_neighbor_locs (list[Tuple]): A list of these locations

        Returns:
            np.array: _description_
        """
        distances = {}
        for loc in relative_neighbor_locs:
            distances[loc] = self._distance(np.zeros(len(loc)), np.array(loc))

        return distances

class EuclideanDistanceMap(DistanceMap):

    def _distance(self, p1, p2):
        return round(distance(p1, p2), 2)

class ManhattanDistanceMap(DistanceMap):

    def _distance(self, p1, p2):
        return np.abs(p1 - p2).sum()