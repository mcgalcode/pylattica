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

    def __init__(self, side_length: int, dimension = 2):
        """Intializes a DistanceMap.

        Args:
            side_length (int): The size of the distance map to create
        """
        if side_length % 2 != 1:
            print("Warning: distance map initialized with even side length has an ambiguous center point")
        self.dimension = dimension
        self.distances: np.array = self.find_distances(side_length)

    def find_distances(self, side_length: int) -> np.array:
        """Generates a side_length x side_length grid where each cell contains it's distance
        from the central cell

        Args:
            side_length (int): _description_

        Returns:
            np.array: _description_
        """
        if self.dimension == 2:
            distances: np.array = np.zeros((side_length, side_length))
            cell_center: np.array = np.array([int(distances.shape[0] / 2), int(distances.shape[1] / 2)])

            for i in range(side_length):
                for j in range(side_length):
                    curr_loc = np.array([i, j])
                    distances[(i, j)] = self._distance(cell_center, curr_loc)

            return distances
        elif self.dimension == 3:
            distances: np.array = np.zeros((side_length, side_length, side_length))
            cell_center: np.array = np.array([int(distances.shape[0] / 2), int(distances.shape[1] / 2), int(distances.shape[1] / 2)])

            for i in range(side_length):
                for j in range(side_length):
                    for k in range(side_length):
                        curr_loc = np.array([i, j, k])
                        distances[(i, j, k)] = self._distance(cell_center, curr_loc)

            return distances

class EuclideanDistanceMap(DistanceMap):

    def _distance(self, p1, p2):
        return round(distance(p1, p2), 2)

class ManhattanDistanceMap(DistanceMap):

    def _distance(self, p1, p2):
        return np.abs(p1 - p2).sum()