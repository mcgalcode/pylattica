from random import randint
import numpy as np

from rxn_ca.core.distance_map import DistanceMap, EuclideanDistanceMap, ManhattanDistanceMap

PADDING_VAL = -1

class Neighborhood():

    def __init__(self, radius, distance_map: DistanceMap = None):
        self.radius = radius
        if distance_map is None:
            self.distance_map = EuclideanDistanceMap(radius * 2 + 1)
        else:
            self.distance_map = distance_map

    def pad_state(self, step, pad_val = PADDING_VAL):
        return np.pad(step.state, self.radius, 'constant', constant_values=pad_val)

    def get_in_step(self, step, i, j):
        return self.get(self.pad_state(step), i, j)

    def get_distance(self, i, j):
        return self.distance_map.distances[(i, j)]

    def _get_square_neighborhood(self, padded_state, i, j):
        # We are accepting coordinates that don't consider padding,
        # So modify them to make 0,0 the first non-padding entry
        i = i + self.radius
        j = j + self.radius

        curr_up = i - self.radius
        curr_down = i + self.radius + 1
        curr_left = j - self.radius
        curr_right = j + self.radius + 1

        return padded_state[curr_up:curr_down, curr_left:curr_right]

    def iterate_step(self, step, i, j, exclude_center = True):
        return self.iterate(self.pad_state(step), i, j, exclude_center)

    def iterate(self, padded_state, i, j, exclude_center = True):
        subcell = self.get(padded_state, i, j)
        cell_center = np.array([int(subcell.shape[0] / 2), int(subcell.shape[1] / 2)])

        for i in range(subcell.shape[0]):
            for j in range(subcell.shape[1]):
                contents = subcell[(i, j)]
                if contents == PADDING_VAL:
                    continue
                if exclude_center and not (i == cell_center[0] and j == cell_center[1]):
                    distance = self.get_distance(i, j)
                    yield subcell[(i, j)], distance


class MooreNeighborhood(Neighborhood):

    def get(self, padded_state, i, j):
        return self._get_square_neighborhood(padded_state, i, j)


class VonNeumannNeighborhood(Neighborhood):

    def __init__(self, radius, distance_map: DistanceMap = None):
        super().__init__(radius, distance_map)
        self.mh_distances = ManhattanDistanceMap(radius * 2 + 1)

    def get(self, padded_state, i, j):
        square = self._get_square_neighborhood(padded_state, i, j)
        sl = square.shape[0]
        for i in range(sl):
            for j in range(sl):
                if self.mh_distances.distances[i][j] > self.radius:
                    square[i][j] = PADDING_VAL
        return square

class CircularNeighborhood(Neighborhood):

    def __init__(self, radius, distance_map: DistanceMap = None):
        super().__init__(radius, distance_map)
        self._radial_distances = EuclideanDistanceMap(radius * 2 + 1)

    def get(self, padded_state, i, j):
        square = self._get_square_neighborhood(padded_state, i, j)
        sl = square.shape[0]
        for i in range(sl):
            for j in range(sl):
                if self._radial_distances.distances[i][j] > self.radius:
                    square[i][j] = PADDING_VAL
        return square

class PseudoHexagonal(Neighborhood):

    def __init__(self):
        super().__init__(1)

    def get(self, padded_state, i , j):
        square = self._get_square_neighborhood(padded_state, i, j)
        sl = square.shape[0]
        to_exclude = randint(0,3)
        for i in range(sl):
            for j in range(sl):
                if to_exclude == 0 and i == 0:
                    square[i][j] = PADDING_VAL
                if to_exclude == 1 and i == 2:
                    square[i][j] = PADDING_VAL
                if to_exclude == 2 and j == 2:
                    square[i][j] = PADDING_VAL
                if to_exclude == 3 and j == 0:
                    square[i][j] = PADDING_VAL

        return square