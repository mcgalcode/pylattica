from random import randint
import numpy as np

from rxn_ca.core.distance_map import DistanceMap, EuclideanDistanceMap, ManhattanDistanceMap


class Neighborhood():

    PADDING_VAL = -1

    def state_at(self, padded_state, i, j, padding = None):
        if padding is None:
            padding = self.radius
        return padded_state[i + padding, j + padding]

    def __init__(self, radius, distance_map: DistanceMap = None):
        self.radius = radius
        if distance_map is None:
            self.distance_map = EuclideanDistanceMap(radius * 2 + 1)
        else:
            self.distance_map = distance_map

    def is_padding(self, state):
        return state == self.PADDING_VAL

    def pad_step(self, step):
        return self.pad_state(step.state)

    def pad_state(self, state):
        return np.pad(state, self.radius, 'constant', constant_values=self.PADDING_VAL)

    def get_in_step(self, step, i, j):
        return self.get(self.pad_step(step), i, j)

    def get_in_state(self, state, i, j):
        return self.get(self.pad_state(state), i, j)

    def get_distance(self, i, j):
        return self.distance_map.distances[(i, j)]

    def _get_square_neighborhood(self, padded_state, i, j, overload_radius = None):

        if overload_radius is not None:
            radius = overload_radius
        else:
            radius = self.radius
        # We are accepting coordinates that don't consider padding,
        # So modify them to make 0,0 the first non-padding entry
        i = i + radius
        j = j + radius

        curr_up = i - radius
        curr_down = i + radius + 1
        curr_left = j - radius
        curr_right = j + radius + 1

        return padded_state[curr_up:curr_down, curr_left:curr_right]

    def iterate_step(self, step, i, j, exclude_center = True, overload_radius = None):
        return self.iterate(self.pad_step(step), i, j, exclude_center, overload_radius=overload_radius)

    def iterate_state(self, state, i, j, exclude_center = True, overload_radius = None):
        return self.iterate(self.pad_state(state), i, j, exclude_center, overload_radius=overload_radius)

    def iterate(self, padded_state, i, j, exclude_center = True, overload_radius = None):
        subcell = self.get(padded_state, i, j, overload_radius=overload_radius)
        cell_center = np.array([int(subcell.shape[0] / 2), int(subcell.shape[1] / 2)])
        for i in range(subcell.shape[0]):
            for j in range(subcell.shape[1]):
                contents = subcell[(i, j)]
                if self.is_padding(contents):
                    continue

                if exclude_center and (i == cell_center[0] and j == cell_center[1]):
                    continue

                distance = self.get_distance(i, j)
                yield subcell[(i, j)], distance

    def get(self, padded_state, i, j, overload_radius = None):
        square = np.copy(self._get_square_neighborhood(padded_state, i, j, overload_radius=overload_radius))
        return self._screen_square(square)


class MooreNeighborhood(Neighborhood):

    def _screen_square(self, square):
        return square


class VonNeumannNeighborhood(Neighborhood):

    def __init__(self, radius, distance_map: DistanceMap = None):
        super().__init__(radius, distance_map)
        self.mh_distances = ManhattanDistanceMap(radius * 2 + 1)

    def _screen_square(self, square):
        sl = square.shape[0]
        for i in range(sl):
            for j in range(sl):
                if self.mh_distances.distances[i][j] > self.radius:
                    square[i][j] = self.PADDING_VAL
        return square

class CircularNeighborhood(Neighborhood):

    def __init__(self, radius, distance_map: DistanceMap = None):
        super().__init__(radius, distance_map)
        self._radial_distances = EuclideanDistanceMap(radius * 2 + 1)

    def _screen_square(self, square):
        sl = square.shape[0]
        for i in range(sl):
            for j in range(sl):
                if self._radial_distances.distances[i][j] > self.radius:
                    square[i][j] = self.PADDING_VAL
        return square

class PseudoHexagonal(Neighborhood):

    def __init__(self):
        super().__init__(1)

    def _screen_square(self, square):
        sl = square.shape[0]
        to_exclude = randint(0,1)
        highest_idx = sl - 1
        if to_exclude == 0:
            square[0][0] = self.PADDING_VAL
            square[highest_idx][highest_idx] = self.PADDING_VAL
        else:
            square[highest_idx][0] = self.PADDING_VAL
            square[0][highest_idx] = self.PADDING_VAL

        return square

class PseudoPentagonal(Neighborhood):

    def __init__(self):
        super().__init__(1)

    def _screen_square(self, square):
        sl = square.shape[0]
        to_exclude = randint(0,3)
        for i in range(sl):
            for j in range(sl):
                if to_exclude == 0 and i == 0:
                    square[i][j] = self.PADDING_VAL
                if to_exclude == 1 and i == 2:
                    square[i][j] = self.PADDING_VAL
                if to_exclude == 2 and j == 2:
                    square[i][j] = self.PADDING_VAL
                if to_exclude == 3 and j == 0:
                    square[i][j] = self.PADDING_VAL

        return square