from random import randint
import numpy as np

from .distance_map import DistanceMap, EuclideanDistanceMap, ManhattanDistanceMap, distance

class NeighborhoodView():

    def __init__(self, coords, view, full_state, distance_map) -> None:
        self.coords = coords
        self.view = view
        self.center_value = full_state[coords[0], coords[1]]
        self.full_state = full_state
        self.distance_map = distance_map

    def get_distance(self, i, j):
        return self.distance_map.distances[(i, j)]

    def count_equal(self, val):
        return self.count_vals(lambda x: x == val)

    def count_vals(self, val_condition):
        return sum([1 for cell, _ in self.iterate()
                   if val_condition(cell)])

    def is_padding(self, state):
        return state == Neighborhood.PADDING_VAL

    def get_distance(self, i, j):
        return self.distance_map.distances[(i, j)]

    def iterate(self, exclude_center = True):
        subcell = self.view
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


class Neighborhood():

    PADDING_VAL = -1

    def __init__(self, radius, distance_map: DistanceMap = None):
        self.radius = radius
        if distance_map is None:
            self.distance_map = EuclideanDistanceMap(radius * 2 + 1)
        else:
            self.distance_map = distance_map

    def pad_state(self, state):
        return np.pad(state, self.radius, 'constant', constant_values=self.PADDING_VAL)

    def get_in_step(self, step, coords):
        return self.get(step.state, coords)

    def _get_square_neighborhood(self, padded_state, coords, overload_radius = None):
        i = coords[0]
        j = coords[1]

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

    def get(self, unpadded_state, coords, overload_radius = None):
        padded_state = self.pad_state(unpadded_state)
        square = np.copy(self._get_square_neighborhood(padded_state, coords, overload_radius=overload_radius))
        screened = self._screen_square(square)
        return NeighborhoodView(coords, screened, unpadded_state, self.distance_map)


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

    def __init__(self, _ = None):
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

    def __init__(self, _ = None):
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