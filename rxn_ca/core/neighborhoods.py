import itertools
from random import randint
import numpy as np

from .distance_map import EuclideanDistanceMap, ManhattanDistanceMap

class NeighborhoodView():

    def __init__(self, coords, view_state, full_state, distance_map) -> None:
        self.coords = tuple(coords)
        self.view_state = view_state
        self.center_value = full_state[self.coords]
        self.full_state = full_state
        self.dimension = len(view_state.shape)
        self.distance_map = distance_map
        self.size = self.view_state.shape[0]

    def count_equal(self, val):
        return self.count_vals(lambda x: x == val)

    def count_vals(self, val_condition):
        return sum([1 for cell, _ in self.iterate()
                   if val_condition(cell)])

    def is_padding(self, state):
        return state == Neighborhood.PADDING_VAL

    def get_distance(self, coords):
        return self.distance_map.distances[coords]

    def as_list(self, exclude_center = True):
        cells = []
        for cell in self.iterate(exclude_center=exclude_center):
            cells.append(cell)

        return cells

    def iterate(self, exclude_center = True):
        subcell = self.view_state
        center_coord = int(self.size / 2)
        for coords in itertools.product(range(self.size), repeat=self.dimension):
            contents = subcell[coords]
            if self.is_padding(contents):
                continue

            if exclude_center and all([coord == center_coord for coord in coords]):
                continue

            distance = self.get_distance(coords)
            yield subcell[coords], distance


class Neighborhood():

    PADDING_VAL = -1

    def __init__(self, radius, distance_map_class = EuclideanDistanceMap):
        self.radius = radius
        self.distances = {
            2: distance_map_class(radius * 2 + 1),
            3: distance_map_class(radius * 2 + 1, dimension=3),
        }

    def pad_state(self, state):
        return np.pad(state, self.radius, 'constant', constant_values=self.PADDING_VAL)

    def get_in_step(self, step, coords):
        return self.get(step.state, coords)

    def _get_untrimmed_neighborhood(self, state, coords, overload_radius = None):
        dimension = len(state.shape)
        if overload_radius is not None:
            radius = overload_radius
        else:
            radius = self.radius

        # We are accepting coordinates that don't consider padding,
        # So modify them to make 0,0 the first non-padding entry
        # i = i + radius
        # j = j + radius
        if dimension == 2:
            i = coords[0]
            j = coords[1]

            state_size = state.shape[0]

            curr_up = state_size + i - radius
            curr_down = state_size + i + radius + 1
            curr_left = state_size + j - radius
            curr_right = state_size + j + radius + 1

            tiled = np.tile(state, (3,3))
            return tiled[curr_up:curr_down, curr_left:curr_right]
        elif dimension == 3:
            i = coords[0]
            j = coords[1]
            k = coords[2]

            state_size = state.shape[0]

            curr_up = state_size + i - radius
            curr_down = state_size + i + radius + 1
            curr_left = state_size + j - radius
            curr_right = state_size + j + radius + 1
            curr_in = state_size + k - radius
            curr_out = state_size + k + radius + 1

            tiled = np.tile(state, (3,3,3))
            return tiled[curr_up:curr_down, curr_left:curr_right, curr_in:curr_out]

    def get(self, state, coords, overload_radius = None):
        dim = len(state.shape)
        untrimmed = np.copy(self._get_untrimmed_neighborhood(state, coords, overload_radius=overload_radius))
        if dim == 2:
            screened = self._screen_square(untrimmed)
        elif dim == 3:
            screened = self._screen_cube(untrimmed)

        return NeighborhoodView(coords, screened, state, self.distances[dim])

    def _screen_cube(self, untrimmed):
        raise NotImplementedError("This neighborhood does not support 3-dimensional states")

class MooreNeighborhood(Neighborhood):

    def _screen_square(self, square):
        return square

    def _screen_cube(self, cube):
        return cube


class VonNeumannNeighborhood(Neighborhood):

    def __init__(self, radius, distance_map_class = ManhattanDistanceMap):
        super().__init__(radius, distance_map_class)

    def _screen_square(self, square):
        dimension = len(square.shape)
        sl = square.shape[0]
        for coords in itertools.product(range(sl), repeat=dimension):
            if self.distances[dimension].distances[coords] > self.radius:
                square[coords] = self.PADDING_VAL
        return square

    def _screen_cube(self, cube):
        dimension = len(cube.shape)
        sl = cube.shape[0]
        for coords in itertools.product(range(sl), repeat=dimension):
            if self.distances[dimension].distances[coords] > self.radius:
                cube[coords] = self.PADDING_VAL
        return cube

class CircularNeighborhood(Neighborhood):

    def __init__(self, radius, distance_map_class = EuclideanDistanceMap):
        super().__init__(radius, distance_map_class)

    def _screen_square(self, square):
        dimension = len(square.shape)
        sl = square.shape[0]
        for coords in itertools.product(range(sl), repeat=dimension):
            if self.distances[dimension].distances[coords] > self.radius:
                square[coords] = self.PADDING_VAL
        return square

    def _screen_cube(self, cube):
        dimension = len(cube.shape)
        sl = cube.shape[0]
        for coords in itertools.product(range(sl), repeat=dimension):
            if self.distances[dimension].distances[coords] > self.radius:
                cube[coords] = self.PADDING_VAL
        return cube

class PseudoHexagonal(Neighborhood):

    def __init__(self, _ = EuclideanDistanceMap):
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

    def __init__(self, _ = EuclideanDistanceMap):
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