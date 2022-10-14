from ..core.coordinate_utils import get_points_in_cube
from ..core.neighborhoods import StructureNeighborhoodBuilder
from .structure_builders import SimpleSquare2DStructureBuilder

class VonNeumannNbHood2DSpec(StructureNeighborhoodBuilder):

    def __init__(self):
        super().__init__({
            SimpleSquare2DStructureBuilder.SITE_CLASS: [
                (1,0),
                (0, 1),
                (-1, 0),
                (0, -1)
            ]
        })

class MooreNbHoodSpec(StructureNeighborhoodBuilder):

    def __init__(self, size = 1, dim = 2):
        points = get_points_in_cube(-size, size + 1, dim)
        super().__init__({
            SimpleSquare2DStructureBuilder.SITE_CLASS: points
        })

# class CircularNeighborhood(Neighborhood):

#     def __init__(self, radius, distance_map_class = EuclideanDistanceMap):
#         super().__init__(radius, distance_map_class)

#     def _screen_square(self, square):
#         dimension = len(square.shape)
#         sl = square.shape[0]
#         for coords in itertools.product(range(sl), repeat=dimension):
#             if self.distances[dimension].distances[coords] > self.radius:
#                 square[coords] = self.PADDING_VAL
#         return square

#     def _screen_cube(self, cube):
#         dimension = len(cube.shape)
#         sl = cube.shape[0]
#         for coords in itertools.product(range(sl), repeat=dimension):
#             if self.distances[dimension].distances[coords] > self.radius:
#                 cube[coords] = self.PADDING_VAL
#         return cube

# class PseudoHexagonal(Neighborhood):

#     def __init__(self, _ = EuclideanDistanceMap):
#         super().__init__(1)

#     def _screen_square(self, square):
#         sl = square.shape[0]
#         to_exclude = randint(0,1)
#         highest_idx = sl - 1
#         if to_exclude == 0:
#             square[0][0] = self.PADDING_VAL
#             square[highest_idx][highest_idx] = self.PADDING_VAL
#         else:
#             square[highest_idx][0] = self.PADDING_VAL
#             square[0][highest_idx] = self.PADDING_VAL

#         return square

# class PseudoPentagonal(Neighborhood):

#     def __init__(self, _ = EuclideanDistanceMap):
#         super().__init__(1)

#     def _screen_square(self, square):
#         sl = square.shape[0]
#         to_exclude = randint(0,3)
#         for i in range(sl):
#             for j in range(sl):
#                 if to_exclude == 0 and i == 0:
#                     square[i][j] = self.PADDING_VAL
#                 if to_exclude == 1 and i == 2:
#                     square[i][j] = self.PADDING_VAL
#                 if to_exclude == 2 and j == 2:
#                     square[i][j] = self.PADDING_VAL
#                 if to_exclude == 3 and j == 0:
#                     square[i][j] = self.PADDING_VAL

#         return square