import numpy as np
from pylattica.core import periodic_structure
from pylattica.core.neighborhoods import Neighborhood, StochasticNeighborhood
from ...core.coordinate_utils import get_points_in_cube
from ...core.neighborhood_builders import (
    NeighborhoodBuilder,
    StochasticNeighborhoodBuilder,
    MotifNeighborhoodBuilder,
    DistanceNeighborhoodBuilder,
)
from .structure_builders import SimpleSquare2DStructureBuilder


class VonNeumannNbHood2DBuilder(MotifNeighborhoodBuilder):
    def __init__(self, size=1):
        points = get_points_in_cube(-size, size + 1, 2)

        filtered_points = []
        for point in points:
            if sum(np.abs(p) for p in point) <= size:
                filtered_points.append(point)

        super().__init__({SimpleSquare2DStructureBuilder.SITE_CLASS: filtered_points})


class VonNeumannNbHood3DBuilder(MotifNeighborhoodBuilder):
    def __init__(self, size):
        points = get_points_in_cube(-size, size + 1, 3)

        filtered_points = []
        for point in points:
            if sum(np.abs(p) for p in point) <= size:
                filtered_points.append(point)

        super().__init__({SimpleSquare2DStructureBuilder.SITE_CLASS: filtered_points})


class MooreNbHoodBuilder(MotifNeighborhoodBuilder):
    def __init__(self, size=1, dim=2):
        points = get_points_in_cube(-size, size + 1, dim)
        super().__init__({SimpleSquare2DStructureBuilder.SITE_CLASS: points})


class CircularNeighborhoodBuilder(DistanceNeighborhoodBuilder):
    pass


class PseudoHexagonalNeighborhoodBuilder2D(StochasticNeighborhoodBuilder):
    def __init__(self):
        motifs = [
            {
                SimpleSquare2DStructureBuilder.SITE_CLASS: [
                    (1, 0),
                    (0, 1),
                    (-1, 0),
                    (0, -1),
                    (1, 1),
                    (-1, -1),
                ]
            },
            {
                SimpleSquare2DStructureBuilder.SITE_CLASS: [
                    (1, 0),
                    (0, 1),
                    (-1, 0),
                    (0, -1),
                    (-1, 1),
                    (1, -1),
                ]
            },
        ]
        self.builders = [MotifNeighborhoodBuilder(m) for m in motifs]


class PseudoHexagonalNeighborhoodBuilder3D(StochasticNeighborhoodBuilder):
    def __init__(self):
        common_neighbors = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
        motifs = [
            {
                SimpleSquare2DStructureBuilder.SITE_CLASS: [
                    *common_neighbors,
                    (-1, -1, 1),
                    (1, 1, -1),
                ]
            },
            {
                SimpleSquare2DStructureBuilder.SITE_CLASS: [
                    *common_neighbors,
                    (-1, 1, 1),
                    (1, -1, -1),
                ]
            },
            {
                SimpleSquare2DStructureBuilder.SITE_CLASS: [
                    *common_neighbors,
                    (1, 1, 1),
                    (-1, -1, -1),
                ]
            },
            {
                SimpleSquare2DStructureBuilder.SITE_CLASS: [
                    *common_neighbors,
                    (1, -1, 1),
                    (-1, 1, -1),
                ]
            },
        ]
        self.builders = [MotifNeighborhoodBuilder(m) for m in motifs]


class PseudoPentagonalNeighborhoodBuilder(StochasticNeighborhoodBuilder):
    def __init__(self):
        motifs = [
            {
                SimpleSquare2DStructureBuilder.SITE_CLASS: [
                    (-1, 0),
                    (1, 0),
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                ]
            },
            {
                SimpleSquare2DStructureBuilder.SITE_CLASS: [
                    (-1, 1),
                    (0, 1),
                    (1, 1),
                    (-1, 0),
                    (1, 0),
                ]
            },
            {
                SimpleSquare2DStructureBuilder.SITE_CLASS: [
                    (0, 1),
                    (1, 1),
                    (1, 0),
                    (0, -1),
                    (1, -1),
                ]
            },
            {
                SimpleSquare2DStructureBuilder.SITE_CLASS: [
                    (-1, 1),
                    (0, 1),
                    (-1, 0),
                    (-1, -1),
                    (0, -1),
                ]
            },
        ]
        self.builders = [MotifNeighborhoodBuilder(m) for m in motifs]