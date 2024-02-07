import numpy as np
from ...core.coordinate_utils import get_points_in_cube
from ...core.neighborhood_builders import (
    StochasticNeighborhoodBuilder,
    MotifNeighborhoodBuilder,
    DistanceNeighborhoodBuilder,
)


class VonNeumannNbHood2DBuilder(MotifNeighborhoodBuilder):
    """A helper class for generating von Neumann type neighborhoods in square 2D structures."""

    def __init__(self, size=1):
        """Constructs the VonNeumannNbHood2DBuilder.

        Parameters
        ----------
        size : int, optional
            The size of the neighborhood, the traditional von Neumann neighborhood would have a size of 1
        """
        points = get_points_in_cube(-size, size + 1, 2)

        filtered_points = []
        for point in points:
            if sum(np.abs(p) for p in point) <= size:
                filtered_points.append(point)

        super().__init__(filtered_points)


class VonNeumannNbHood3DBuilder(MotifNeighborhoodBuilder):
    """A helper class for generating von Neumann type neighborhoods in square 3D structures."""

    def __init__(self, size: int):
        """Constructs the VonNeumannNbHood3D Builder

        Parameters
        ----------
        size : int
            The size of the neighborhood.
        """
        points = get_points_in_cube(-size, size + 1, 3)

        filtered_points = []
        for point in points:
            if sum(np.abs(p) for p in point) <= size:
                filtered_points.append(point)

        super().__init__(filtered_points)


class MooreNbHoodBuilder(MotifNeighborhoodBuilder):
    """A helper class for generating Moore type neighborhoods in square structures."""

    def __init__(self, size=1, dim=2):
        """Constructs the MooreNbHoodBuilder

        Parameters
        ----------
        size : int, optional
            The size of the neighborhood, by default 1
        dim : int, optional
            The dimension of the structures to which this builder will apply, by default 2
        """
        points = get_points_in_cube(-size, size + 1, dim)
        super().__init__(points)


class CircularNeighborhoodBuilder(DistanceNeighborhoodBuilder):
    """A semantic class for generating Circular neighborhoods in any structure."""


class PseudoHexagonalNeighborhoodBuilder2D(StochasticNeighborhoodBuilder):
    def __init__(self):
        motifs = [
            [
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
                (1, 1),
                (-1, -1),
            ],
            [
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
                (-1, 1),
                (1, -1),
            ],
        ]
        super().__init__([MotifNeighborhoodBuilder(m) for m in motifs])


class PseudoHexagonalNeighborhoodBuilder3D(StochasticNeighborhoodBuilder):
    """A helper class for generating Pseudohexagonal neighborhoods in 3D."""

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
            [
                *common_neighbors,
                (-1, -1, 1),
                (1, 1, -1),
            ],
            [
                *common_neighbors,
                (-1, 1, 1),
                (1, -1, -1),
            ],
            [
                *common_neighbors,
                (1, 1, 1),
                (-1, -1, -1),
            ],
            [
                *common_neighbors,
                (1, -1, 1),
                (-1, 1, -1),
            ],
        ]
        super().__init__([MotifNeighborhoodBuilder(m) for m in motifs])


class PseudoPentagonalNeighborhoodBuilder(StochasticNeighborhoodBuilder):
    """A helper class for generating Pseudopentagonal neighborhoods in 2D."""

    def __init__(self):
        motifs = [
            [
                (-1, 0),
                (1, 0),
                (-1, -1),
                (0, -1),
                (1, -1),
            ],
            [
                (-1, 1),
                (0, 1),
                (1, 1),
                (-1, 0),
                (1, 0),
            ],
            [
                (0, 1),
                (1, 1),
                (1, 0),
                (0, -1),
                (1, -1),
            ],
            [
                (-1, 1),
                (0, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
            ],
        ]
        self.builders = [MotifNeighborhoodBuilder(m) for m in motifs]
