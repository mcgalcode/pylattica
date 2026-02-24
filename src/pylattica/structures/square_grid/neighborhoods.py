import numpy as np
import rustworkx as rx
from ...core.coordinate_utils import get_points_in_cube
from ...core.neighborhood_builders import (
    StochasticNeighborhoodBuilder,
    MotifNeighborhoodBuilder,
    DistanceNeighborhoodBuilder,
    NeighborhoodBuilder,
)
from ...core.neighborhoods import Neighborhood
from ...core.periodic_structure import PeriodicStructure


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


class VonNeumannNbHood3DBuilder(NeighborhoodBuilder):
    """Optimized Von Neumann neighborhood builder for simple cubic 3D grids.

    Uses direct index math instead of coordinate lookups, providing
    much faster performance for large grids.

    For a cubic grid of size n³:
    - Site i is at position (x, y, z) where x = i % n, y = (i // n) % n, z = i // n²
    - Neighbor at offset (dx, dy, dz) has ID: ((x+dx) % n) + ((y+dy) % n) * n + ((z+dz) % n) * n²
    """

    def __init__(self, size: int):
        """Constructs the VonNeumannNbHood3DBuilder.

        Parameters
        ----------
        size : int
            The size of the neighborhood (Manhattan distance).
        """
        # Generate Von Neumann neighborhood offsets (excluding origin)
        points = get_points_in_cube(-size, size + 1, 3)
        self._offsets = [
            tuple(point) for point in points
            if sum(np.abs(p) for p in point) <= size and any(p != 0 for p in point)
        ]
        # Precompute distances for edge weights
        self._distances = {
            offset: np.sqrt(sum(p**2 for p in offset))
            for offset in self._offsets
        }
        # Cache for grid size (computed once per structure)
        self._cached_n = None
        self._cached_n_sites = None

    def _get_grid_size(self, struct: PeriodicStructure) -> int:
        """Infer grid size n from structure (cached)."""
        n_sites = len(struct.site_ids)
        if n_sites != self._cached_n_sites:
            n = int(round(n_sites ** (1/3)))
            if n ** 3 != n_sites:
                raise ValueError(f"Structure has {n_sites} sites, not a perfect cube.")
            self._cached_n = n
            self._cached_n_sites = n_sites
        return self._cached_n

    def get_neighbors(self, curr_site: dict, struct: PeriodicStructure) -> list:
        """Get neighbors of a site using fast index math.

        Parameters
        ----------
        curr_site : dict
            Site dictionary with 'id' key
        struct : PeriodicStructure
            The structure (used to infer grid size)

        Returns
        -------
        list
            List of (neighbor_id, distance) tuples
        """
        from ...core.constants import SITE_ID

        n = self._get_grid_size(struct)
        site_id = curr_site[SITE_ID]

        # Convert site ID to (x, y, z) coordinates
        x = site_id % n
        y = (site_id // n) % n
        z = site_id // (n * n)

        neighbors = []
        for dx, dy, dz in self._offsets:
            # Compute neighbor coordinates with periodic boundary conditions
            nx = (x + dx) % n
            ny = (y + dy) % n
            nz = (z + dz) % n

            # Convert back to site ID
            neighbor_id = nx + ny * n + nz * (n * n)
            neighbors.append((neighbor_id, self._distances[(dx, dy, dz)]))

        return neighbors

    def get(self, struct: PeriodicStructure, site_class: str = None) -> Neighborhood:
        """Build neighborhood graph using vectorized index math.

        This override provides much faster performance than the base class
        by computing all edges in bulk using numpy operations.
        """
        n_sites = len(struct.site_ids)
        n = self._get_grid_size(struct)

        graph = rx.PyDiGraph()

        # Add all nodes at once
        graph.add_nodes_from(range(n_sites))

        # Vectorized computation: create coordinate arrays for all sites
        site_ids = np.arange(n_sites, dtype=np.int64)
        x = site_ids % n
        y = (site_ids // n) % n
        z = site_ids // (n * n)

        # Collect all edges across all offsets, then add in one batch
        all_edges = []
        for dx, dy, dz in self._offsets:
            # Compute neighbor coordinates with periodic boundary conditions
            nx = (x + dx) % n
            ny = (y + dy) % n
            nz = (z + dz) % n

            # Convert to neighbor site IDs
            neighbor_ids = nx + ny * n + nz * (n * n)
            weight = self._distances[(dx, dy, dz)]

            # Stack source, dest, weight as columns and extend
            # Use numpy operations to avoid Python loop overhead
            edge_data = np.column_stack([site_ids, neighbor_ids])
            all_edges.extend((int(s), int(d), weight) for s, d in edge_data)

        # Add all edges in one batch
        graph.extend_from_weighted_edge_list(all_edges)

        return Neighborhood(graph)


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
