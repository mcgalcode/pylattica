from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import rustworkx as rx
from tqdm import tqdm

from .constants import LOCATION, SITE_CLASS, SITE_ID
from .distance_map import EuclideanDistanceMap
from .neighborhoods import Neighborhood, StochasticNeighborhood
from .periodic_structure import PeriodicStructure
from .lattice import pbc_diff_cart


class NeighborhoodBuilder(ABC):
    def get(self, struct: PeriodicStructure) -> Neighborhood:
        graph = rx.PyGraph()

        all_sites = struct.sites()

        for site in struct.sites():
            graph.add_node(site[SITE_ID])

        for curr_site in tqdm(all_sites):
            nbs = self.get_neighbors(curr_site, struct)
            for nb_id, weight in nbs:
                graph.add_edge(curr_site[SITE_ID], nb_id, weight)

        return Neighborhood(graph)

    @abstractmethod
    def get_neighbors(self, curr_site: Dict, struct: PeriodicStructure) -> List[Tuple]:
        pass  # pragma: no cover


class StochasticNeighborhoodBuilder(ABC):
    def get(self, struct: PeriodicStructure) -> Neighborhood:
        return StochasticNeighborhood([b.get(struct) for b in self.builders])


class DistanceNeighborhoodBuilder(NeighborhoodBuilder):
    """This neighborhood builder creates neighbor connections between
    sites which are within some cutoff distance of eachother.
    """

    def __init__(self, cutoff: float):
        """Instantiates a DistanceNeighborhoodBuilder

        Parameters
        ----------
        cutoff : float
            The maximum distance at which two sites are considered neighbors.
        """
        self.cutoff = cutoff

    def get_neighbors(self, curr_site: Dict, struct: PeriodicStructure) -> List[Tuple]:
        """Builds a NeighborGraph from the provided structure according
        to the cutoff distance of this Builder.

        Parameters
        ----------
        struct : PeriodicStructure
            The structure from which a NeighborGraph should be constructed.

        Returns
        -------
        NeighborGraph
            The resulting NeighborGraph
        """
        nbs = []
        for other_site in struct.sites():
            if curr_site[SITE_ID] != other_site[SITE_ID]:
                dist = pbc_diff_cart(
                    np.array(other_site[LOCATION]),
                    np.array(curr_site[LOCATION]),
                    struct.lattice,
                )
                print(dist)
                if dist < self.cutoff:
                    nbs.append((other_site[SITE_ID], dist))

        return nbs


class AnnularNeighborhoodBuilder(NeighborhoodBuilder):
    """This neighborhood builder creates neighbor connections between
    sites which are within a ring-shaped region around eachother. This region
    is specified by a minimum (inner radius) and maximum (outer radius) distance.
    """

    def __init__(self, inner_radius: float, outer_radius: float):
        """Instantiates a DistanceNeighborhoodBuilder

        Parameters
        ----------
        inner_radius : float
            The minimum at which two sites are considered neighbors.
        outer_radius : float
            The maximum distance at which two sites are considered neighbors.
        """
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def get_neighbors(self, curr_site: Dict, struct: PeriodicStructure) -> List[Tuple]:
        """Builds a NeighborGraph from the provided structure according
        to the cutoff distance of this Builder.

        Parameters
        ----------
        struct : PeriodicStructure
            The structure from which a NeighborGraph should be constructed.

        Returns
        -------
        NeighborGraph
            The resulting NeighborGraph
        """
        nbs = []
        for other_site in struct.sites():
            if curr_site[SITE_ID] != other_site[SITE_ID]:
                dist = pbc_diff_cart(
                    np.array(other_site[LOCATION]),
                    np.array(curr_site[LOCATION]),
                    struct.lattice,
                )

                if dist < self.outer_radius and dist > self.inner_radius:
                    nbs.append((other_site[SITE_ID], dist))

        return nbs


class StructureNeighborhoodBuilder(NeighborhoodBuilder):
    """This NeighborhoodBuilder constructs NeighborGraphs with connections between
    points that are separated by one of a set of specific offset vectors.

    For example, consider a 2D structure with two types of sites, A and B. Each A
    site is connected to two other A sites, one offset by 1 unit in each of the positive
    and negative x directions. Each A site is also connected to two B sites, one offset
    by one unit in each of the positive and negative y directions, then the spec
    parameter for this arrangement would look as follows.

    {
        "A": [
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
        ],
        "B": [
            [0, 1],
            [0, -1],
        ]
    }

    Note that there is reciprocity here between the A and B sites. The A sites
    list B sites as their neighbors, and the B sites list A sites as their neighbors.
    """

    def __init__(self, spec: Dict[str, List[List[float]]]):
        """Instantiates the StructureNeighborhoodBuilder by a spec as described in
        the docstring for the class.

        Parameters
        ----------
        spec : Dict[str, List[List[float]]]
            See class docstring.
        """
        self._spec = spec

        neighbor_locs = [loc for loclist in spec.values() for loc in loclist]
        self.distances = EuclideanDistanceMap(neighbor_locs)

    def get_neighbors(self, curr_site: Dict, struct: PeriodicStructure) -> List[Tuple]:
        """Given a structure, constructs a NeighborGraph with site connections
        according to the spec.

        Parameters
        ----------
        struct : PeriodicStructure
            The structure for which a NeighborGraph should be constructed.

        Returns
        -------
        NeighborGraph
            The resulting NeighborGraph.
        """

        site_class = curr_site[SITE_CLASS]
        location = curr_site[LOCATION]
        site_class_neighbors = self._spec[site_class]
        nbs = []
        for neighbor_vec in site_class_neighbors:
            loc = tuple(s + n for s, n in zip(location, neighbor_vec))
            nb_site = struct.site_at(loc)
            if nb_site[SITE_ID] != curr_site[SITE_ID]:
                nbs.append((nb_site[SITE_ID], self.distances.get_dist(neighbor_vec)))

        return nbs
