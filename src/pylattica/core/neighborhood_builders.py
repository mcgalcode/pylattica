from typing import Dict, List, Tuple

import numpy as np
import rustworkx as rx
from tqdm import tqdm

from abc import abstractmethod

from .constants import LOCATION, SITE_ID
from .distance_map import EuclideanDistanceMap
from .neighborhoods import Neighborhood, StochasticNeighborhood, SiteClassNeighborhood
from .periodic_structure import PeriodicStructure
from .lattice import pbc_diff_cart


class NeighborhoodBuilder:
    def get(self, struct: PeriodicStructure, site_class=None) -> Neighborhood:
        graph = rx.PyDiGraph()

        if site_class is None:
            sites = struct.sites()
        else:
            sites = struct.sites(site_class=site_class)

        for site in struct.sites():
            graph.add_node(site[SITE_ID])

        for curr_site in tqdm(sites):
            nbs = self.get_neighbors(curr_site, struct)
            for nb_id, weight in nbs:
                graph.add_edge(curr_site[SITE_ID], nb_id, weight)

        return Neighborhood(graph)

    @abstractmethod
    def get_neighbors(self, curr_site: Dict, struct: PeriodicStructure) -> List[Tuple]:
        pass  # pragma: no cover


class StochasticNeighborhoodBuilder(NeighborhoodBuilder):
    def __init__(self, builders):
        self.builders = builders

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
        curr_loc = curr_site[LOCATION]
        curr_id = curr_site[SITE_ID]
        for other_site in struct.sites():
            other_id = other_site[SITE_ID]
            if curr_id != other_id:
                dist = struct.lattice.cartesian_periodic_distance(
                    other_site[LOCATION],
                    curr_loc,
                )

                if dist < self.cutoff:
                    nbs.append((other_id, dist))

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

                if self.inner_radius < dist < self.outer_radius:
                    nbs.append((other_site[SITE_ID], dist))

        return nbs


class MotifNeighborhoodBuilder(NeighborhoodBuilder):
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

    def __init__(self, motif: List[List[float]]):
        """Instantiates the MotifNeighborhoodBuilder by a motif as described in
        the docstring for the class.

        Parameters
        ----------
        motif : Dict[str, List[List[float]]]
            See class docstring.
        """
        self._motif = motif

        self.distances = EuclideanDistanceMap(motif)

    def get_neighbors(self, curr_site: Dict, struct: PeriodicStructure) -> List[Tuple]:
        """Given a structure, constructs a NeighborGraph with site connections
        according to the motif.

        Parameters
        ----------
        struct : PeriodicStructure
            The structure for which a NeighborGraph should be constructed.

        Returns
        -------
        NeighborGraph
            The resulting NeighborGraph.
        """

        location = curr_site[LOCATION]
        nbs = []
        for neighbor_vec in self._motif:
            loc = tuple(s + n for s, n in zip(location, neighbor_vec))
            nb_id = struct.id_at(loc)
            if nb_id != curr_site[SITE_ID]:
                nbs.append((nb_id, self.distances.get_dist(neighbor_vec)))
        return nbs


class SiteClassNeighborhoodBuilder(NeighborhoodBuilder):
    def __init__(self, nb_builders: Dict[str, NeighborhoodBuilder]):
        self._builders = nb_builders

    def get(self, struct: PeriodicStructure) -> Neighborhood:
        nbhood_map = {}
        for sclass, builder in self._builders.items():
            nbhood = builder.get(struct, site_class=sclass)
            nbhood_map[sclass] = nbhood

        return SiteClassNeighborhood(struct, nbhood_map)
