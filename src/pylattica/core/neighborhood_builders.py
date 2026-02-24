from typing import Dict, List, Tuple

import numpy as np
import rustworkx as rx
from tqdm import tqdm
from scipy.spatial import cKDTree

from abc import abstractmethod

from .constants import LOCATION, SITE_ID
from .distance_map import EuclideanDistanceMap
from .neighborhoods import Neighborhood, StochasticNeighborhood, SiteClassNeighborhood
from .periodic_structure import PeriodicStructure
from .lattice import pbc_diff_cart


class NeighborhoodBuilder:
    """An abstract class to extend in order to implement a new type of
    NeighborhoodBuilder"""

    def get(self, struct: PeriodicStructure, site_class: str = None) -> Neighborhood:
        """Given a structure and a site class to build a neighborhood for,
        build the neighborhood.

        Parameters
        ----------
        struct : PeriodicStructure
            The structure for which the Neighborhood of every site should be
            calculated
        site_class : str, optional
            Specify a single class of sites to calculate the neighborhood for,
            by default None

        Returns
        -------
        Neighborhood
            _description_
        """
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
    """A helper class for building StochasticNeighborhoods - that is,
    neighborhoods for which one of several random neighbor sets is chosen
    each time a site's neighbors are requested."""

    def __init__(self, builders: List[NeighborhoodBuilder]):
        """Instantiates the StochasticNeighborhoodBuilder class.

        Parameters
        ----------
        builders : List[NeighborhoodBuilder]
            A list of builders which will give the neighborhoods that
            might be returned by the StochasticNeighborhood
        """
        self.builders = builders

    def get(self, struct: PeriodicStructure) -> Neighborhood:
        """For the provided structure, calculate the StochasticNeighborhood
        specified by the list of builders originally provided to this class.

        Parameters
        ----------
        struct : PeriodicStructure
            The structure for which the neighborhood should be calculated.

        Returns
        -------
        Neighborhood
            The resulting StochasticNeighborhood
        """
        return StochasticNeighborhood([b.get(struct) for b in self.builders])


class DistanceNeighborhoodBuilder(NeighborhoodBuilder):
    """This neighborhood builder creates neighbor connections between
    sites which are within some cutoff distance of eachother.

    Uses a KD-tree with periodic boundary conditions for O(n log n)
    performance instead of O(n²) brute force.
    """

    def __init__(self, cutoff: float):
        """Instantiates a DistanceNeighborhoodBuilder

        Parameters
        ----------
        cutoff : float
            The maximum distance at which two sites are considered neighbors.
        """
        self.cutoff = cutoff

    def get(self, struct: PeriodicStructure, site_class: str = None) -> Neighborhood:
        """Builds a Neighborhood from the provided structure using an optimized
        KD-tree algorithm with periodic boundary conditions.

        Parameters
        ----------
        struct : PeriodicStructure
            The structure for which the Neighborhood should be constructed.
        site_class : str, optional
            Specify a single class of sites to calculate the neighborhood for,
            by default None

        Returns
        -------
        Neighborhood
            The resulting Neighborhood
        """
        graph = rx.PyDiGraph()

        # Add all nodes first
        all_sites = struct.sites()
        for site in all_sites:
            graph.add_node(site[SITE_ID])

        # Get sites to process (either all or filtered by class)
        if site_class is None:
            sites_to_process = all_sites
        else:
            sites_to_process = struct.sites(site_class=site_class)

        n_sites = len(all_sites)

        # Extract locations and IDs as arrays for vectorized operations
        locations = np.array([s[LOCATION] for s in all_sites])
        site_ids = np.array([s[SITE_ID] for s in all_sites])

        # Convert to fractional coordinates for periodic KD-tree
        frac_coords = np.array(
            [struct.lattice.get_fractional_coords(loc) for loc in locations]
        )

        # Compute the maximum fractional radius that could correspond to
        # the Cartesian cutoff. For non-orthogonal lattices, we need to use
        # the maximum stretch factor of the inverse matrix (largest singular value).
        # This gives an upper bound; we then post-filter with exact Cartesian distances.
        inv_matrix = struct.lattice.inv_matrix
        max_stretch = np.linalg.norm(inv_matrix, ord=2)  # Largest singular value
        frac_cutoff = self.cutoff * max_stretch + 0.1  # Add margin

        # Determine periodicity - use boxsize for periodic dims
        periodic = struct.lattice.periodic
        dim = struct.lattice.dim

        # Build boxsize array: 1.0 for periodic dimensions, large value for non-periodic
        boxsize = np.array([1.0 if periodic[i] else 1e10 for i in range(dim)])

        # Wrap fractional coordinates to [0, 1) for periodic dimensions
        frac_coords_wrapped = frac_coords.copy()
        for i in range(dim):
            if periodic[i]:
                frac_coords_wrapped[:, i] = frac_coords_wrapped[:, i] % 1.0

        # Build KD-tree with periodic boundary conditions
        tree = cKDTree(frac_coords_wrapped, boxsize=boxsize)

        # Create index mapping from site_id to array index
        id_to_idx = {sid: idx for idx, sid in enumerate(site_ids)}

        # Process each site
        sites_to_process_ids = set(s[SITE_ID] for s in sites_to_process)

        for idx, site_id in enumerate(tqdm(site_ids, desc="Building neighborhood")):
            if site_id not in sites_to_process_ids:
                continue

            # Query tree for candidates within fractional cutoff
            curr_frac = frac_coords_wrapped[idx]
            candidate_indices = tree.query_ball_point(curr_frac, frac_cutoff)

            curr_loc = locations[idx]

            # Post-filter with exact Cartesian periodic distance
            for cand_idx in candidate_indices:
                if cand_idx == idx:
                    continue

                cand_loc = locations[cand_idx]
                dist = struct.lattice.cartesian_periodic_distance(cand_loc, curr_loc)

                if dist < self.cutoff:
                    graph.add_edge(site_id, site_ids[cand_idx], dist)

        return Neighborhood(graph)

    def get_neighbors(self, curr_site: Dict, struct: PeriodicStructure) -> List[Tuple]:
        """Builds a neighbor list for a single site. This method exists for
        compatibility but using get() directly is more efficient for building
        the full neighborhood graph.

        Parameters
        ----------
        curr_site : Dict
            The site to find neighbors for
        struct : PeriodicStructure
            The structure containing the sites

        Returns
        -------
        List[Tuple]
            List of (neighbor_id, distance) tuples
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

    Uses a KD-tree with periodic boundary conditions for O(n log n)
    performance instead of O(n²) brute force.
    """

    def __init__(self, inner_radius: float, outer_radius: float):
        """Instantiates an AnnularNeighborhoodBuilder

        Parameters
        ----------
        inner_radius : float
            The minimum at which two sites are considered neighbors.
        outer_radius : float
            The maximum distance at which two sites are considered neighbors.
        """
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def get(self, struct: PeriodicStructure, site_class: str = None) -> Neighborhood:
        """Builds a Neighborhood from the provided structure using an optimized
        KD-tree algorithm with periodic boundary conditions.

        Parameters
        ----------
        struct : PeriodicStructure
            The structure for which the Neighborhood should be constructed.
        site_class : str, optional
            Specify a single class of sites to calculate the neighborhood for,
            by default None

        Returns
        -------
        Neighborhood
            The resulting Neighborhood
        """
        graph = rx.PyDiGraph()

        # Add all nodes first
        all_sites = struct.sites()
        for site in all_sites:
            graph.add_node(site[SITE_ID])

        # Get sites to process (either all or filtered by class)
        if site_class is None:
            sites_to_process = all_sites
        else:
            sites_to_process = struct.sites(site_class=site_class)

        # Extract locations and IDs as arrays for vectorized operations
        locations = np.array([s[LOCATION] for s in all_sites])
        site_ids = np.array([s[SITE_ID] for s in all_sites])

        # Convert to fractional coordinates for periodic KD-tree
        frac_coords = np.array(
            [struct.lattice.get_fractional_coords(loc) for loc in locations]
        )

        # Compute the maximum fractional radius for the outer cutoff.
        # Use the maximum stretch factor of the inverse matrix for non-orthogonal lattices.
        inv_matrix = struct.lattice.inv_matrix
        max_stretch = np.linalg.norm(inv_matrix, ord=2)  # Largest singular value
        frac_cutoff = self.outer_radius * max_stretch + 0.1

        # Determine periodicity
        periodic = struct.lattice.periodic
        dim = struct.lattice.dim

        # Build boxsize array
        boxsize = np.array([1.0 if periodic[i] else 1e10 for i in range(dim)])

        # Wrap fractional coordinates to [0, 1) for periodic dimensions
        frac_coords_wrapped = frac_coords.copy()
        for i in range(dim):
            if periodic[i]:
                frac_coords_wrapped[:, i] = frac_coords_wrapped[:, i] % 1.0

        # Build KD-tree with periodic boundary conditions
        tree = cKDTree(frac_coords_wrapped, boxsize=boxsize)

        # Create index mapping
        sites_to_process_ids = set(s[SITE_ID] for s in sites_to_process)

        for idx, site_id in enumerate(tqdm(site_ids, desc="Building neighborhood")):
            if site_id not in sites_to_process_ids:
                continue

            # Query tree for candidates within fractional cutoff
            curr_frac = frac_coords_wrapped[idx]
            candidate_indices = tree.query_ball_point(curr_frac, frac_cutoff)

            curr_loc = locations[idx]

            # Post-filter with exact Cartesian periodic distance
            for cand_idx in candidate_indices:
                if cand_idx == idx:
                    continue

                cand_loc = locations[cand_idx]
                dist = pbc_diff_cart(
                    np.array(cand_loc),
                    np.array(curr_loc),
                    struct.lattice,
                )

                if self.inner_radius < dist < self.outer_radius:
                    graph.add_edge(site_id, site_ids[cand_idx], dist)

        return Neighborhood(graph)

    def get_neighbors(self, curr_site: Dict, struct: PeriodicStructure) -> List[Tuple]:
        """Builds a neighbor list for a single site. This method exists for
        compatibility but using get() directly is more efficient.

        Parameters
        ----------
        curr_site : Dict
            The site to find neighbors for
        struct : PeriodicStructure
            The structure containing the sites

        Returns
        -------
        List[Tuple]
            List of (neighbor_id, distance) tuples
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

    ```
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
    ```

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
    """A class which constructs the neighborhood of each site as a function
    of the class of that site."""

    def __init__(self, nb_builders: Dict[str, NeighborhoodBuilder]):
        """Instantiates the SiteClassNeighborhoodBuilder.

        Parameters
        ----------
        nb_builders : Dict[str, NeighborhoodBuilder]
            A mapping of site classes to the NeighborhoodBuilders which
            specify what neighborhood that class of sites should have.
        """
        self._builders = nb_builders

    def get(self, struct: PeriodicStructure) -> Neighborhood:
        """Constructs the neighborhood of every site in the provided
        structure, conditional on the class of each site.

        Parameters
        ----------
        struct : PeriodicStructure
            The structure for which the neigborhood should be calculated.

        Returns
        -------
        Neighborhood
            The resulting Neighborhood object.
        """
        nbhood_map = {}
        for sclass, builder in self._builders.items():
            nbhood = builder.get(struct, site_class=sclass)
            nbhood_map[sclass] = nbhood

        return SiteClassNeighborhood(struct, nbhood_map)
