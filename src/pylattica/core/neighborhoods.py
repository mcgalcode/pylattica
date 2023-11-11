import random
from abc import ABC, abstractmethod
from typing import List, Dict
import rustworkx as rx

from .periodic_structure import PeriodicStructure


class AbstractNeighborhood(ABC):
    @abstractmethod
    def neighbors_of(self, site_id: int, include_weights: bool = False) -> List[int]:
        pass  # pragma: no cover


class Neighborhood(AbstractNeighborhood):
    """A specific Neighborhood. An instance of this classes corresponds
    to a particular SimulationState. It stores a map of each site_id
    in the SimulationState to the IDs of the sites which are it's neighbors.
    """

    def __init__(self, graph: rx.PyGraph):
        """Instantiates a NeighborhoodGraph."""
        self._graph = graph

    def neighbors_of(self, site_id: int, include_weights: bool = False) -> List[int]:
        """Retrieves a list of the IDs of the sites which are neighbors of the
        provided site. Optionally includes the weights of the connections to those
        neighbors.

        Parameters
        ----------
        site_id : int
            The site for which neighbors should be retrieved
        include_weights : bool, optional
            Whether or not weights if the neighbor connections should
            be included, by default False

        Returns
        -------
        list[int]
            Either a list of site IDs, or a list of tuples of (site ID, connection weight)
        """
        nbs = self._graph.neighbors(site_id)
        if include_weights:
            weighted_nbs = [
                (nb_id, self._graph.get_edge_data(site_id, nb_id)) for nb_id in nbs
            ]
            return weighted_nbs

        return list(nbs)


class MultiNeighborhood(AbstractNeighborhood):
    def neighbors_of(self, site_id, include_weights: bool = False) -> List[int]:
        selected_neighborhood = self._get_nbhood(site_id)

        if selected_neighborhood is None:
            return []
        else:
            return selected_neighborhood.neighbors_of(
                site_id, include_weights=include_weights
            )


class StochasticNeighborhood(MultiNeighborhood):
    """A NeighborhoodGraph for stochastic neighborhoods."""

    def __init__(self, neighborhoods: List[Neighborhood]):
        self._neighborhoods = neighborhoods

    def _get_nbhood(self, _) -> List[int]:
        return random.choice(self._neighborhoods)


class SiteClassNeighborhood(MultiNeighborhood):
    """A Neighborhood that distinguished neighbors of sites based on their class"""

    def __init__(
        self, structure: PeriodicStructure, neighborhoods: Dict[str, Neighborhood]
    ):
        self._struct = structure
        self._nbhoods = neighborhoods

    def _get_nbhood(self, site_id: int) -> List[int]:
        site_class = self._struct.site_class(site_id)
        return self._nbhoods.get(site_class)
