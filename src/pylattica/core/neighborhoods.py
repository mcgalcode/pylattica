import random
from abc import ABC, abstractmethod
from typing import List
import rustworkx as rx


class AbstractNeighborhood(ABC):
    @abstractmethod
    def neighbors_of(self, site_id: int, include_weights: bool = False) -> List[int]:
        pass


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


class StochasticNeighborhood(AbstractNeighborhood):
    """A NeighborhoodGraph for stochastic neighborhoods."""

    def __init__(self, neighborhoods: List[Neighborhood]):
        self._neighborhoods = neighborhoods

    def neighbors_of(self, site_id, include_weights: bool = False) -> List[int]:
        selected_neighborhood = random.choice(self._neighborhoods)
        return list(
            selected_neighborhood.neighbors_of(site_id, include_weights=include_weights)
        )
