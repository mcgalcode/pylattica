import random
from typing import Dict, List
import numpy as np
import networkx as nx
from tqdm import tqdm

from .distance_map import EuclideanDistanceMap
from .periodic_structure import PeriodicStructure
from .constants import SITE_ID, SITE_CLASS, LOCATION

def distance(x0: np.ndarray, x1: np.ndarray, dimensions: np.ndarray) -> float:
    """Returns the distance between two points under periodic boundary
    conditions. Can handle many points at once if x0 and x1 are more than 1 dimensional.

    https://stackoverflow.com/questions/11108869/optimizing-python-distance-calculation-while-accounting-for-periodic-boundary-co

    There are some other options for this here:

    https://pymatgen.org/pymatgen.core.lattice.html#pymatgen.core.lattice.Lattice.get_all_distances
    https://pymatgen.org/pymatgen.core.lattice.html#pymatgen.core.lattice.Lattice.get_distance_and_image

    Parameters
    ----------
    x0 : np.ndarray
        The first point, or array of points
    x1 : np.ndarray
        The second point, or array of points
    dimensions : np.ndarray
        The periodic bounds

    Returns
    -------
    float
        The distance between two points
    """    
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

class NeighborGraph():
    """A specific Neighborhood. An instance of this classes corresponds
    to a particular SimulationState. It stores a map of each site_id
    in the SimulationState to the IDs of the sites which are it's neighbors.
    """    

    def __init__(self, graph: nx.Graph):
        """Instantiates a NeighborhoodGraph."""        
        self._graph = graph

    def neighbors_of(self, site_id: int, include_weights: bool = False) -> list[int]:
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
            weighted_nbs = [(nb_id, self._graph.edges[site_id, nb_id]['weight']) for nb_id in nbs]
            return weighted_nbs
        else:
            return list(nbs)

class StochasticNeighborhoodGraph():
    """A NeighborhoodGraph for stochastic neighborhoods.
    """    

    def __init__(self, graphs: List[nx.Graph]):
        self._graphs = graphs

    def neighbors_of(self, site_id):
        selected_graph = random.choice(self._graphs)
        return list(selected_graph.neighbors(site_id))


class DistanceNeighborhoodBuilder():
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

    def get(self, struct: PeriodicStructure) -> NeighborGraph:
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
        graph = nx.Graph()
        dimensions = np.array(struct.bounds)

        all_sites = struct.sites()
        for curr_site in tqdm(all_sites):
            for other_site in struct.sites():
                if curr_site[SITE_ID] != other_site[SITE_ID]:
                    dist = distance(np.array(other_site['location']), np.array(curr_site['location']), dimensions)
                    if dist < self.cutoff:
                        graph.add_edge(curr_site[SITE_ID], other_site[SITE_ID])

        return NeighborGraph(graph)


class StructureNeighborhoodBuilder():
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

    def get(self, struct: PeriodicStructure) -> NeighborGraph:
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
        graph = nx.Graph()

        for site in struct.sites():
            graph.add_node(site[SITE_ID])

        all_sites = struct.sites()
        print("Constructing neighborhood graph")
        for i in tqdm(range(len(all_sites))):
            site = all_sites[i]
            site_class = site[SITE_CLASS]
            location = site[LOCATION]
            site_class_neighbors = self._spec[site_class]
            edges = []
            for neighbor_vec in site_class_neighbors:
                loc = [s + n for s, n in zip(location, neighbor_vec)]
                nb_site = struct.site_at(loc)
                if nb_site[SITE_ID] != site[SITE_ID]:
                    edges.append((nb_site[SITE_ID], site[SITE_ID], self.distances.get_dist(neighbor_vec)))
            graph.add_weighted_edges_from(edges)

        return NeighborGraph(graph)

