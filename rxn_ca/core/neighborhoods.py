import random
import typing
import numpy as np
import networkx as nx

from .distance_map import DistanceMap, EuclideanDistanceMap

from .periodic_structure import PeriodicStructure

class NeighborGraph():

    def __init__(self, graph: nx.Graph):
        self._graph = graph

    def neighbors_of(self, site_id):
        return self._graph.neighbors(site_id)

class StochasticNeighborhoodGraph():

    def __init__(self, graphs: typing.List[nx.Graph]):
        self._graphs = graphs

    def neighbors_of(self, site_id):
        selected_graph = random.choice(self._graphs)
        return selected_graph.neighbors(site_id)

class NeighborhoodView():

    def __init__(self, site, neighbors, site_state, full_state, distance_map) -> None:
        self.coords: typing.Tuple[int, int] = site['location']
        self.neighbors: np.array = neighbors
        self.site_state = site_state
        self.full_state: np.array = full_state
        self.dimension: int = len(self.coords)
        self.distance_map: DistanceMap = distance_map
        self.size: int = self.view_state.shape[0]

    def count_equal(self, state_key, state_val):
        return self.count_vals(lambda state: state[state_key] == state_val)

    def count_vals(self, val_condition):
        return sum([1 for cell, _ in self.iterate()
                   if val_condition(cell)])

    def get_distance(self, coords):
        return self.distance_map.distances[coords]

    def as_list(self, exclude_center = True):
        cells = []
        for cell in self.iterate(exclude_center=exclude_center):
            cells.append(cell)

        return cells

    def iterate(self, include_relative_coords = False):
        full_state = self.full_state
        for nb_site in self.neighbors:
            contents = full_state[nb_site['id']]
            loc = nb_site['location']
            relative_coords = (loc[0] - self.coords[0], loc[1] - self.coords[1])
            distance = self.get_distance(relative_coords)
            if include_relative_coords:
                yield contents, distance, relative_coords
            else:
                yield contents, distance

class StructureNeighborhoodSpec():

    def __init__(self, spec):
        self._spec = spec
        neighbor_locs = [loc for loclist in spec.values() for loc in loclist]
        self.distances = EuclideanDistanceMap(neighbor_locs)

    def get(self, struct: PeriodicStructure):
        graph = nx.Graph()

        for site in struct.sites():
            graph.add_node(site["id"])

        for site in struct.sites():
            site_class = site["site_class"]
            location = site["location"]
            site_class_neighbors = self._spec[site_class]
            for neigbor_vec in site_class_neighbors:
                nb_site = struct.site_at(np.array(location) + np.array(neigbor_vec))
                graph.add_edge(nb_site["id"], site["id"])

        return NeighborGraph(graph)
