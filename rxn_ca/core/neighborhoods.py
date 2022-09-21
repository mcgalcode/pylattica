import random
import typing
import numpy as np
import networkx as nx
from tqdm import tqdm

from .distance_map import DistanceMap, EuclideanDistanceMap

from .periodic_structure import PeriodicStructure

def distance(x0, x1, dimensions):
    delta = np.abs(x0 - x1)
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

class NeighborGraph():

    def __init__(self, graph: nx.Graph):
        self._graph = graph

    def neighbors_of(self, site_id, include_weights=False):
        nbs = self._graph.neighbors(site_id)
        if include_weights:
            weighted_nbs = [(nb_id, self._graph.edges[site_id, nb_id]['weight']) for nb_id in nbs]
            return weighted_nbs
        else:
            return list(nbs)

class StochasticNeighborhoodGraph():

    def __init__(self, graphs: typing.List[nx.Graph]):
        self._graphs = graphs

    def neighbors_of(self, site_id):
        selected_graph = random.choice(self._graphs)
        return list(selected_graph.neighbors(site_id))

import numpy

def distance(x0, x1, dimensions):
    delta = numpy.abs(x0 - x1)
    delta = numpy.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return numpy.sqrt((delta ** 2).sum(axis=-1))

class DistanceNeighborhoodSpec():

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def get(self, struct: PeriodicStructure):
        graph = nx.Graph()
        dimensions = np.array(struct.bounds)

        all_sites = struct.sites()
        for curr_site in tqdm(all_sites):
            for other_site in struct.sites():
                if curr_site['id'] != other_site['id']:
                    dist = distance(np.array(other_site['location']), np.array(curr_site['location']), dimensions)
                    if dist < self.cutoff:
                        graph.add_edge(curr_site["id"], other_site["id"])

        return NeighborGraph(graph)


class StructureNeighborhoodSpec():

    def __init__(self, spec):
        self._spec = spec
        all_neighbor_vecs = []
        for _, neighbor_vecs in spec.items():
            all_neighbor_vecs = all_neighbor_vecs + neighbor_vecs

        self.distances = EuclideanDistanceMap(all_neighbor_vecs)

        neighbor_locs = [loc for loclist in spec.values() for loc in loclist]
        self.distances = EuclideanDistanceMap(neighbor_locs)

    def get(self, struct: PeriodicStructure):
        graph = nx.Graph()

        for site in struct.sites():
            graph.add_node(site["id"])

        all_sites = struct.sites()
        print("Constructing neighborhood graph")
        for i in tqdm(range(len(all_sites))):
            site = all_sites[i]
            site_class = site["site_class"]
            location = site["location"]
            site_class_neighbors = self._spec[site_class]
            edges = []
            for neighbor_vec in site_class_neighbors:
                loc = [s + n for s, n in zip(location, neighbor_vec)]
                # loc = np.array(location) + np.array(neighbor_vec)
                nb_site = struct.site_at(loc)
                if nb_site['id'] != site['id']:
                    edges.append((nb_site["id"], site["id"], self.distances.get_dist(neighbor_vec)))
            graph.add_weighted_edges_from(edges)

        return NeighborGraph(graph)

