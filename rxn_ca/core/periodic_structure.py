import math

import networkx as nx

def get_pt_in_range(bound, pt):
    if pt > bound:
        return math.remainder(pt, bound)
    elif pt < 0:
        return bound + math.remainder(pt, bound)
    else:
        return pt

def get_periodic_point(bounds, pt):
    return tuple([get_pt_in_range(b, p) for b, p in zip(bounds, pt)])

def float_loc(loc):
    return tuple([float(coord) for coord in loc])

class PeriodicStructure():

    def __init__(self, size, dimensionality):
        self._sites = {}
        self._location_lookup = {}
        self._graph = nx.Graph()
        self.size = size
        self.dim = dimensionality

    def periodized_coords(self, location):
        float_coords = float_loc(location)
        bounds = tuple([self.size for _ in range(len(location))])
        return get_periodic_point(bounds, float_coords)

    def add_site(self, site_class, location):
        new_site_id = len(self._sites)
        periodized_coords = self.periodized_coords(location)
        assert self._location_lookup.get(periodized_coords, None) is None, "That site is already occupied"

        self._sites[new_site_id] = {
            "site_class": site_class,
            "location": periodized_coords,
            "id": new_site_id,
            "state": {}
        }

        self._location_lookup[periodized_coords] = {
            "id": new_site_id
        }
        return new_site_id

    def site_at(self, location):
        periodized_coords = self.periodized_coords(location)
        record = self._location_lookup.get(periodized_coords)
        if record is not None:
            return self.get_site(record["id"])
        else:
            return None

    def get_site(self, site_id):
        return self._sites.get(site_id)

    def sites(self, site_class = None):
        all_sites = list(self._sites.values())
        if site_class is None:
            return all_sites
        else:
            return [site for site in all_sites if site['site_class'] == site_class]
