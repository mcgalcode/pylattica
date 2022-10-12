import numpy as np
from pymatgen.core import Structure

def get_pt_in_range(bound, pt):
    return pt % bound

def get_periodic_point(bounds, pt):
    return tuple([get_pt_in_range(b, p) for b, p in zip(bounds, pt)])

def float_loc(loc):
    return tuple([float(coord) for coord in loc])

Vacant = "Vacant"
occupancy = 'occupancy'

class PeriodicStructure():

    def __init__(self, bounds, size, dimensionality):
        self.bounds = bounds
        self._sites = {}
        self._location_lookup = {}
        self.size = size
        self.dim = dimensionality

    def periodized_coords(self, location):
        float_coords = float_loc(location)
        return get_periodic_point(self.bounds, float_coords)

    def add_site(self, site_class, location):
        new_site_id = len(self._sites)
        periodized_coords = self.periodized_coords(location)
        # print(periodized_coords)

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

    def to_pymatgen(self, state):

        lattice_vecs = []
        for idx, b in enumerate(self.bounds):
            vec = []
            for i in range(idx):
                vec.append(0)
            vec.append(b)
            for i in range(idx, len(self.bounds) - 1):
                vec.append(0)
            lattice_vecs.append(vec)

        lattice_vecs = np.array(lattice_vecs)

        species = []
        sites = []

        for site in self.sites():
            site_state = state.get_site_state(site["id"])
            occ = site_state[occupancy]
            if occ is not Vacant:
                species.append(occ)
                frac_loc = [loc / b for loc, b in zip(site['location'], self.bounds)]
                sites.append(np.array(frac_loc))

        sites = np.array(sites)

        return Structure(lattice_vecs, species, sites)
