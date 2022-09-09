from rxn_ca.core.coordinate_utils import get_points_in_box
from rxn_ca.core.periodic_structure import PeriodicStructure
import numpy as np

class Lattice():

    def __init__(self, vecs):
        self.vecs = vecs
        self.dim = len(vecs[0])
        assert len(list(set([len(v) for v in vecs]))) == 1, "Lattice instantiated with vectors of unequal dimension"

    def build_from(self, state_size: int, site_motif: dict) -> PeriodicStructure:
        struct = PeriodicStructure(state_size, self.dim)

        points = get_points_in_box(0, state_size, self.dim)
        for point in points:
            for site_class, basis_vec in site_motif.items():
                site_loc = tuple(np.array(point) + np.array(basis_vec))
                struct.add_site(site_class, site_loc)
        return struct
