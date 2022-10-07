from rxn_ca.core.coordinate_utils import get_points_in_box
from rxn_ca.core.periodic_structure import PeriodicStructure
import numpy as np

class Lattice():

    def __init__(self, vecs):
        self.vecs = vecs
        self.dim = len(vecs[0])
        self.vec_lengths = [np.linalg.norm(np.array(vec)) for vec in vecs]
        assert len(list(set([len(v) for v in vecs]))) == 1, "Lattice instantiated with vectors of unequal dimension"

    def build_from(self, num_cells: int, site_motif: dict) -> PeriodicStructure:
        bounds = np.array([l * num_cells for l in self.vec_lengths])
        struct = PeriodicStructure(bounds, num_cells, self.dim)

        vec_coeffs = get_points_in_box(0, num_cells, self.dim)
        vec_offset = np.array([0.1, 0.1, 0.1])
        for vec_coeff_set in vec_coeffs:
            point = np.zeros(self.dim)

            for vec_coeff, vec in zip(vec_coeff_set, self.vecs):
                np_vec = np.array(vec)
                point = point + (vec_coeff * np_vec)
            point = point + vec_offset

            for site_class, basis_vecs in site_motif.items():
                for vec in basis_vecs:

                    site_loc = tuple(point + np.array(vec))
                    struct.add_site(site_class, site_loc)

        return struct
