from ...core import Lattice

import math

HONEYCOMB_SIDE_LENGTH = 1
ROOT_3 = math.sqrt(3)


class RhombohedralLattice(Lattice):
    """A helper class for generating rhombohedral lattices."""

    def __init__(self):
        lattice_vecs = [
            (HONEYCOMB_SIDE_LENGTH, 0),
            (HONEYCOMB_SIDE_LENGTH / 2, ROOT_3 * HONEYCOMB_SIDE_LENGTH / 2),
        ]
        super().__init__(lattice_vecs)
