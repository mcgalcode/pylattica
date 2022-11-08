from typing import List, Tuple

import numpy as np

from .coordinate_utils import get_points_in_box
from .periodic_structure import PeriodicStructure


class Lattice:
    """A lattice is specified by it's lattice vectors. This class can then
    be used to create PeriodicStructure instances which are filled with
    a given motif. The usage flow for this class is:

    1) Define your lattice by specifying the lattice vectors and instatiating this class
    2) Define a motif of sites, which is a dictionary mapping each site class to the
    basis vectors that point to the site locations
    3) Use build_from to generate a periodic structure by repeating the unit cell
    defined by this lattice in every direction.

    Attributes
    ----------

    vecs : np.ndarray
        The lattice vectors defining the unit cell of this lattice.
    """

    def __init__(self, vecs: List[Tuple[float]]):
        """Initializes a lattice with the vectors defining its unit cell provided.
        The dimension of the lattice is inferred from the dimension of the lattice vectors.

        Parameters
        ----------
        vecs : List[Tuple[float]]
            A list of vectors establishing the unit cell of the lattice. Any dimension is accepted.
        """
        self.vecs = np.array(vecs)
        self.dim = len(vecs[0])
        self._vec_lengths = [np.linalg.norm(np.array(vec)) for vec in vecs]
        assert (
            len(list(set(len(v) for v in vecs))) == 1
        ), "Lattice instantiated with vectors of unequal dimension"

    def build_from(self, num_cells: List[int], site_motif: dict) -> PeriodicStructure:
        """Builds a PeriodicStructure lattice by repeating the unit cell num_cell times
        in each dimension. For instance, to build a structure that has 2 unit cells in
        each direction (and itself is three dimensional), the num_cells parameter should be

        [2, 2, 2]

        Lattice sites must also be specified by the site_motif parameter. This allows
        specification of which sites are where in the unit cell. For instance, if my 2D
        lattice has two types of sites, A and B, and each type exists in two different
        places, I might use the following site_motif:

        {
            "A": [
                [0.2, 0.2],
                [0.4, 0.4]
            ],
            "B": [
                [0.6, 0.6],
                [0.8, 0.8]
            ]
        }

        Parameters
        ----------
        num_cells : List[int]
            As described above, a list of the number of repetitions of the unit cell
            in each direction.
        site_motif : dict
            A dictionary mapping string site classes to lists of the locations within
            the unit cell at which a site of that type exists.

        Returns
        -------
        PeriodicStructure
            The structure resulting from the lattice tiling and motif filling specified.
        """
        bounds = np.array([l * n for n, l in zip(num_cells, self._vec_lengths)])
        struct = self.initialize_structure(bounds)

        vec_coeffs = get_points_in_box([0 for _ in range(self.dim)], num_cells)

        for vec_coeff_set in vec_coeffs:
            point = np.zeros(self.dim)

            for vec_coeff, vec in zip(vec_coeff_set, self.vecs):
                point = point + (vec_coeff * vec)

            for site_class, basis_vecs in site_motif.items():
                for vec in basis_vecs:

                    site_loc = tuple(point + np.array(vec))
                    struct.add_site(site_class, site_loc)

        return struct

    def initialize_structure(self, bounds: np.ndarray) -> PeriodicStructure:
        """Instantiates the structure that is being built and filled out by this lattice.

        Parameters
        ----------
        bounds : np.ndarray
            The extent of the structure along each direction

        Returns
        -------
        PeriodicStructure
            The correctly sized (but as of yet unfilled) resulting structure.
        """
        return PeriodicStructure(bounds)
