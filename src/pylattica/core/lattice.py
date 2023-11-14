from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from .constants import OFFSET_PRECISION
import math


def periodize(frac_coords, periodic=True):
    if not isinstance(periodic, tuple):
        periodic = [periodic for _ in frac_coords]

    return frac_coords - np.floor(frac_coords) * np.array(periodic, dtype=int)


def pbc_diff_frac_vec(fcoords1: ArrayLike, fcoords2: ArrayLike, periodic):
    """Returns the 'fractional distance' between two coordinates taking into
    account periodic boundary conditions. (from pymatgen)

    Args:
        fcoords1: First set of fractional coordinates. e.g., [0.5, 0.6,
            0.7] or [[1.1, 1.2, 4.3], [0.5, 0.6, 0.7]]. It can be a single
            coord or any array of coords.
        fcoords2: Second set of fractional coordinates.
        pbc: a tuple defining the periodic boundary conditions along the three
            axis of the lattice.

    Returns:
        Fractional distance. Each coordinate must have the  operty that
        abs(a) <= 0.5. Examples:
        pbc_diff([0.1, 0.1, 0.1], [0.3, 0.5, 0.9]) = [-0.2, -0.4, 0.2]
        pbc_diff([0.9, 0.1, 1.01], [0.3, 0.5, 0.9]) = [-0.4, -0.4, 0.11]
    """
    fdist = np.subtract(fcoords1, fcoords2)
    return fdist - np.round(fdist) * periodic


def pbc_diff_cart(cart_coords1: ArrayLike, cart_coords2: ArrayLike, lattice: Lattice):
    """Returns the Cartesian distance between two coordinates taking into
    account periodic boundary conditions. (from pymatgen)

    Args:
        fcoords1: First set of fractional coordinates. e.g., [0.5, 0.6,
            0.7] or [[1.1, 1.2, 4.3], [0.5, 0.6, 0.7]]. It can be a single
            coord or any array of coords.
        fcoords2: Second set of fractional coordinates.
        pbc: a tuple defining the periodic boundary conditions along the three
            axis of the lattice.

    Returns:
        Fractional distance. Each coordinate must have the property that
        abs(a) <= 0.5. Examples:
        pbc_diff([0.1, 0.1, 0.1], [0.3, 0.5, 0.9]) = [-0.2, -0.4, 0.2]
        pbc_diff([0.9, 0.1, 1.01], [0.3, 0.5, 0.9]) = [-0.4, -0.4, 0.11]
    """
    fcoords1 = lattice.get_fractional_coords(cart_coords1)
    fcoords2 = lattice.get_fractional_coords(cart_coords2)
    frac_dist = pbc_diff_frac_vec(fcoords1, fcoords2, lattice.periodic)
    return np.round(
        np.linalg.norm(lattice.get_cartesian_coords(frac_dist)), OFFSET_PRECISION
    )


class Lattice:
    """A lattice is specified by it's lattice vectors. This class can then
    be used to create PeriodicStructure instances which are filled with
    a given motif. The usage flow for this class is:

    1) Define your lattice by specifying the lattice vectors and instantiating this class
    2) Define a motif of sites, which is a dictionary mapping each site class to the
    basis vectors that point to the site locations
    3) Use build_from to generate a periodic structure by repeating the unit cell
    defined by this lattice in every direction.

    Attributes
    ----------

    vecs : np.ndarray
        The lattice vectors defining the unit cell of this lattice.
    """

    def __init__(self, vecs: List[Tuple[float]], periodic=True):
        """Initializes a lattice with the vectors defining its unit cell provided.
        The dimension of the lattice is inferred from the dimension of the lattice vectors.

        Parameters
        ----------
        vecs : List[Tuple[float]]
            A list of vectors establishing the unit cell of the lattice. Any dimension is accepted.
        """

        # This set up of matrix and inversion matrix is taken from pymatgen
        # I would prefer to use the pymatgen lattice directly, but it is hardcoded
        # to utilize 3 dimensions - i.e. no game of life, 2D Ising, etc

        self.vecs = np.array(vecs)

        if not isinstance(periodic, tuple):
            self.periodic = tuple(periodic for _ in vecs)
        else:
            self.periodic = periodic

        self._periodic_bool = np.array(periodic, dtype=int)

        dim = int(math.sqrt(len(np.array(self.vecs).flatten())))
        mat = np.array(self.vecs, dtype=np.float64).reshape((dim, dim))
        mat.setflags(write=False)

        self._matrix: np.ndarray = mat
        self._inv_matrix: np.ndarray | None = None

        self.dim = len(vecs[0])
        self.vec_lengths = [np.linalg.norm(np.array(vec)) for vec in vecs]
        assert (
            len(list(set(len(v) for v in vecs))) == 1
        ), "Lattice instantiated with vectors of unequal dimension"

    @property
    def matrix(self) -> np.ndarray:
        """Copy of matrix representing the Lattice. (Taken from pymatgen)"""
        return self._matrix

    @property
    def inv_matrix(self) -> np.ndarray:
        """Inverse of lattice matrix. (Taken from pymatgen)"""
        if self._inv_matrix is None:
            self._inv_matrix = np.linalg.inv(self._matrix)
            self._inv_matrix.setflags(write=False)
        return self._inv_matrix

    def get_cartesian_coords(self, fractional_coords: ArrayLike) -> np.ndarray:
        """Returns the Cartesian coordinates given fractional coordinates. (taken from pymatgen)

        Args:
            fractional_coords (dimx1 array): Fractional coords.

        Returns:
            Cartesian coordinates
        """
        return np.dot(fractional_coords, self._matrix)

    def get_fractional_coords(self, cart_coords: ArrayLike) -> np.ndarray:
        """Returns the fractional coordinates given Cartesian coordinates. (taken from pymatgen)

        Args:
            cart_coords (dimx1 array): Cartesian coords.

        Returns:
            Fractional coordinates.
        """
        return np.dot(cart_coords, self.inv_matrix)

    def get_periodized_cartesian_coords(self, cart_coords: ArrayLike) -> np.ndarray:
        frac = self.get_fractional_coords(cart_coords)
        return self.get_cartesian_coords(periodize(frac, self.periodic))

    def get_scaled_lattice(self, num_cells: ArrayLike) -> Lattice:
        return Lattice(
            np.array([v * amt for amt, v in zip(num_cells, self.vecs)]), self.periodic
        )

    def cartesian_periodic_distance(self, loc1, loc2):
        return pbc_diff_cart(
            loc1,
            loc2,
            self,
        )
