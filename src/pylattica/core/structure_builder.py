from typing import Tuple, Union, Dict, List

from abc import ABC
from .periodic_structure import PeriodicStructure
from .lattice import Lattice


class StructureBuilder(ABC):
    """An abstract class for building structures out of motifs and lattices. In general.
    this class works by tiling space to an extent specified, and then filling that
    space according to the motif (mapping of site types to locations within the unit cell)
    provided.
    """

    frac_coords = False

    def __init__(self, lattice: Lattice, motif: Dict[str, List[Tuple]]):
        """Instantiates a StructureBuilder using a Lattice and a motif.

        Parameters
        ----------
        lattice : Lattice
            The Lattice with which space will be tiled.
        motif : Dict[str, List[Tuple]]
            The motif with which sites will be placed inside the structure. These motifs
            are given by a mapping of site type (str) to list of locations inside
            the unit cell that that site type exists.
        """
        self.lattice = lattice
        self.motif = motif

    def build(self, size: Union[Tuple[int], int]) -> PeriodicStructure:
        """Builds a structure with the provided size.

        Parameters
        ----------
        size : Union[Tuple[int], int]
            Either an integer or a tuple of integers specifying the extent
            of the desired structure in each dimension.

        Returns
        -------
        PeriodicStructure
            The resulting structure.

        Raises
        ------
        ValueError
            If the size is a tuple that is not of length equal to the lattice
            dimension, a value error is thrown. In other words, you must specify
            size along each dimension of the lattice.
        """
        if isinstance(size, tuple):
            if not len(size) == self.lattice.dim:
                raise ValueError(
                    f"Desired structure dimensions, {size}, does not match "
                    "dimensionality of lattice: {self.lattice.dim}"
                )
        else:
            size = [size for _ in range(self.lattice.dim)]

        return PeriodicStructure.build_from(
            self.lattice, size, self.motif, frac_coords=self.frac_coords
        )
