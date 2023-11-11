from typing import Tuple, Union

from abc import ABC
from .periodic_structure import PeriodicStructure
from .lattice import Lattice


class StructureBuilder(ABC):
    frac_coords = False

    def __init__(self, lattice: Lattice, motif):
        self.lattice = lattice
        self.motif = motif

    def build(self, size: Union[Tuple[int], int]):
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
