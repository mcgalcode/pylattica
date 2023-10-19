from abc import abstractmethod, ABC
from .lattice import SquareGridLattice2D, SquareGridLattice3D
from ..core import PeriodicStructure

SITE_POSITION = 0


class StructureBuilder(ABC):
    @abstractmethod
    def build(self, size):
        pass


class SimpleSquare2DStructureBuilder(StructureBuilder):
    SITE_CLASS = "_A"

    def __init__(self):
        self.lattice = SquareGridLattice2D()

        self._motif = {
            SimpleSquare2DStructureBuilder.SITE_CLASS: [(SITE_POSITION, SITE_POSITION)],
        }

    def build(self, size):
        return PeriodicStructure.build_from(self.lattice, (size, size), self._motif)


class SimpleSquare3DStructureBuilder(StructureBuilder):
    SITE_CLASS = "_A"

    def __init__(self):
        self.lattice = SquareGridLattice3D()
        self._motif = {
            SimpleSquare3DStructureBuilder.SITE_CLASS: [
                (SITE_POSITION, SITE_POSITION, SITE_POSITION)
            ],
        }

    def build(self, size):
        return PeriodicStructure.build_from(self.lattice, (size, size, size), self._motif)
