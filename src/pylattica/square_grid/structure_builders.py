from abc import abstractmethod
from .lattice import SquareGridLattice2D, SquareGridLattice3D

SITE_POSITION = 0


class StructureBuilder:
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
        return self.lattice.build_from((size, size), self._motif)


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
        return self.lattice.build_from((size, size, size), self._motif)
