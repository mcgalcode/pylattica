

from abc import abstractmethod

from .lattice import SquareGridLattice2D, SquareGridLattice3D


class StructureBuilder():

    @abstractmethod
    def build(self, size):
        pass


class SimpleSquare2DStructureBuilder(StructureBuilder):

    SITE_CLASS = "_A"

    def __init__(self):
        self.lattice = SquareGridLattice2D()
        self._motif = {
            SimpleSquare2DStructureBuilder.SITE_CLASS: [(0.5, 0.5)],
        }

    def build(self, size):
        return self.lattice.build_from(size, self._motif)

class SimpleSquare3DStructureBuilder(StructureBuilder):

    SITE_CLASS = "_A"

    def __init__(self):
        self.lattice = SquareGridLattice3D()
        self._motif = {
            SimpleSquare3DStructureBuilder.SITE_CLASS: [(0.5, 0.5, 0.5)],
        }

    def build(self, size):
        return self.lattice.build_from(size, self._motif)
