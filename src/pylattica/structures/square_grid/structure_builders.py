from .lattice import SquareGridLattice2D, SquareGridLattice3D
from ...core.structure_builder import StructureBuilder

SITE_POSITION = 0


class SimpleSquare2DStructureBuilder(StructureBuilder):
    """A helper class for generating square 2D grid structures."""

    SITE_CLASS = "_A"

    def __init__(self):
        self.lattice = SquareGridLattice2D()

        self.motif = {
            SimpleSquare2DStructureBuilder.SITE_CLASS: [(SITE_POSITION, SITE_POSITION)],
        }


class SimpleSquare3DStructureBuilder(StructureBuilder):
    """A helper class for generating square 3D grid structures."""

    SITE_CLASS = "_A"

    def __init__(self):
        self.lattice = SquareGridLattice3D()
        self.motif = {
            SimpleSquare3DStructureBuilder.SITE_CLASS: [
                (SITE_POSITION, SITE_POSITION, SITE_POSITION)
            ],
        }
