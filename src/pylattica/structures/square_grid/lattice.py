from ...core import Lattice


class SquareGridLattice2D(Lattice):
    """A helper class for generating square 2D grid lattices."""

    def __init__(self):
        lattice_vecs = [(0, 1), (1, 0)]
        super().__init__(lattice_vecs)


class SquareGridLattice3D(Lattice):
    """A helper class for generating square 3D grid lattices."""

    def __init__(self):
        lattice_vecs = [(0, 1, 0), (1, 0, 0), (0, 0, 1)]
        super().__init__(lattice_vecs)
