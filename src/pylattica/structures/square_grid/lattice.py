from ...core import Lattice


class SquareGridLattice2D(Lattice):
    def __init__(self):
        lattice_vecs = [(0, 1), (1, 0)]
        super().__init__(lattice_vecs)


class SquareGridLattice3D(Lattice):
    def __init__(self):
        lattice_vecs = [(0, 1, 0), (1, 0, 0), (0, 0, 1)]
        super().__init__(lattice_vecs)
