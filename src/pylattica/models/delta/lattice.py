from ...core.lattice import Lattice
from .consts import TET_SITE, OCT_SITE

class DeltaLattice(Lattice):

    def __init__(self):
        lattice_vecs = [
            (2,0,0),
            (0,2,0),
            (0,0,2)
        ]
        super().__init__(lattice_vecs)

class DeltaStructureBuilder():

    def build(self, size):
        lattice = DeltaLattice()

        motif = {
            OCT_SITE: [
                (0,0,0),
                (1,1,0),
                (1,0,1),
                (0,1,1)
            ],
            TET_SITE: [
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 1.5),
                (1/2, 3/2, 1/2),
                (3/2, 1/2, 1/2),
                (1/2, 3/2, 3/2),
                (3/2, 1/2, 3/2),
                (3/2, 3/2, 1/2),
                (3/2, 3/2, 3/2),
            ]
        }

        return lattice.build_from(size, motif)
