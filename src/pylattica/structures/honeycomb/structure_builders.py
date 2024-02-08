from ...core.structure_builder import StructureBuilder
from .lattice import RhombohedralLattice


class HoneycombTilingBuilder(StructureBuilder):
    """A helper class for generating honeycomb tiling structures."""

    SITE_CLASS = "A"

    def __init__(self):
        self.lattice = RhombohedralLattice()
        self.frac_coords = True
        self.motif = {HoneycombTilingBuilder.SITE_CLASS: [(1 / 2, 1 / 2)]}
