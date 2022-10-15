from pylattica.core import periodic_structure
from pylattica.core.neighborhoods import Neighborhood, StochasticNeighborhood
from ..core.coordinate_utils import get_points_in_cube
from ..core.neighborhood_builders import NeighborhoodBuilder, StructureNeighborhoodBuilder, DistanceNeighborhoodBuilder
from .structure_builders import SimpleSquare2DStructureBuilder

class VonNeumannNbHood2DBuilder(StructureNeighborhoodBuilder):

    def __init__(self):
        super().__init__({
            SimpleSquare2DStructureBuilder.SITE_CLASS: [
                (1,0),
                (0, 1),
                (-1, 0),
                (0, -1)
            ]
        })

class VonNeumannNbHood2DBuilder(StructureNeighborhoodBuilder):

    def __init__(self):
        super().__init__({
            SimpleSquare2DStructureBuilder.SITE_CLASS: [
                (1,0),
                (0, 1),
                (-1, 0),
                (0, -1)
            ]
        })

class MooreNbHoodBuilder(StructureNeighborhoodBuilder):

    def __init__(self, size = 1, dim = 2):
        points = get_points_in_cube(-size, size + 1, dim)
        super().__init__({
            SimpleSquare2DStructureBuilder.SITE_CLASS: points
        })

class CircularNeighborhoodBuilder(DistanceNeighborhoodBuilder):

    pass

class PseudoHexagonalNeighborhoodBuilder(NeighborhoodBuilder):

    def __init__(self):
        motif_one = {
            SimpleSquare2DStructureBuilder.SITE_CLASS: [
                (1,0),
                (0, 1),
                (-1, 0),
                (0, -1),
                (1, 1),
                (-1, -1)
            ]
        }

        motif_two = {
            SimpleSquare2DStructureBuilder.SITE_CLASS: [
                (1,0),
                (0, 1),
                (-1, 0),
                (0, -1),
                (-1, 1),
                (1, -1)
            ]
        }
        self.builder_one = StructureNeighborhoodBuilder(motif_one)
        self.builder_two = StructureNeighborhoodBuilder(motif_two)

    def get(self, struct: periodic_structure) -> Neighborhood:
        return StochasticNeighborhood([
            self.builder_one.get(struct),
            self.builder_two.get(struct)
        ])

class PseudoPentagonalNeighborhoodBuilder(Neighborhood):

    def __init__(self):
        motifs = [{
            SimpleSquare2DStructureBuilder.SITE_CLASS: [
                (-1, 0),
                (1, 0),
                (-1, -1),
                (0, -1),
                (1, -1)
            ]
        }, {
            SimpleSquare2DStructureBuilder.SITE_CLASS: [
                (-1, 1),
                (0, 1),
                (1, 1),
                (-1, 0),
                (1, 0),
            ]
        }, {
            SimpleSquare2DStructureBuilder.SITE_CLASS: [
                (0, 1),
                (1, 1),
                (1, 0),
                (0, -1),
                (1, -1)
            ]
        }, {
            SimpleSquare2DStructureBuilder.SITE_CLASS: [
                (-1, 1),
                (0, 1),
                (-1, 0),
                (-1, -1),
                (0, -1),
            ]
        }]
        self.builders = [StructureNeighborhoodBuilder(m) for m in motifs]

    def get(self, struct: periodic_structure) -> Neighborhood:
        return StochasticNeighborhood([
            b.get(struct) for b in self.builders
        ])