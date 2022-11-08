import pytest

from pylattica.core.periodic_structure import PeriodicStructure
from pylattica.square_grid.neighborhoods import (
    CircularNeighborhoodBuilder,
    MooreNbHoodBuilder,
)
from pylattica.square_grid.structure_builders import SimpleSquare2DStructureBuilder


def test_moore_neighborhood(square_grid_2D_2x2: PeriodicStructure):
    spec = MooreNbHoodBuilder()
    nb_hood = spec.get(square_grid_2D_2x2)

def test_circular_neighborhood():
    struct = SimpleSquare2DStructureBuilder().build(20)
    
    nb_builder = CircularNeighborhoodBuilder(3)
    nbh = nb_builder.get(struct)
    nbs = nbh.neighbors_of(0)
    assert len(nbs) == 24

    nb_builder = CircularNeighborhoodBuilder(2)
    nbh = nb_builder.get(struct)
    nbs = nbh.neighbors_of(0)
    assert len(nbs) == 8
