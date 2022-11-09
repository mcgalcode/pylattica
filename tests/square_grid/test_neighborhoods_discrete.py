import pytest

from pylattica.core.periodic_structure import PeriodicStructure
from pylattica.square_grid.neighborhoods import (
    CircularNeighborhoodBuilder,
    MooreNbHoodBuilder,
    VonNeumannNbHood2DBuilder
)
from pylattica.core.constants import SITE_ID
from pylattica.square_grid.structure_builders import SimpleSquare2DStructureBuilder

def test_von_neumann_neighborhood():
    struct = SimpleSquare2DStructureBuilder().build(10)

    spec = VonNeumannNbHood2DBuilder()
    nb_hood = spec.get(struct)

    site = struct.site_at((5,5))
    nbs = nb_hood.neighbors_of(site[SITE_ID])
    assert len(nbs) == 4


def test_moore_neighborhood():
    struct = SimpleSquare2DStructureBuilder().build(10)

    spec = MooreNbHoodBuilder()
    nb_hood = spec.get(struct)

    site = struct.site_at((5,5))
    nbs = nb_hood.neighbors_of(site[SITE_ID])
    assert len(nbs) == 8

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
