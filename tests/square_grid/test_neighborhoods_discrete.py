import pytest

from pylattica.square_grid.neighborhoods import (
    CircularNeighborhoodBuilder,
    MooreNbHoodBuilder,
    VonNeumannNbHood2DBuilder,
    PseudoHexagonalNeighborhoodBuilder2D,
    PseudoPentagonalNeighborhoodBuilder,
    PseudoHexagonalNeighborhoodBuilder3D,
    VonNeumannNbHood3DBuilder
)
from pylattica.core.constants import SITE_ID
from pylattica.square_grid.structure_builders import SimpleSquare2DStructureBuilder, SimpleSquare3DStructureBuilder

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

def test_pseudo_hexagonal_nb_hood():
    struct = SimpleSquare2DStructureBuilder().build(10)
    
    nb_builder = PseudoHexagonalNeighborhoodBuilder2D()
    nbh = nb_builder.get(struct)
    nbs = nbh.neighbors_of(0)
    assert len(nbs) == 6

def test_pseudo_pentagonal_nb_hood():
    struct = SimpleSquare2DStructureBuilder().build(10)
    
    nb_builder = PseudoPentagonalNeighborhoodBuilder()
    nbh = nb_builder.get(struct)
    nbs = nbh.neighbors_of(0)
    assert len(nbs) == 8

def test_pseudo_hexagonal_nb_3d_hood():
    struct = SimpleSquare3DStructureBuilder().build(10)
    
    nb_builder = PseudoHexagonalNeighborhoodBuilder3D()
    nbh = nb_builder.get(struct)
    nbs = nbh.neighbors_of(0)
    assert len(nbs) == 8

def test_von_neumann_nb_3d_hood():
    struct = SimpleSquare3DStructureBuilder().build(10)
    
    nb_builder = VonNeumannNbHood3DBuilder(1)
    nbh = nb_builder.get(struct)
    nbs = nbh.neighbors_of(0)
    assert len(nbs) == 6
