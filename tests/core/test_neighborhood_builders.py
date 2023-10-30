import pytest

import numpy as np
import math

from pylattica.core.neighborhood_builders import DistanceNeighborhoodBuilder, StructureNeighborhoodBuilder, AnnularNeighborhoodBuilder
from pylattica.structures.square_grid.structure_builders import SimpleSquare2DStructureBuilder

def test_distance_nb_builder(square_grid_2D_4x4):

    builder = DistanceNeighborhoodBuilder(1.01)
    nbhood = builder.get(square_grid_2D_4x4)
    neighbors = nbhood.neighbors_of(1)

    assert len(neighbors) == 4
    assert len(set(neighbors)) == 4

    nbs_w_dists = nbhood.neighbors_of(1, include_weights=True)

    for _, nb_dist in nbs_w_dists:
        assert nb_dist == 1.0

    
    builder = DistanceNeighborhoodBuilder(1.5)
    nbhood = builder.get(square_grid_2D_4x4)
    neighbors = nbhood.neighbors_of(1)

    assert len(neighbors) == 8
    assert len(set(neighbors)) == 8

    nbs_w_dists = nbhood.neighbors_of(1, include_weights=True)
    root_2 = math.sqrt(2)
    for _, nb_dist in nbs_w_dists:
        assert nb_dist == 1.0 or np.isclose(nb_dist, root_2, 0.01)

def test_annular_nb_hood_builder():
    struct = SimpleSquare2DStructureBuilder().build((5,5))

    builder = AnnularNeighborhoodBuilder(1.2, 2.1)
    nb_hood = builder.get(struct)

    nbs = nb_hood.neighbors_of(1, True)
    assert len(nbs) == 8

    for nb, dist in nbs:
        assert dist > 1.2
        assert dist < 2.1


def test_struct_nb_hood_builder(square_grid_2D_4x4):
    site_class = SimpleSquare2DStructureBuilder.SITE_CLASS
    builder = StructureNeighborhoodBuilder({
        site_class: [(0, 1), (0, -1)]
    })

    nbhood = builder.get(square_grid_2D_4x4)

    nbs_w_dists = nbhood.neighbors_of(1, include_weights=True)

    assert len(nbs_w_dists) == 2
    assert len(set([nb[0] for nb in nbs_w_dists])) == 2
    
    for nb_id, nb_dist in nbs_w_dists:
        assert nb_dist == 1.0
