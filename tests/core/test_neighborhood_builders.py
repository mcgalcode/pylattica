import pytest

import numpy as np
import math

from pylattica.core import Lattice, PeriodicStructure
from pylattica.core.neighborhood_builders import (
    DistanceNeighborhoodBuilder,
    MotifNeighborhoodBuilder,
    AnnularNeighborhoodBuilder,
)
from pylattica.structures.square_grid.structure_builders import (
    SimpleSquare2DStructureBuilder,
)


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
    struct = SimpleSquare2DStructureBuilder().build((5, 5))

    builder = AnnularNeighborhoodBuilder(1.2, 2.1)
    nb_hood = builder.get(struct)

    nbs = nb_hood.neighbors_of(1, True)
    assert len(nbs) == 8

    for nb, dist in nbs:
        assert dist > 1.2
        assert dist < 2.1


def test_struct_nb_hood_builder(square_grid_2D_4x4):
    builder = MotifNeighborhoodBuilder([(0, 1), (0, -1)])

    nbhood = builder.get(square_grid_2D_4x4)

    nbs_w_dists = nbhood.neighbors_of(1, include_weights=True)

    assert len(nbs_w_dists) == 2
    assert len(set([nb[0] for nb in nbs_w_dists])) == 2

    for nb_id, nb_dist in nbs_w_dists:
        assert nb_dist == 1.0


def test_motif_nb_builder_non_periodic_no_none_neighbors():
    # A non-periodic lattice means edge sites have motif offsets that point
    # off the grid, where there is no site. id_at returns None for those, and
    # those Nones must not be added as neighbors.
    lattice = Lattice([(1, 0), (0, 1)], periodic=False)
    struct = PeriodicStructure.build_from(lattice, [3, 3], {"A": [(0, 0)]})

    builder = MotifNeighborhoodBuilder([(0, 1), (0, -1), (1, 0), (-1, 0)])

    # Building the neighborhood must not crash on the off-grid (None) neighbors
    nbhood = builder.get(struct)

    # No site should report a None neighbor
    for site in struct.sites():
        for nb_id in nbhood.neighbors_of(site["_site_id"]):
            assert nb_id is not None

    # The corner site (0, 0) has only two in-bounds neighbors: (0, 1) and (1, 0)
    corner_id = struct.id_at((0, 0))
    assert len(nbhood.neighbors_of(corner_id)) == 2
