import pytest

from pylattica.structures.honeycomb import HoneycombTilingBuilder, HoneycombNeighborhoodBuilder

def test_basic_tiling_neighborhood():
    # this test is for a hexagonal tiling - each point
    # has 6 neighbors

    struct = HoneycombTilingBuilder().build((3,3))

    nbhood = HoneycombNeighborhoodBuilder().get(struct)

    nbs = nbhood.neighbors_of(0, True)
    assert len(nbs) == 6