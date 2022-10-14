from pylattica.square_grid.structure_builders import SimpleSquare2DStructureBuilder
import pytest


def test_creates_square_grid():
    struct = SimpleSquare2DStructureBuilder().build(4)
    assert struct.site_at((0,0)) is not None