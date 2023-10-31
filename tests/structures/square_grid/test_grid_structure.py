from pylattica.structures.square_grid.structure_builders import SimpleSquare2DStructureBuilder, SimpleSquare3DStructureBuilder

from pylattica.core.constants import LOCATION

def test_grid_has_points_as_expected(square_grid_2D_2x2):
    assert square_grid_2D_2x2.site_at((0, 0)) is not None
    assert square_grid_2D_2x2.site_at((0, 1)) is not None
    assert square_grid_2D_2x2.site_at((1, 0)) is not None
    assert square_grid_2D_2x2.site_at((1, 1)) is not None

def test_grid_has_points_as_expected_outside_cell(square_grid_2D_2x2):
    assert square_grid_2D_2x2.site_at((0, -1)) is not None
    assert square_grid_2D_2x2.site_at((0, 2)) is not None
    assert square_grid_2D_2x2.site_at((4, 0)) is not None
    assert square_grid_2D_2x2.site_at((12,31)) is not None

def test_grid_does_not_have_unexpected_points(square_grid_2D_2x2):
    assert square_grid_2D_2x2.site_at((0, 0.5)) is None
    assert square_grid_2D_2x2.site_at((0, -0.1)) is None
    assert square_grid_2D_2x2.site_at((1, 1.5)) is None
    assert square_grid_2D_2x2.site_at((-12.001, 1)) is None

def test_retrieved_site_has_correct_location(square_grid_2D_2x2):
    assert square_grid_2D_2x2.site_at((0, 0))[LOCATION] == (0,0)