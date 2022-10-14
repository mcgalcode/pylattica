from pylattica.core.constants import LOCATION
import pytest

from pylattica.core import PeriodicStructure

def test_can_instantiate_structure():
    assert PeriodicStructure((10, 10, 10)) is not None

def test_simple_structure_has_correct_sites(square_2x2_2D_grid):
    assert square_2x2_2D_grid.site_at((0.5, 0.5)) is not None
    assert square_2x2_2D_grid.site_at((0.5, 1.5)) is not None
    assert square_2x2_2D_grid.site_at((1.5, 0.5)) is not None
    assert square_2x2_2D_grid.site_at((1.5, 1.5)) is not None

def test_simple_structure_doesnt_have_incorrect_sites(square_2x2_2D_grid):
    assert square_2x2_2D_grid.site_at((0.5, 0.501)) is None

def test_simple_structure_has_periodic_sites(square_2x2_2D_grid):
    assert square_2x2_2D_grid.site_at((-1.5, -1.5)) is not None
    assert square_2x2_2D_grid.site_at((-1.5, 1.5)) is not None
    assert square_2x2_2D_grid.site_at((-1.5, 2.5)) is not None
    assert square_2x2_2D_grid.site_at((2.5, -0.5)) is not None

def test_sites_have_unaltered_lication(square_2x2_2D_grid):
    location = (0.5, 0.5)
    site = square_2x2_2D_grid.site_at(location)
    assert site[LOCATION][0] == 0.5
    assert site[LOCATION][1] == 0.5