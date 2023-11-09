from pylattica.core.constants import LOCATION
import pytest

from pylattica.core import PeriodicStructure

def test_can_instantiate_structure(square_lattice):
    assert PeriodicStructure(square_lattice) is not None

def test_simple_structure_has_correct_sites(square_2x2_2D_grid_in_test: PeriodicStructure):
    assert square_2x2_2D_grid_in_test.site_at((0.5, 0.5)) is not None
    assert square_2x2_2D_grid_in_test.site_at((0.5, 1.5)) is not None
    assert square_2x2_2D_grid_in_test.site_at((1.5, 0.5)) is not None
    assert square_2x2_2D_grid_in_test.site_at((1.5, 1.5)) is not None

def test_simple_structure_doesnt_have_incorrect_sites(square_2x2_2D_grid_in_test: PeriodicStructure):
    assert square_2x2_2D_grid_in_test.site_at((0.5, 0.501)) is None

def test_simple_structure_has_periodic_sites(square_2x2_2D_grid_in_test: PeriodicStructure):
    assert square_2x2_2D_grid_in_test.site_at((-1.5, -1.5)) is not None
    assert square_2x2_2D_grid_in_test.site_at((-1.5, 1.5)) is not None
    assert square_2x2_2D_grid_in_test.site_at((-1.5, 2.5)) is not None
    assert square_2x2_2D_grid_in_test.site_at((2.5, -0.5)) is not None

def test_sites_have_unaltered_location(square_2x2_2D_grid_in_test: PeriodicStructure):
    location = (0.5, 0.5)
    site = square_2x2_2D_grid_in_test.site_at(location)
    assert site[LOCATION][0] == 0.5
    assert site[LOCATION][1] == 0.5

def test_structure_returns_sites(square_2x2_2D_grid_in_test):
    no_sites = square_2x2_2D_grid_in_test.sites("NOT_A_SITE")
    assert len(no_sites) == 0

    all_sites = square_2x2_2D_grid_in_test.sites("A")
    assert len(all_sites) == 4

def test_build_structure_from_list_motif(square_2D_lattice):
    motif = [(
        0.5, 0.5
    )]

    struct = PeriodicStructure.build_from(
        square_2D_lattice,
        num_cells=(3,3),
        site_motif=motif
    )

    assert len(struct.site_ids) == 9
    assert struct.site_at((0.5, 0.5)) is not None
    assert struct.site_at((-0.5, 2.5)) is not None
    assert struct.site_at((0.25, 0.25)) is None