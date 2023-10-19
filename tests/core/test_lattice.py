import pytest

from pylattica.core import Lattice, PeriodicStructure
from pylattica.core.constants import SITE_ID

def test_instantiate_lattice(square_2D_basis_vecs):
    assert Lattice(square_2D_basis_vecs) is not None

def test_can_build_simple_2x2_grid(square_2D_lattice, simple_motif):
    structure = PeriodicStructure.build_from(square_2D_lattice, [2, 2], simple_motif)
    assert structure is not None

def test_simple_structure_has_correct_sites(square_2D_lattice, simple_motif):
    structure = PeriodicStructure.build_from(square_2D_lattice, [2, 2], simple_motif)

    site_1 = structure.site_at((0.5, 0.5))
    site_2 = structure.site_at((0.5, 1.5))
    site_3 = structure.site_at((1.5, 0.5))
    site_4 = structure.site_at((1.5, 1.5))

    assert site_1 is not None
    assert site_2 is not None
    assert site_3 is not None
    assert site_4 is not None

    assert len(set([s[SITE_ID] for s in [site_1, site_2, site_3, site_4]])) == 4
