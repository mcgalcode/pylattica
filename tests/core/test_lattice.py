import pytest

from pylattica.core import Lattice

def test_instantiate_lattice(square_2D_basis_vecs):
    assert Lattice(square_2D_basis_vecs) is not None

def test_can_build_simple_2x2_grid(square_2D_lattice, simple_motif):
    structure = square_2D_lattice.build_from([2, 2], simple_motif)
    assert structure is not None

def test_simple_structure_has_correct_sites(square_2D_lattice, simple_motif):
    structure = square_2D_lattice.build_from([2, 2], simple_motif)

    assert structure.site_at((0.5, 0.5)) is not None
    assert structure.site_at((0.5, 1.5)) is not None
    assert structure.site_at((1.5, 0.5)) is not None
    assert structure.site_at((1.5, 1.5)) is not None
