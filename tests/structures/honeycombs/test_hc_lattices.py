import pytest

import numpy as np

from pylattica.structures.honeycomb.lattice import RhombohedralLattice, HONEYCOMB_SIDE_LENGTH, ROOT_3


def test_rhombohedral_lattice():
    lattice = RhombohedralLattice()

    pt1 = (1/2, 1/2)
    periodized_pt1 = lattice.get_periodized_cartesian_coords(pt1)
    assert (np.array(pt1) == periodized_pt1).all()

    pt2 = (0, 1/2)
    periodized_pt2 = lattice.get_periodized_cartesian_coords(pt2)
    assert (np.array([1, 1/2]) == periodized_pt2).all()

    pt3 = (1, 3/2)
    periodized_pt3 = lattice.get_periodized_cartesian_coords(pt3)
    assert np.allclose(np.array([1/2, 3/2 - ROOT_3 /2]), periodized_pt3)