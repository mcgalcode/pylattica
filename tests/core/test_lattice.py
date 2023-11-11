import pytest

import numpy as np

from pylattica.core import Lattice, PeriodicStructure
from pylattica.core.lattice  import periodize
from pylattica.core.constants import SITE_ID

def assert_points_equal(pt1, pt2):
    assert (np.array(pt1) == np.array(pt2)).all()

def test_instantiate_lattice(square_2D_basis_vecs):
    assert Lattice(square_2D_basis_vecs) is not None

def test_basic_points(square_2D_basis_vecs):
    lat = Lattice(square_2D_basis_vecs)
    assert_points_equal(lat.get_cartesian_coords((1.5, 0.5)), (1.5, 0.5))

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

def test_rectangular_lattice_point_conversions():
    lattice = Lattice(
        [[1, 0],
        [0, 1/2]]
    )

    # points
    pt1 = (1/2, 1/4)
    pt2 = (1/2, 3/4)
    pt3 = (0, 1/2)

    # periodizing cartesian points
    assert (np.array([1/2, 1/4]) == lattice.get_periodized_cartesian_coords(pt1)).all()
    assert (np.array([1/2, 1/4]) == lattice.get_periodized_cartesian_coords(pt2)).all()
    assert (np.array([0, 0]) == lattice.get_periodized_cartesian_coords(pt3)).all()
    
    # conversion to fractional coords
    assert (np.array([1/2, 1/2]) == lattice.get_fractional_coords(pt1)).all()
    assert (np.array([1/2, 3/2]) == lattice.get_fractional_coords(pt2)).all()
    assert (np.array([0, 1]) == lattice.get_fractional_coords(pt3)).all()

    # conversion from fractional_coords coords
    assert (np.array([1/2, 1/8]) == lattice.get_cartesian_coords(pt1)).all()
    assert (np.array([1/2, 3/8]) == lattice.get_cartesian_coords(pt2)).all()
    assert (np.array([0, 1/4]) == lattice.get_cartesian_coords(pt3)).all()

def test_scaled_rectangular_lattice_point_conversions():
    lattice = Lattice(
        [[1, 0],
        [0, 1/2]]
    )

    lattice = lattice.get_scaled_lattice((2,1))

    # points
    pt1 = (3/2, 1/4)
    pt2 = (5/2, 3/4)
    pt3 = (0, 1/2)
    pt4 = (1,1/4)

    # periodizing cartesian points
    assert (np.array([3/2, 1/4]) == lattice.get_periodized_cartesian_coords(pt1)).all()
    assert (np.array([1/2, 1/4]) == lattice.get_periodized_cartesian_coords(pt2)).all()
    assert (np.array([0, 0]) == lattice.get_periodized_cartesian_coords(pt3)).all()
    assert (np.array([1, 1/4]) == lattice.get_periodized_cartesian_coords(pt4)).all()
    
    # conversion to fractional coords
    assert (np.array([3/4, 1/2]) == lattice.get_fractional_coords(pt1)).all()
    assert (np.array([5/4, 3/2]) == lattice.get_fractional_coords(pt2)).all()
    assert (np.array([0, 1]) == lattice.get_fractional_coords(pt3)).all()
    assert (np.array([1/2, 1/2]) == lattice.get_fractional_coords(pt4)).all()

    # conversion from fractional_coords coords
    assert (np.array([3, 1/8]) == lattice.get_cartesian_coords(pt1)).all()
    assert (np.array([5, 3/8]) == lattice.get_cartesian_coords(pt2)).all()
    assert (np.array([0, 1/4]) == lattice.get_cartesian_coords(pt3)).all()
    assert (np.array([2, 1/8]) == lattice.get_cartesian_coords(pt4)).all()

def test_rectangular_lattice_pbc_distance():
    lattice = Lattice(
        [[1, 0],
        [0, 1/2]]
    )

    scaled = lattice.get_scaled_lattice((2,2))

    pt1 = (0.5, 0.5)
    pt2 = (0.5, 0.75)
    pt3 = (2.5, 0.5)

    assert scaled.cartesian_periodic_distance(pt1, pt2) == 0.25
    assert scaled.cartesian_periodic_distance(pt1, pt3) == 0.0
    assert scaled.cartesian_periodic_distance(pt2, pt3) == 0.25

def test_square_lattice_non_pbc_distance():
    lattice = Lattice(
        [[1, 0],
        [0, 1]],
        periodic=False
    )

    pt1 = (0.1, 0.5)
    pt2 = (0.9, 0.5)
    pt3 = (1.5, 0.5)

    assert lattice.cartesian_periodic_distance(pt1, pt2) == 0.8
    assert lattice.cartesian_periodic_distance(pt1, pt3) == 1.4
    assert lattice.cartesian_periodic_distance(pt2, pt3) == 0.6


def test_square_lattice_non_pbc_coords():
    lattice = Lattice(
        [[1, 0],
        [0, 1]],
        periodic=False
    )

    pt1 = (0.1, 0.5)
    pt2 = (0.9, 0.5)
    pt3 = (1.5, 0.5)

    assert_points_equal(lattice.get_periodized_cartesian_coords(pt1), pt1)
    assert_points_equal(lattice.get_periodized_cartesian_coords(pt2), pt2)
    assert_points_equal(lattice.get_periodized_cartesian_coords(pt3), (1.5, 0.5))

def test_canted_lattice_pbc_distance():
    lattice = Lattice(
        [[1, 0],
        [1, 1]]
    )

    scaled = lattice.get_scaled_lattice((2,2))

    pt1 = (0.75, 0.5)
    pt2 = (1.75, 0.5)
    pt3 = (0, 0.5)
    pt4 = (2.5, 0.25)
    pt5 = (0.5, 0.25)
    pt6 = (1, 1.5)
    pt7 = (1, 0.5)

    assert scaled.cartesian_periodic_distance(pt1, pt2) == 1
    assert scaled.cartesian_periodic_distance(pt1, pt3) == 0.75
    assert scaled.cartesian_periodic_distance(pt4, pt5) == 0.0
    assert scaled.cartesian_periodic_distance(pt6, pt7) == 1
    
def test_canted_rectangular_lattice_point_conversions():
    lattice = Lattice(
        [[1, 0],
        [1, 1]]
    )

    # points
    pt1 = (1/2, 1/4)
    pt2 = (1/2, 3/4)
    pt3 = (0, 1/2)

    # periodizing cartesian points
    assert (np.array([1/2, 1/4]) == lattice.get_periodized_cartesian_coords(pt1)).all()
    assert (np.array([3/2, 3/4]) == lattice.get_periodized_cartesian_coords(pt2)).all()
    assert (np.array([1, 1/2]) == lattice.get_periodized_cartesian_coords(pt3)).all()
    
    # conversion to fractional coords
    assert (np.array([1/4, 1/4]) == lattice.get_fractional_coords(pt1)).all()
    assert (np.array([-1/4, 3/4]) == lattice.get_fractional_coords(pt2)).all()
    assert (np.array([-1/2, 1/2]) == lattice.get_fractional_coords(pt3)).all()

    # conversion from fractional_coords coords
    assert (np.array([3/4, 1/4]) == lattice.get_cartesian_coords(pt1)).all()
    assert (np.array([5/4, 3/4]) == lattice.get_cartesian_coords(pt2)).all()
    assert (np.array([1/2, 1/2]) == lattice.get_cartesian_coords(pt3)).all()

def test_periodizing_point():
    pt1 = (0.5, 1.5, 0.8)
    assert (np.array((0.5, 0.5, 0.8)) == periodize(pt1)).all()

    pt2 = (0, 0, 0)
    assert (np.array((0, 0, 0)) == periodize(pt2)).all()

    pt3 = (1.5, 0.8, -0.5)
    assert (np.array((1.5, 0.8, -0.5)) == periodize(pt3, False)).all()

    assert (np.array((1.5, 0.8, 0.5)) == periodize(pt3, (False, False, True))).all()
    assert (np.array((0.5, 0.8, 0.5)) == periodize(pt3, (True, False, True))).all()

    pt4 = (1.5, 1.5)
    assert_points_equal((0.5, 1.5), periodize(pt4, (True, False)))

def test_nonperiodic_lattice(square_2D_basis_vecs):
    lat = Lattice(square_2D_basis_vecs, False)

    assert_points_equal(lat.get_periodized_cartesian_coords([1.5, 1.5]), [1.5, 1.5])

    lat = Lattice(square_2D_basis_vecs, True)
    assert_points_equal(lat.get_periodized_cartesian_coords([1.5, 1.5]), [0.5, 0.5])

    lat_slab = Lattice(square_2D_basis_vecs, (True, False))
    assert_points_equal(lat_slab.get_periodized_cartesian_coords([1.5, 1.5]), [0.5, 1.5])


def test_partially_periodic_lattice(square_2D_basis_vecs):
    lat = Lattice(square_2D_basis_vecs, (True, False)).get_scaled_lattice((3,3))

    assert_points_equal(lat.get_periodized_cartesian_coords((-0.5, 1.5)), (2.5, 1.5))
    assert_points_equal(lat.get_periodized_cartesian_coords((1.5, -0.5)), (1.5, -0.5))
    assert_points_equal(lat.get_periodized_cartesian_coords((-0.5, -0.5)), (2.5, -0.5))

def test_scaling_lattice_retains_periodicity(square_2D_basis_vecs):
    lat = Lattice(square_2D_basis_vecs, False)
    scaled = lat.get_scaled_lattice((2,2))
    assert scaled.periodic == (False, False)

    lat = Lattice(square_2D_basis_vecs, (False, True))
    scaled = lat.get_scaled_lattice((2,2))
    assert scaled.periodic == (False, True)

    lat = Lattice(square_2D_basis_vecs, True)
    scaled = lat.get_scaled_lattice((2,2))
    assert scaled.periodic == (True, True)