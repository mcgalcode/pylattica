import pytest

import numpy as np

from pylattica.structures.honeycomb import HoneycombTilingBuilder
from pylattica.structures.honeycomb.lattice import ROOT_3
from pylattica.core.constants import LOCATION


def test_small_honeycomb_tiling_builder():
    builder = HoneycombTilingBuilder()

    tiling = builder.build(1)

    assert len(tiling.site_ids) == 1

    site_0_cart = tiling.get_site(0)[LOCATION]
    site_0_frac = tiling.lattice.get_fractional_coords(site_0_cart)
    assert (np.array(site_0_frac) == np.array([0.5, 0.5])).all()
    assert (np.array(site_0_cart) == np.array([3/4, ROOT_3 / 4])).all()

def test_medium_honeycomb_tiling_builder():
    builder = HoneycombTilingBuilder()

    tiling2 = builder.build((2,1))
    site_0_cart = tiling2.get_site(0)[LOCATION]
    site_0_frac = tiling2.lattice.get_fractional_coords(site_0_cart)
    assert (np.array(site_0_frac) == np.array([0.25, 0.5])).all()
    assert (np.array(site_0_cart) == np.array([3/4, ROOT_3 / 4])).all()

    site_1_cart = tiling2.get_site(1)[LOCATION]
    site_1_frac = tiling2.lattice.get_fractional_coords(site_1_cart)
    assert (np.array(site_1_frac) == np.array([3 / 4, 0.5])).all()
    assert (np.array(site_1_cart) == np.array([7/4, ROOT_3 / 4])).all()


def test_medium_honeycomb_tiling_builder():
    builder = HoneycombTilingBuilder()

    tiling = builder.build((3,3))

    middle_loc = tiling.lattice.get_cartesian_coords((0.5, 0.5))
    assert (middle_loc == np.array((9/4, 3 * ROOT_3 / 4))).all()
    middle_site = tiling.site_at(middle_loc)
    assert middle_site is not None

    upper_left_loc = tiling.lattice.get_cartesian_coords((1/6, 5/6))
    assert np.allclose(upper_left_loc, np.array((7/4, 5 * ROOT_3 / 4)))

    assert np.linalg.norm(middle_loc - upper_left_loc) == 1.0
    ul_site = tiling.site_at(upper_left_loc)
    assert ul_site is not None

    assert tiling.lattice.cartesian_periodic_distance(middle_loc, upper_left_loc) == 1
    