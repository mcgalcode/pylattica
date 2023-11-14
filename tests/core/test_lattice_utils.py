from pylattica.core.lattice import pbc_diff_frac_vec, Lattice

import numpy as np

def test_pbc_diff_frac_vec():
    pt1 = (0.1, 0)
    pt2 = (0.9, 0)
    pt3 = (1.9, 0)
    pt4 = (0.9, 0.9)
    pt5 = (0.1, 0.1)
    pt6 = (-0.1, -0.1)
    pt7 = (1.1, 1.1)

    diff = pbc_diff_frac_vec(pt1, pt2, np.array([1, 1]))
    assert np.allclose(diff, [0.2, 0])

    diff = pbc_diff_frac_vec(pt1, pt2, np.array([0, 0]))
    assert np.allclose(diff, [-0.8, 0])
    assert np.allclose(pbc_diff_frac_vec(pt1, pt3, np.array([0, 0])), [-1.8, 0])

    assert np.allclose(pbc_diff_frac_vec(pt4, pt5, np.array([1, 1])), [-0.2, -0.2])
    assert np.allclose(pbc_diff_frac_vec(pt4, pt5, np.array([0, 0])), [0.8, 0.8])
    assert np.allclose(pbc_diff_frac_vec(pt4, pt5, np.array([0, 1])), [0.8, -0.2])
    assert np.allclose(pbc_diff_frac_vec(pt4, pt5, np.array([1, 0])), [-0.2, 0.8])

    assert np.allclose(pbc_diff_frac_vec(pt7, pt6, np.array([1, 1])), [0.2, 0.2])
    assert np.allclose(pbc_diff_frac_vec(pt7, pt6, np.array([1, 0])), [0.2, 1.2])
    assert np.allclose(pbc_diff_frac_vec(pt7, pt6, np.array([0, 1])), [1.2, 0.2])
    assert np.allclose(pbc_diff_frac_vec(pt7, pt6, np.array([0, 0])), [1.2, 1.2])

def test_pbc_diff_cart():
    lvecs = [
        [1, 0],
        [0, 1]
    ]

    pt1 = (0.1, 0.1)
    pt2 = (0.1, 0.9)
    pt5 = (-0.9, 0.1)

    l1 = Lattice(lvecs)
    l2 = Lattice(lvecs, (False, True))
    l3 = Lattice(lvecs, (True, False))
    l4 = Lattice(lvecs, False)

    assert np.isclose(l1.cartesian_periodic_distance(pt1, pt2), 0.2)
    assert np.isclose(l2.cartesian_periodic_distance(pt1, pt2), 0.2)
    assert np.isclose(l3.cartesian_periodic_distance(pt1, pt2), 0.8)
    assert np.isclose(l4.cartesian_periodic_distance(pt1, pt2), 0.8)

    assert np.isclose(l1.cartesian_periodic_distance(pt1, pt5), 0)
    assert np.isclose(l2.cartesian_periodic_distance(pt1, pt5), 1.0)
    assert np.isclose(l3.cartesian_periodic_distance(pt1, pt5), 0)
    assert np.isclose(l4.cartesian_periodic_distance(pt1, pt5), 1.0)

    




