import pytest

from pylattica.core.coordinate_utils import get_points_in_box


def test_get_points_in_box():
    lbs = (0, 0)
    ubs = (2, 2)

    pts = get_points_in_box(lbs, ubs)

    assert len(pts) == 4
    assert (0, 0) in pts
