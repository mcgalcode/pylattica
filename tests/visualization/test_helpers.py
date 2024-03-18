import pytest
import random

from pylattica.visualization.helpers import color_map, COLORS


def test_color_map_no_extra_phases():
    len_colors = len(COLORS)

    phases = [str(random.randint(0, 100000)) for _ in range(10)]
    cmap = color_map(phases)

    all_colors = set(cmap.values())
    assert len(all_colors) == len(phases)
