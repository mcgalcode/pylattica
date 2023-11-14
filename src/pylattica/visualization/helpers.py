import typing

from .colors import COLORS


def color_map(phases):
    color_map: typing.Dict[str, typing.Tuple[int, int, int]] = {}
    c_idx: int = 0
    for p in phases:
        color_map[p] = COLORS[c_idx]
        c_idx += 1

    return color_map
