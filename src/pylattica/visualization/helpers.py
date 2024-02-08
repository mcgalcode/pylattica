from typing import List, Dict, Tuple

from .colors import COLORS


def color_map(phases: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """Generates a mapping of phase to RGB color tuple from a list of phase names.

    Parameters
    ----------
    phases : List[str]
        The phases to include in the color map.

    Returns
    -------
    Dict[str, Tuple[int, int, int]]
        The mapping of phase name to color.
    """
    color_map: Dict[str, Tuple[int, int, int]] = {}
    c_idx: int = 0
    for p in phases:
        color_map[p] = COLORS[c_idx]
        c_idx += 1

    return color_map
