from ...core.neighborhood_builders import (
    DistanceNeighborhoodBuilder,
)
from .lattice import HONEYCOMB_SIDE_LENGTH


class HoneycombNeighborhoodBuilder(DistanceNeighborhoodBuilder):
    """A neighborhood for Honeycomb Lattices"""

    def __init__(self):
        self.cutoff = HONEYCOMB_SIDE_LENGTH + 0.01
