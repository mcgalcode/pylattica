from pylattica.core.constants import SITE_ID
from pylattica.core.periodic_structure import PeriodicStructure
from pylattica.core.simulation_state import SimulationState
import pytest

from pylattica.core import PeriodicState


def test_can_instantiate_periodic_state(square_2x2_2D_grid):
    state = SimulationState()
    assert PeriodicState(state, square_2x2_2D_grid) is not None

def test_retrieves_state_correctly(square_2x2_2D_grid: PeriodicStructure):
    state = SimulationState()
    site = square_2x2_2D_grid.site_at((0.5, 0.5))
    state.set_site_state(site[SITE_ID], { 'my_state_key': 2 })

    periodic_state = PeriodicState(state, square_2x2_2D_grid)
    site_state = periodic_state.state_at((0.5, 0.5))
    assert site_state['my_state_key'] == 2
