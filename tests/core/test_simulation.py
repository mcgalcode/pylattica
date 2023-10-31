from pylattica.core.constants import SITE_ID
from pylattica.core.periodic_structure import PeriodicStructure
from pylattica.core.simulation_state import SimulationState
import pytest

from pylattica.core import Simulation


def test_can_instantiate_periodic_state(square_2x2_2D_grid_in_test):
    state = SimulationState()
    assert Simulation(state, square_2x2_2D_grid_in_test) is not None

def test_retrieves_state_correctly(square_2x2_2D_grid_in_test: PeriodicStructure):
    state = SimulationState()
    site = square_2x2_2D_grid_in_test.site_at((0.5, 0.5))
    state.set_site_state(site[SITE_ID], { 'my_state_key': 2 })

    periodic_state = Simulation(state, square_2x2_2D_grid_in_test)
    site_state = periodic_state.state_at((0.5, 0.5))
    assert site_state['my_state_key'] == 2

def test_returns_none_if_no_state(square_2x2_2D_grid_in_test: PeriodicStructure):
    state = SimulationState()
    sim = Simulation(state, square_2x2_2D_grid_in_test)

    assert sim.state_at((0.6, 0.6)) is None
