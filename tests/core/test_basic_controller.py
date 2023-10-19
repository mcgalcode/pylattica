import pytest

from pylattica.core.basic_controller import BasicController
from pylattica.core.periodic_structure import PeriodicStructure
from pylattica.core.simulation_state import SimulationState

def test_simple_controller(square_grid_2D_4x4: PeriodicStructure):

    class SimpleController(BasicController):

        def get_state_update(self, site_id: int, prev_state: SimulationState):
            return {}

    sc = SimpleController(square_grid_2D_4x4)

    assert type(sc.get_random_site()) == int