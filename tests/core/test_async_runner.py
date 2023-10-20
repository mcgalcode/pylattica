import pytest

from pylattica.core import Runner, BasicController
from pylattica.core.simulation_state import SimulationState
from pylattica.core.periodic_structure import PeriodicStructure
from pylattica.core.constants import SITE_ID

import random

def test_simple_async_controller(square_grid_2D_4x4: PeriodicStructure):
    
    class SimpleAsyncController(BasicController):

        def get_state_update(self, site_id: int, prev_state: SimulationState):
            new_state = 3
            return {
                site_id: {
                    "value": new_state
                }
            }
    
    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], { "value": 0 })
    
    runner = Runner(is_async=True)

    controller = SimpleAsyncController(square_grid_2D_4x4)
    result = runner.run(initial_state, controller=controller, num_steps = 10, structure = square_grid_2D_4x4)

    last_step = result.last_step

    num_converted = 0
    for site_state in last_step.all_site_states():
        if site_state["value"] == 3:
            num_converted += 1
    
    assert num_converted > 0
    assert num_converted <= 10
    

def test_simple_async_controller_async_flag(square_grid_2D_4x4: PeriodicStructure):
    
    class SimpleAsyncController(BasicController):

        is_async = True

        def get_state_update(self, site_id: int, prev_state: SimulationState):
            new_state = 3
            return {
                site_id: {
                    "value": new_state
                }
            }
    
    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], { "value": 0 })
    
    runner = Runner()

    controller = SimpleAsyncController(square_grid_2D_4x4)
    result = runner.run(initial_state, controller=controller, num_steps = 10, structure = square_grid_2D_4x4)

    last_step = result.last_step

    num_converted = 0
    for site_state in last_step.all_site_states():
        if site_state["value"] == 3:
            num_converted += 1
    
    assert num_converted > 0
    assert num_converted <= 10