import pytest
import time
import sys

from pylattica.core import SynchronousRunner, BasicController
from pylattica.core.simulation_state import SimulationState
from pylattica.core.periodic_structure import PeriodicStructure
from pylattica.core.constants import SITE_ID

from test_helpers.helpers import skip_windows_due_to_parallel

@skip_windows_due_to_parallel
def test_parallel_runner(square_grid_2D_4x4: PeriodicStructure):
    
    class SimpleParallelController(BasicController):

        def get_state_update(self, site_id: int, prev_state: SimulationState):
            prev = prev_state.get_site_state(site_id)["value"]
            new_state = prev + 1
            return {
                site_id: {
                    "value": new_state
                }
            }
    
    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], { "value": 0 })
    
    runner = SynchronousRunner(parallel=True)

    controller = SimpleParallelController()
    result = runner.run(initial_state, controller=controller, num_steps = 1000)

    last_step = result.last_step

    for site_state in last_step.all_site_states():
        assert site_state["value"] == 1000

@skip_windows_due_to_parallel
def test_parallel_runner_speed(square_grid_2D_4x4: PeriodicStructure):
    
    class SimpleParallelController(BasicController):

        def get_state_update(self, site_id: int, prev_state: SimulationState):
            prev = prev_state.get_site_state(site_id)["value"]
            new_state = prev + 1
            return {
                site_id: {
                    "value": new_state
                }
            }
    
    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], { "value": 0 })
    

    
    parallel_runner = SynchronousRunner(parallel=True)
    series_runner = SynchronousRunner()

    controller = SimpleParallelController()

    num_steps = 1000

    t0 = time.time()
    parallel_result = parallel_runner.run(initial_state, controller=controller, num_steps = num_steps)
    t1 = time.time()

    t2 = time.time()
    series_result = series_runner.run(initial_state, controller=controller, num_steps = num_steps)
    t3 = time.time()

    assert (t3 - t2) < (t1 - t0)

    for site_state in parallel_result.last_step.all_site_states():
        assert site_state["value"] == num_steps

    for site_state in series_result.last_step.all_site_states():
        assert site_state["value"] == num_steps