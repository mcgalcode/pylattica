import pytest

from pylattica.core import AsynchronousRunner, BasicController
from pylattica.core.simulation_state import SimulationState
from pylattica.core.simulation_result import SimulationResult
from pylattica.core.periodic_structure import PeriodicStructure
from pylattica.core.constants import SITE_ID, GENERAL

import random


def test_simple_async_controller(square_grid_2D_4x4: PeriodicStructure):
    class SimpleAsyncController(BasicController):
        def get_state_update(self, site_id: int, prev_state: SimulationState):
            new_state = 3
            return {site_id: {"value": new_state}}

    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], {"value": 0})

    runner = AsynchronousRunner()

    controller = SimpleAsyncController()
    result = runner.run(initial_state, controller=controller, num_steps=10)

    last_step = result.last_step

    num_converted = 0
    for site_state in last_step.all_site_states():
        if site_state["value"] == 3:
            num_converted += 1

    assert num_converted > 0
    assert num_converted <= 10


def test_simple_async_controller_async_flag(square_grid_2D_4x4: PeriodicStructure):
    class SimpleAsyncController(BasicController):
        def get_state_update(self, site_id: int, prev_state: SimulationState):
            new_state = 3
            return {site_id: {"value": new_state}}

    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], {"value": 0})

    runner = AsynchronousRunner()

    controller = SimpleAsyncController()
    result = runner.run(initial_state, controller=controller, num_steps=10)

    last_step = result.last_step

    num_converted = 0
    for site_state in last_step.all_site_states():
        if site_state["value"] == 3:
            num_converted += 1

    assert num_converted > 0
    assert num_converted <= 10


def test_general_state_async_controller(square_grid_2D_4x4: PeriodicStructure):
    class GeneralStateAsyncController(BasicController):
        def get_state_update(self, site_id: int, prev_state: SimulationState):
            prev = prev_state.get_general_state("value")
            if prev is not None:
                nxt = prev + 1
            else:
                nxt = 1
            return {GENERAL: {"value": nxt}}

    initial_state = SimulationState()
    initial_state.set_site_state(1, {"value": 0})

    runner = AsynchronousRunner()

    controller = GeneralStateAsyncController()
    result = runner.run(initial_state, controller=controller, num_steps=10)

    for i in range(1, 10):
        state = result.get_step(i)
        gen = state.get_general_state("value")
        assert gen == i

    fname = "tmp.json"
    result.to_file(fname)
    rehyd = SimulationResult.from_file(fname)
    for i in range(1, 10):
        state = rehyd.get_step(i)
        gen = state.get_general_state("value")
        assert gen == i


def test_async_controller_next_sites(square_grid_2D_4x4):
    CHANGED_STATE = 3

    class NextSiteAsyncController(BasicController):
        def get_random_site(self, _):
            return 0

        def get_state_update(self, site_id: int, _):
            return {site_id: {"value": CHANGED_STATE}}, [site_id + 1]

    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], {"value": 0})

    runner = AsynchronousRunner()

    controller = NextSiteAsyncController()
    num_steps = 4
    result = runner.run(initial_state, controller=controller, num_steps=num_steps)

    last_step = result.last_step

    for site_state in last_step.all_site_states():
        if site_state[SITE_ID] < num_steps:
            assert site_state["value"] == CHANGED_STATE
        else:
            assert site_state["value"] == 0


def test_async_controller_random_site_list(square_grid_2D_4x4):
    CHANGED_STATE = 3
    chosen_sites = [0, 3, 5, 9]

    class NextSiteAsyncController(BasicController):
        def get_random_site(self, _):
            return chosen_sites

        def get_state_update(self, site_id: int, _):
            return {site_id: {"value": CHANGED_STATE}}

    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], {"value": 0})

    runner = AsynchronousRunner()

    controller = NextSiteAsyncController()
    num_steps = 10
    result = runner.run(initial_state, controller=controller, num_steps=num_steps)

    last_step = result.last_step

    for site_state in last_step.all_site_states():
        if site_state[SITE_ID] in chosen_sites:
            assert site_state["value"] == CHANGED_STATE
        else:
            assert site_state["value"] == 0


def test_async_controller_empty_random_site_list(square_grid_2D_4x4):
    CHANGED_STATE = 3
    chosen_sites = []

    class NextSiteAsyncController(BasicController):
        def get_random_site(self, _):
            return chosen_sites

        def get_state_update(self, site_id: int, _):
            return {site_id: {"value": CHANGED_STATE}}

    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], {"value": 0})

    runner = AsynchronousRunner()

    controller = NextSiteAsyncController()
    num_steps = 10
    with pytest.raises(RuntimeError, match="Controller provided"):
        result = runner.run(initial_state, controller=controller, num_steps=num_steps)


def test_async_controller_limited_sites(square_grid_2D_4x4):
    CHANGED_STATE = 3
    chosen_sites = [0, 3, 5, 9]
    original_chosen_sites = chosen_sites.copy()

    class NextSiteAsyncController(BasicController):
        steps_taken = 0

        def get_random_site(self, _):
            if len(chosen_sites) > 0:
                return chosen_sites.pop()
            else:
                return []

        def get_state_update(self, site_id: int, _):
            self.steps_taken += 1
            return {site_id: {"value": CHANGED_STATE}}

    initial_state = SimulationState()
    for site in square_grid_2D_4x4.sites():
        initial_state.set_site_state(site[SITE_ID], {"value": 0})

    runner = AsynchronousRunner()

    controller = NextSiteAsyncController()
    num_steps = 10
    result = runner.run(initial_state, controller=controller, num_steps=num_steps)
    assert controller.steps_taken == 4
    last_step = result.last_step

    for site_state in last_step.all_site_states():
        if site_state[SITE_ID] in original_chosen_sites:
            assert site_state["value"] == CHANGED_STATE
        else:
            assert (
                site_state["value"] == 0
            ), f"{site_state[SITE_ID]} had the wrong state!"
