import pytest

import random
import os
from pylattica.core import SimulationResult, SimulationState


@pytest.fixture
def initial_state():
    return SimulationState()

@pytest.fixture
def random_result_big(initial_state):
    result = SimulationResult(initial_state)

    for _ in range(999):
        site_id = random.randint(0, 10)
        val = random.random()

        updates = {
            site_id: {
                "a": val
            }
        }
        result.add_step(updates)

    return result

@pytest.fixture
def random_result_small(initial_state):
    result = SimulationResult(initial_state)

    for _ in range(3):
        site_id = random.randint(0, 10)
        val = random.random()

        updates = {
            site_id: {
                "a": val
            }
        }
        result.add_step(updates)

    return result

def test_can_add_step(initial_state):
    result = SimulationResult(initial_state)

    updates = {
        24: { "a": 1 }
    }
    
    result.add_step(updates)

    assert len(result) == 2
    first_step = result.first_step
    assert first_step.as_dict() == initial_state.as_dict()

def test_can_load_at_intervals(random_result_big):

    assert len(random_result_big) == 1000

    random_result_big.load_steps(interval = 10)
    assert len(random_result_big._stored_states) == 100

def test_serialization(random_result_big: SimulationResult):
    d = random_result_big.as_dict()

    rehydrated = SimulationResult.from_dict(d)

    for idx, step in enumerate(rehydrated.steps()):
        orig = random_result_big.get_step(idx)
        assert step.as_dict() == orig.as_dict()

def test_write_file(random_result_small: SimulationResult):
    fname = "tmp_test_res.json"
    random_result_small.to_file(fname)

    rehydrated = SimulationResult.from_file(fname)

    os.remove(fname)
    assert random_result_small.as_dict() == rehydrated.as_dict()

def test_write_file_autoname(random_result_small: SimulationResult):
    fname = random_result_small.to_file()

    rehydrated = SimulationResult.from_file(fname)
    assert os.path.exists(fname)
    os.remove(fname)
    assert random_result_small.as_dict() == rehydrated.as_dict()
