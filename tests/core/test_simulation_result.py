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

        updates = {site_id: {"a": val}}
        result.add_step(updates)

    return result


@pytest.fixture
def random_result_small(initial_state):
    result = SimulationResult(initial_state)

    for _ in range(3):
        site_id = random.randint(0, 10)
        val = random.random()

        updates = {site_id: {"a": val}}
        result.add_step(updates)

    return result


@pytest.fixture
def random_result_small_ordered(initial_state):
    result = SimulationResult(initial_state)

    for i in range(3):
        site_id = random.randint(0, 10)

        updates = {site_id: {"a": i}}
        result.add_step(updates)

    return result


def test_can_add_step(initial_state):
    result = SimulationResult(initial_state)

    updates = {24: {"a": 1}}

    result.add_step(updates)

    assert len(result) == 2
    first_step = result.first_step
    assert first_step.as_dict() == initial_state.as_dict()


def test_can_load_at_intervals(random_result_big):
    assert len(random_result_big) == 1000

    random_result_big.load_steps(interval=10)
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
    print(rehydrated._diffs[0])
    os.remove(fname)
    assert random_result_small.as_dict() == rehydrated.as_dict()


def test_write_file_autoname(random_result_small: SimulationResult):
    fname = random_result_small.to_file()

    rehydrated = SimulationResult.from_file(fname)
    assert os.path.exists(fname)
    os.remove(fname)
    assert random_result_small.as_dict() == rehydrated.as_dict()


def test_diff_storage(random_result_small_ordered: SimulationResult):
    diff_one = random_result_small_ordered._diffs[0]
    assert len(diff_one.keys()) == 1


def test_max_history_limits_memory(initial_state):
    """Test that max_history limits the number of diffs kept in memory."""
    result = SimulationResult(initial_state, max_history=50)

    # Add 200 steps
    for step in range(200):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Should have at most max_history diffs in memory
    assert len(result._diffs) <= 50

    # But total steps should still be correct
    assert len(result) == 201  # 200 steps + initial state
    assert result._total_steps == 200


def test_max_history_creates_checkpoint(initial_state):
    """Test that exceeding max_history creates a checkpoint."""
    result = SimulationResult(initial_state, max_history=50)

    # Add 100 steps to trigger checkpointing
    for step in range(100):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Checkpoint should have been created
    assert result._checkpoint_state is not None
    assert result._checkpoint_step > 0
    assert result.earliest_available_step == result._checkpoint_step


def test_max_history_get_step_recent(initial_state):
    """Test that recent steps are still accessible with max_history."""
    result = SimulationResult(initial_state, max_history=50)

    for step in range(100):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Should be able to get the last step
    last_step = result.get_step(100)
    assert last_step.get_site_state(0)["value"] == 99

    # Should be able to get steps after checkpoint
    earliest = result.earliest_available_step
    step = result.get_step(earliest + 1)
    assert step is not None


def test_max_history_get_step_early_raises(initial_state):
    """Test that requesting steps before checkpoint raises ValueError."""
    result = SimulationResult(initial_state, max_history=50)

    for step in range(100):
        updates = {0: {"value": step}}
        result.add_step(updates)

    earliest = result.earliest_available_step
    assert earliest > 0  # Checkpoint should exist

    with pytest.raises(ValueError, match="Cannot retrieve step"):
        result.get_step(0)


def test_max_history_serialization(initial_state):
    """Test that results with max_history serialize and deserialize correctly."""
    result = SimulationResult(initial_state, max_history=50)

    for step in range(100):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Serialize and deserialize
    d = result.as_dict()
    rehydrated = SimulationResult.from_dict(d)

    # Check state is preserved
    assert rehydrated.max_history == result.max_history
    assert rehydrated._checkpoint_step == result._checkpoint_step
    assert rehydrated._total_steps == result._total_steps
    assert len(rehydrated._diffs) == len(result._diffs)

    # Check we can get the same steps
    for step_no in range(result.earliest_available_step, len(result)):
        orig = result.get_step(step_no)
        rehyd = rehydrated.get_step(step_no)
        assert orig.as_dict() == rehyd.as_dict()


def test_max_history_none_unlimited(initial_state):
    """Test that max_history=None allows unlimited growth (default behavior)."""
    result = SimulationResult(initial_state, max_history=None)

    for step in range(500):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # All diffs should be in memory
    assert len(result._diffs) == 500
    assert result._checkpoint_state is None
    assert result.earliest_available_step == 0
