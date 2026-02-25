import pytest

import random
import os
from pylattica.core import SimulationResult, SimulationState
from pylattica.core.simulation_result import compress_result


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


def test_max_history_steps_generator(initial_state):
    """Test that steps() works correctly with checkpoints."""
    result = SimulationResult(initial_state, max_history=50)

    for step in range(100):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Iterate through available steps
    steps_list = list(result.steps())

    # Should have steps from checkpoint onward
    expected_count = len(result._diffs) + 1  # diffs + checkpoint state
    assert len(steps_list) == expected_count

    # Each step should be a separate object (copies)
    assert steps_list[0] is not steps_list[1]


def test_max_history_load_steps(initial_state):
    """Test that load_steps() works correctly with checkpoints."""
    result = SimulationResult(initial_state, max_history=50)

    for step in range(100):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Load steps at interval
    result.load_steps(interval=10)

    # Should have cached states
    assert len(result._stored_states) > 0

    # Cached states should be after checkpoint
    for step_no in result._stored_states:
        assert step_no >= result.earliest_available_step


def test_original_length(initial_state):
    """Test the original_length property."""
    result = SimulationResult(initial_state, compress_freq=1)

    for step in range(10):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # With compress_freq=1, original_length should equal len
    assert result.original_length == len(result)

    # With compress_freq=2, original_length should be doubled
    result_compressed = SimulationResult(initial_state, compress_freq=2)
    for step in range(10):
        updates = {0: {"value": step}}
        result_compressed.add_step(updates)

    assert result_compressed.original_length == len(result_compressed) * 2


def test_compress_result(initial_state):
    """Test the compress_result function."""
    result = SimulationResult(initial_state)

    # Add 100 steps with deterministic values
    for step in range(100):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Compress to 20 steps
    compressed = compress_result(result, 20)

    # Should have fewer steps
    assert len(compressed) <= 25  # Some margin for sampling

    # compress_freq should be updated
    assert compressed.compress_freq > 1


def test_compress_result_invalid_size(initial_state):
    """Test that compress_result raises error for invalid target size."""
    result = SimulationResult(initial_state)

    for step in range(10):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Can't compress to more steps than we have
    with pytest.raises(ValueError, match="Cannot compress"):
        compress_result(result, 100)


def test_live_compress_stores_frames(initial_state):
    """Test that live_compress stores frames at compress_freq intervals."""
    result = SimulationResult(initial_state, compress_freq=10, live_compress=True)

    # Add 25 steps
    for step in range(25):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Should have frames at 0, 10, 20 (initial + steps 10 and 20)
    assert 0 in result._frames
    assert 10 in result._frames
    assert 20 in result._frames
    assert 25 not in result._frames  # Not a multiple of 10

    # Diffs should be empty in live_compress mode
    assert len(result._diffs) == 0


def test_live_compress_get_step(initial_state):
    """Test that get_step works with live_compress mode."""
    result = SimulationResult(initial_state, compress_freq=5, live_compress=True)

    for step in range(10):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Can get steps at frame intervals
    state_5 = result.get_step(5)
    assert state_5.get_site_state(0)["value"] == 4  # 0-indexed, step 5 has value 4

    state_10 = result.get_step(10)
    assert state_10.get_site_state(0)["value"] == 9

    # Cannot get steps that aren't at frame intervals
    with pytest.raises(ValueError, match="live_compress"):
        result.get_step(3)


def test_live_compress_load_steps_noop(initial_state):
    """Test that load_steps is a no-op in live_compress mode."""
    result = SimulationResult(initial_state, compress_freq=5, live_compress=True)

    for step in range(10):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # load_steps with matching interval should be a no-op
    result.load_steps(interval=5)  # Should not raise

    # load_steps with non-matching interval should raise
    with pytest.raises(ValueError, match="interval"):
        result.load_steps(interval=1)


def test_live_compress_steps_generator(initial_state):
    """Test that steps() yields frames in live_compress mode."""
    result = SimulationResult(initial_state, compress_freq=5, live_compress=True)

    for step in range(10):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Should yield frames in order: 0, 5, 10
    steps = list(result.steps())
    assert len(steps) == 3  # Frames at 0, 5, 10

    # First frame is initial (no value set yet)
    # Frame at step 5 has value 4 (last update before step 5 frame is taken)
    assert steps[1].get_site_state(0)["value"] == 4
    # Frame at step 10 has value 9
    assert steps[2].get_site_state(0)["value"] == 9


def test_live_compress_serialization(initial_state):
    """Test that live_compress results serialize and deserialize correctly."""
    result = SimulationResult(initial_state, compress_freq=5, live_compress=True)

    for step in range(10):
        updates = {0: {"value": step}}
        result.add_step(updates)

    # Serialize and deserialize
    result_dict = result.as_dict()
    restored = SimulationResult.from_dict(result_dict)

    # Check properties are preserved
    assert restored.live_compress is True
    assert restored.compress_freq == 5
    assert len(restored._frames) == 3  # 0, 5, 10
    assert len(restored._diffs) == 0

    # Check live_state is correctly restored
    assert restored.live_state.get_site_state(0)["value"] == 9


def test_live_state_property(initial_state):
    """Test that live_state property reflects current state."""
    result = SimulationResult(initial_state)

    # Initial live_state - site 0 doesn't exist yet
    assert result.live_state.get_site_state(0) is None

    # After adding steps, live_state is updated
    result.add_step({0: {"value": 42}})
    assert result.live_state.get_site_state(0)["value"] == 42

    result.add_step({0: {"value": 100}})
    assert result.live_state.get_site_state(0)["value"] == 100


def test_from_dict_restores_live_state_from_diffs(initial_state):
    """Test that from_dict replays diffs to restore live_state."""
    result = SimulationResult(initial_state)

    for step in range(5):
        result.add_step({0: {"value": step}})

    # Serialize and deserialize
    result_dict = result.as_dict()
    restored = SimulationResult.from_dict(result_dict)

    # live_state should be restored by replaying diffs
    assert restored.live_state.get_site_state(0)["value"] == 4


def test_from_dict_restores_live_state_from_checkpoint(initial_state):
    """Test that from_dict uses checkpoint when restoring live_state."""
    # Use max_history to trigger checkpoint creation
    result = SimulationResult(initial_state, max_history=5)

    # Add enough steps to trigger checkpoint
    for step in range(10):
        result.add_step({0: {"value": step}})

    # Should have a checkpoint now
    assert result._checkpoint_state is not None

    # Serialize and deserialize
    result_dict = result.as_dict()
    restored = SimulationResult.from_dict(result_dict)

    # live_state should be restored correctly (final value is 9)
    assert restored.live_state.get_site_state(0)["value"] == 9
    # Checkpoint should be restored
    assert restored._checkpoint_state is not None


def test_output_property(initial_state):
    """Test that output property is an alias for live_state."""
    result = SimulationResult(initial_state)

    result.add_step({0: {"value": 42}})

    # output should be same as live_state
    assert result.output is result.live_state
