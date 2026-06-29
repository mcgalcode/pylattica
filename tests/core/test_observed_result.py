import os

import pytest

from pylattica.core.observed_result import ObservedResult
from pylattica.core.observers import StateObserver
from pylattica.core.simulation_state import SimulationState


class CountingObserver(StateObserver):
    """Records how many sites have been touched (a tiny reduced view)."""

    def observe(self, state, step_no):
        return len(state.site_ids())


@pytest.fixture
def state():
    s = SimulationState()
    s.set_site_state(0, {"value": 0})
    return s


def step_reducer(state, step_no):
    """A bare-callable reducer returning the step number."""
    return step_no


def test_records_step_zero_immediately(state):
    res = ObservedResult(observer=step_reducer)
    res.record(state, 0)
    assert res.observation_at(0) == 0
    assert res.observation_steps == [0]


def test_samples_on_cadence(state):
    res = ObservedResult(observer=step_reducer, compress_freq=5)
    for step in range(11):
        res.record(state, step)

    assert res.observation_steps == [0, 5, 10]
    assert res.observation_at(5) == 5


def test_no_observer_records_nothing(state):
    res = ObservedResult(observer=None, compress_freq=1)
    for step in range(5):
        res.record(state, step)
    assert len(res) == 0
    # total_steps still tracked for serialization fidelity.
    assert res._total_steps == 4


def test_finalize_captures_final_step(state):
    res = ObservedResult(observer=step_reducer, compress_freq=5)
    for step in range(8):
        res.record(state, step)

    # Step 7 is off the cadence, so it is absent until finalized.
    assert 7 not in res.observations
    res.finalize(state, 7)
    assert res.observation_at(7) == 7


def test_finalize_no_duplicate(state):
    res = ObservedResult(observer=step_reducer, compress_freq=5)
    for step in range(11):
        res.record(state, step)

    # Step 10 already on cadence; finalize must not overwrite/duplicate.
    res.finalize(state, 10)
    assert res.observation_steps == [0, 5, 10]


def test_finalize_noop_before_any_steps(state):
    res = ObservedResult(observer=step_reducer, compress_freq=5)
    res.record(state, 0)
    res.finalize(state, 0)
    assert res.observation_steps == [0]


def test_num_frames_derives_cadence(state):
    res = ObservedResult(observer=step_reducer, num_frames=5)
    assert res.compress_freq == 1
    res.prepare_for_steps(20)
    assert res.compress_freq == 4


def test_state_observer_subclass(state):
    res = ObservedResult(observer=CountingObserver(), compress_freq=1)
    res.record(state, 0)
    assert res.observation_at(0) == 1


def test_observation_at_missing_raises(state):
    res = ObservedResult(observer=step_reducer, compress_freq=5)
    res.record(state, 0)
    with pytest.raises(KeyError):
        res.observation_at(3)


def test_both_cadence_args_raise():
    with pytest.raises(ValueError, match="either"):
        ObservedResult(observer=step_reducer, compress_freq=5, num_frames=5)


def test_serialization_round_trip(state):
    res = ObservedResult(observer=step_reducer, compress_freq=5)
    for step in range(11):
        res.record(state, step)
    res.finalize(state, 10)

    restored = ObservedResult.from_dict(res.as_dict())

    assert restored.compress_freq == 5
    assert restored.observation_steps == [0, 5, 10]
    # Keys round-trip back to ints.
    assert restored.observation_at(10) == 10
    assert restored._total_steps == 10
    # No observer survives deserialization (observations already computed).
    assert restored.observer is None


def test_num_frames_serialization_round_trip(state):
    res = ObservedResult(observer=step_reducer, num_frames=5)
    res.prepare_for_steps(20)
    for step in range(21):
        res.record(state, step)
    res.finalize(state, 20)

    restored = ObservedResult.from_dict(res.as_dict())
    assert restored.num_frames == 5
    assert restored.compress_freq == 4


def test_write_file_round_trip(state):
    res = ObservedResult(observer=step_reducer, compress_freq=2)
    for step in range(5):
        res.record(state, step)

    fname = "tmp_test_observed.json"
    res.to_file(fname)
    try:
        restored = ObservedResult.from_file(fname)
    finally:
        os.remove(fname)

    assert restored.observation_steps == res.observation_steps
