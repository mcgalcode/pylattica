import pytest

from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY
from pylattica.discrete import DiscreteStepAnalyzer
from pylattica.core import SimulationState


def test_cell_count():
    state = SimulationState()
    state.set_site_state(1, {DISCRETE_OCCUPANCY: "A"})
    state.set_site_state(3, {DISCRETE_OCCUPANCY: "A"})
    state.set_site_state(2, {DISCRETE_OCCUPANCY: "B"})

    analyzer = DiscreteStepAnalyzer()

    assert analyzer.cell_count(state, "A") == 2
    assert analyzer.cell_count(state, "B") == 1

    assert analyzer.cell_ratio(state, "A", "B") == 2
    assert analyzer.cell_ratio(state, "B", "A") == 0.5

    assert analyzer.phase_count(state) == 2
