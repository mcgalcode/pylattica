import numpy as np

from ..core import SimulationState, StateAnalyzer
from .state_constants import DISCRETE_OCCUPANCY

class DiscreteStepAnalyzer(StateAnalyzer):

    def cell_fraction(self, state: SimulationState, phase_name: str):
        phase_count = self.cell_count(state, phase_name)
        total_occupied_cells = state.size
        return phase_count / total_occupied_cells

    def cell_count(self, state: SimulationState, phase_name: str):
        return self.get_site_count_where_equal(state, {
            DISCRETE_OCCUPANCY: phase_name
        })

    def cell_ratio(self, step, p1, p2):
        return self.cell_count(step, p1) / self.cell_count(step, p2)

    def phase_count(self, step):
        return len(np.unique(step.state)) - 1

    def phases_present(self, state: SimulationState):
        return list(set([site_state[DISCRETE_OCCUPANCY] for site_state in state.all_site_states()]))
