import numpy as np

from rxn_ca.core.simulation_step import SimulationState
from .phase_set import PhaseSet


class DiscreteStepAnalyzer():

    def cell_fraction(self, state: SimulationState, phase_name: str):
        phase_count = self.cell_count(state, phase_name)
        total_occupied_cells = state.size
        return phase_count / total_occupied_cells

    def cell_count(self, state: SimulationState, phase_name: str):
        count = 0
        for site_state in state.all_site_states:
            if site_state['_disc_occupancy'] == phase_name:
                count += 1
        return count

    def cell_ratio(self, step, p1, p2):
        return self.cell_ratio(step, p1) / self.cell_ratio(step, p2)

    def phase_count(self, step):
        return len(np.unique(step.state)) - 1

    def phases_present(self, state: SimulationState):
        return list(set([site_state['_disc_occupancy'] for site_state in state.all_site_states()]))
