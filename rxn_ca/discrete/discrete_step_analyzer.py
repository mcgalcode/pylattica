import numpy as np

from rxn_ca.core.basic_simulation_step import BasicSimulationStep

from .phase_map import PhaseMap

class DiscreteStepAnalyzer():

    def __init__(self, phase_map: PhaseMap) -> None:
        self.phase_map: PhaseMap = phase_map

    def cell_fraction(self, step: BasicSimulationStep, phase_name: str):
        phase_count = self.cell_count(step, phase_name)
        total_occupied_cells = step.size ** step.dim
        return phase_count / total_occupied_cells

    def cell_count(self, step, phase_name: str):
        phase_int = self.phase_map.get_state_value(phase_name)
        filtered = np.count_nonzero(step.state == phase_int)
        return filtered

    def cell_ratio(self, step, p1, p2):
        return self.cell_ratio(step, p1) / self.cell_ratio(step, p2)

    def species_at(self, step, i, j):
        return self.phase_map.get_state_name(step.state[i, j])

    def phase_count(self, step):
        return len(np.unique(step.state)) - 1

    def phases_present(self, step):
        return [self.phase_map.get_state_name(p) for p in np.unique(step.state) if self.phase_map.is_valid_state_value(p)]

    def phases_present_in_state(self, state):
        return [self.phase_map.get_state_name(p) for p in np.unique(state) if self.phase_map.is_valid_state_value(p)]
