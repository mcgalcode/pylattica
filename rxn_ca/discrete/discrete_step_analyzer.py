import numpy as np
from .phase_map import PhaseMap

class DiscreteStepAnalyzer():

    def __init__(self, phase_map: PhaseMap) -> None:
        self.phase_map: PhaseMap = phase_map

    def cell_fraction(self, step, phase_name: str):
        phase_count = self.cell_count(step, phase_name)
        total_occupied_cells = step.size ** 2
        return phase_count / total_occupied_cells

    def cell_count(self, step, phase_name: str):
        phase_int = self.phase_map.phase_to_int[phase_name]
        filtered = np.count_nonzero(step.state == phase_int)
        return filtered

    def cell_ratio(self, step, p1, p2):
        return self.cell_ratio(step, p1) / self.cell_ratio(step, p2)

    def species_at(self, step, i, j):
        return self.phase_map.int_to_phase[step.state[i, j]]

    def phase_count(self, step):
        return len(np.unique(step.state)) - 1

    def phases_present(self, step):
        return [self.phase_map.int_to_phase[p] for p in np.unique(step.state)]
