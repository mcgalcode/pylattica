import numpy as np

class ReactionStep():
    
    def __init__(self, state, _filter, lab):
        self.state = state
        self.shape = state.shape
        self.filter = _filter
        self._total = np.count_nonzero(self.state != 0)
        self.lab = lab
    
    def phase_fraction(self, phase_name):
        phase_int = self.lab.phase_to_int[phase_name]
        filtered = np.count_nonzero(self.state == phase_int)
        return filtered / self._total
    
    @property
    def phase_count(self):
        return len(np.unique(self.state)) - 1
    
    @property
    def phases_present():
        return [self.lab.into_to_phase[p] for p in np.unique(self.state)]
    
    def as_phase_name_array(self):
        phase_name_map = np.empty(self.state.shape, dtype=np.str_)
        for phase_idx in self.lab.int_to_phase.keys():
            phase_name_map[self.state == phase_idx] = self.lab.int_to_phase[phase_idx]
        return phase_name_map