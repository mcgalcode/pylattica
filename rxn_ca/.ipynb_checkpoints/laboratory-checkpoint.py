import numpy as np
from .reaction_step import ReactionStep
from .reaction import Reaction

class Laboratory():
    
    def __init__(self, reactions):
        phases = []
        for r in reactions:
            phases = phases + r.products + r.reactants
        
        self_reactions = [Reaction.self_reaction(phase) for phase in phases]
        phases = list(set(phases))
        phases = ["_free_space"] + phases
        self.int_to_phase = {}
        self.phase_to_int = {}  
        self.reactions = reactions + self_reactions
        for idx, phase_name in enumerate(phases):
            self.int_to_phase[idx] = phase_name
            self.phase_to_int[phase_name] = idx
    
    def get_reaction(self, r1, r2):
        for reaction in self.reactions:
            if reaction.can_proceed_with([r1, r2]):
                return reaction
    
        return None

    import numpy as np

    def pad_experiment(self, experiment, _filter):
        padding = int(_filter[0] / 2)
        experiment = np.pad(experiment, padding)
        return experiment

    def build_blank_experiment(self, size):
        experiment = np.zeros((size, size))
        return experiment

    def prepare_interface(self, size, _filter, p1, p2):
        experiment = self.build_blank_experiment(size)
        half = int(size/2)
        experiment[:,0:half] = self.phase_to_int[p1]
        experiment[:,half:size] = self.phase_to_int[p2]
        return ReactionStep(self.pad_experiment(experiment, _filter), _filter, self)

    def prepare_particle(self, size, _filter, radius, bulk_phase, particle_phase):
        experiment = self.build_blank_experiment(size)
        experiment[:,:] = self.phase_to_int[bulk_phase]
        center = (int(size/2), int(size/2)) 
        experiment = self.add_particle_to_experiment(experiment, center, radius, particle_phase)
        return ReactionStep(self.pad_experiment(experiment, _filter), _filter, self)
    
    def prepare_random_particles(self, size, _filter, radius, num_particles, bulk_phase, particle_phase):
        experiment = self.build_blank_experiment(size)
        experiment[:,:] = self.phase_to_int[bulk_phase]
        for i in range(num_particles):
            rand_x = np.random.choice(size)
            rand_y = np.random.choice(size)
            experiment = self.add_particle_to_experiment(experiment, (rand_x, rand_y), radius, particle_phase)
        
        return ReactionStep(self.pad_experiment(experiment, _filter), _filter, self)

    def add_particle_to_experiment(self, experiment, center, radius, particle_phase):
        border_u = 0
        border_d = experiment.shape[0]
        border_l = 0
        border_r = experiment.shape[1]
        p_ub = max(border_u, center[0] - radius)
        p_db = min(border_d, center[0] + radius + 1)
        p_lb = max(border_l, center[1] - radius)
        p_rb = min(border_r, center[1] + radius + 1)
        for i in range(p_ub, p_db):
            for j in range(p_lb, p_rb):
                if np.abs(i - center[0]) + np.abs(j - center[1]) <= radius:
                    experiment[i][j] = self.phase_to_int[particle_phase]
        return experiment

    def prepare_mixture(self, num_cells, _filter, grain_size, phase1, phase2):
        side_length = num_cells * grain_size * 2
        backdrop = np.zeros((side_length, side_length))
        print(self.phase_to_int)
        phase1_cell = np.ones((grain_size, grain_size)) * self.phase_to_int[phase1]
        phase2_cell = np.ones((grain_size, grain_size)) * self.phase_to_int[phase2]
        tile = np.concatenate(
            [
                np.concatenate([phase1_cell, phase2_cell]).T.squeeze(),
                np.concatenate([phase2_cell, phase1_cell]).T.squeeze(),
            ]
        )
        result = np.tile(tile, (num_cells, num_cells))
        return ReactionStep(self.pad_experiment(result, _filter), _filter, self)
