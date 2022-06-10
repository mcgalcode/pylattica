import numpy as np

def pad_experiment(experiment, _filter):
    padding = int(_filter[0] / 2)
    experiment = np.pad(experiment, padding)
    return experiment

def build_blank_experiment(size):
    experiment = np.zeros((size, size))
    return experiment

def prepare_interface(size, _filter, p1, p2):
    experiment = build_blank_experiment(size)
    half = int(size/2)
    experiment[:,0:half] = p1
    experiment[:,half:size] = p2
    return pad_experiment(experiment, _filter)

def prepare_particle(size, _filter, radius, bulk_phase, particle_phase):
    experiment = build_blank_experiment(size)
    experiment[:,:] = bulk_phase
    center = int(size/2)
    add_particle_to_experiment(center, radius, particle_phase)
    return pad_experiment(experiment, _filter)

def add_particle_to_experiment(center, radius, particle_phase):
    border_u = 0
    border_d = experiment.shape[0]
    border_l = 0
    border_r = experiment.shape[1]
    p_ub = max(border_u, center - radius)
    p_db = min(border_d, center + radius)
    p_lb = max(border_l, center - radius)
    p_rb = min(border_r, center + radius)
    experiment[p_ub:p_lb, p_eb:p_rb] = particle_phase
    return experiment

def prepare_mixture(num_cells, _filter, grain_size, phase1, phase2):
    side_length = num_cells * grain_size * 2
    backdrop = np.zeros((side_length, side_length))
    phase1_cell = np.ones((grain_size, grain_size)) * phase1
    phase2_cell = np.ones((grain_size, grain_size)) * phase2
    tile = np.concatenate(
        [
            np.concatenate([phase1_cell, phase2_cell]).T.squeeze(),
            np.concatenate([phase2_cell, phase1_cell]).T.squeeze(),
        ]
    )
    result = np.tile(tile, (num_cells, num_cells))
    # result = np.concatenate(
    #     [
    #         np.concatenate([phase1_cell, phase2_cell, phase1_cell, phase2_cell]).T.squeeze(),
    #         np.concatenate([phase2_cell, phase1_cell, phase2_cell, phase1_cell]).T.squeeze(),
    #         np.concatenate([phase1_cell, phase2_cell, phase1_cell, phase2_cell]).T.squeeze(),
    #         np.concatenate([phase2_cell, phase1_cell, phase2_cell, phase1_cell]).T.squeeze(),            
    #     ]
    # )
    return pad_experiment(result, _filter)
