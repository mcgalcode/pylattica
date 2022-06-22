import typing
import numpy as np
from ..discrete import PhaseMap

from .scored_reaction_set import ScoredReactionSet
from .reaction_step import ReactionStep
import random


class Laboratory():
    """A class for setting up experiments. Provides helper methods for creating starting
    states in specific shapes
    """

    def __init__(self, rxn_set: ScoredReactionSet, phase_map: PhaseMap):
        """Initializes a laboratory object by providing a reaction set that is used to
        specify cell states

        Args:
            rxn_set (ScoredReactionSet):
        """
        self.rxn_set: ScoredReactionSet = rxn_set
        self.phase_map: PhaseMap = phase_map

    def _build_blank_experiment(self, size: int) -> np.array:
        experiment: np.array = np.ones((size, size)) * self.phase_map.free_space_id
        return experiment

    def prepare_interface(self, size: int, p1: str, p2: str) -> ReactionStep:
        """Generates a starting state that is divided into two phases. One phase
        occupies the left half of the state, and one phase occupies the right half of the state

        Args:
            size (int): The side length of the state
            p1 (str): The name of the left phase
            p2 (str): The name of the right phase

        Returns:
            ReactionStep:
        """
        experiment: np.array = self._build_blank_experiment(size)
        half: int = int(size/2)
        experiment[:,0:half] = self.phase_map.phase_to_int[p1]
        experiment[:,half:size] = self.phase_map.phase_to_int[p2]
        return ReactionStep(experiment)

    def prepare_particle(self, size: int, radius: int, bulk_phase: str, particle_phase: str) -> ReactionStep:
        """Generates a starting state with a bulk phase surrounding a particle in the
        center of the state.

        Args:
            size (int): The side length of the state
            radius (int): The radius of the particle
            bulk_phase (str): The name of the bulk
            particle_phase (str): The name of the particle phase

        Returns:
            ReactionStep:
        """
        experiment: np.array = self._build_blank_experiment(size)
        experiment[:,:] = self.phase_map.phase_to_int[bulk_phase]
        center: typing.Tuple[int, int] = (int(size/2), int(size/2))
        experiment: np.array = self._add_particle_to_experiment(experiment, center, radius, particle_phase)
        return ReactionStep(experiment)

    def prepare_random_particles(self, size: int, radius: int, num_particles: int, bulk_phase: str, particle_phases: str) -> ReactionStep:
        """Generates a starting state with a one phase in the background and num_particles particles distributed
        onto it randomly

        Args:
            size (int): The size of the state
            radius (int): The radius of the particles to drop
            num_particles (int): The number of particles
            bulk_phase (str): The name of the containing phase
            particle_phases (str): The name of the particulate phase

        Returns:
            ReactionStep:
        """
        experiment: np.array = self._build_blank_experiment(size)
        experiment[:,:] = self.phase_map.phase_to_int[bulk_phase]
        for _ in range(num_particles):
            rand_x: int = np.random.choice(size)
            rand_y: int = np.random.choice(size)
            phase: str = random.choice(particle_phases)
            experiment: np.array = self._add_particle_to_experiment(experiment, (rand_x, rand_y), radius, phase)

        return ReactionStep(experiment)

    def _add_particle_to_experiment(self, experiment: np.array, center: typing.Tuple[int, int], radius: int, particle_phase: str) -> np.array:
        border_u: int = 0
        border_d: int = experiment.shape[0]
        border_l: int = 0
        border_r: int = experiment.shape[1]
        p_ub: int = max(border_u, center[0] - radius)
        p_db: int = min(border_d, center[0] + radius + 1)
        p_lb: int = max(border_l, center[1] - radius)
        p_rb: int = min(border_r, center[1] + radius + 1)
        for i in range(p_ub, p_db):
            for j in range(p_lb, p_rb):
                if np.abs(i - center[0]) + np.abs(j - center[1]) <= radius:
                    experiment[i][j] = self.phase_map.phase_to_int[particle_phase]
        return experiment

    def prepare_even_mixture(self, num_cells, grain_size, phase1, phase2):
        phase1_cell = np.ones((grain_size, grain_size)) * self.phase_map.phase_to_int[phase1]
        phase2_cell = np.ones((grain_size, grain_size)) * self.phase_map.phase_to_int[phase2]
        tile = np.concatenate(
            [
                np.concatenate([phase1_cell, phase2_cell]).T.squeeze(),
                np.concatenate([phase2_cell, phase1_cell]).T.squeeze(),
            ]
        )
        result = np.tile(tile, (num_cells, num_cells))
        return ReactionStep(result)

    def prepare_random_mixture(self, side_length, grain_size, phases, weights = None):
        cells = []
        for p in phases:
            cells.append(np.ones((grain_size, grain_size)) * self.phase_map.phase_to_int[p])

        rows = []
        if weights is None:
            p = np.array([1 for _ in phases]) / len(phases)
        else:
            p = np.array(weights) / np.sum(weights)

        choices = list(range(len(phases)))

        for _ in range(side_length):
            rows.append(np.concatenate([cells[np.random.choice(choices, p = p)] for _ in range(side_length)]).T.squeeze())


        result = np.concatenate(rows)
        return ReactionStep(result)
