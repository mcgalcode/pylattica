import typing
import numpy as np
from rxn_ca.core.basic_simulation_step import BasicSimulationStep

from rxn_ca.core.neighborhoods import MooreNeighborhood, Neighborhood
from .phase_map import PhaseMap

from ..rxn.reaction_step import ReactionStep
import random


class DiscreteStateSetup():
    """A class for setting up states. Provides helper methods for creating starting
    states in specific shapes
    """

    def __init__(self, phase_map: PhaseMap, step_class = BasicSimulationStep):
        """Initializes a laboratory object by providing a reaction set that is used to
        specify cell states

        Args:
            rxn_set (ScoredReactionSet):
        """
        self.step_class = step_class
        self.phase_map: PhaseMap = phase_map

    def _build_blank_state(self, size: int, fill = None) -> np.array:
        state: np.array = np.ones((size, size))
        if fill is not None:
            state = state * fill
        return state

    def setup_interface(self, size: int, p1: str, p2: str) -> ReactionStep:
        """Generates a starting state that is divided into two phases. One phase
        occupies the left half of the state, and one phase occupies the right half of the state

        Args:
            size (int): The side length of the state
            p1 (str): The name of the left phase
            p2 (str): The name of the right phase

        Returns:
            ReactionStep:
        """
        state: np.array = self._build_blank_state(size)
        half: int = int(size/2)
        state[:,0:half] = self.phase_map.get_state_value(p1)
        state[:,half:size] = self.phase_map.get_state_value(p2)
        return self.step_class(state)

    def setup_particle(self, size: int, radius: int, bulk_phase: str, particle_phase: str) -> ReactionStep:
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
        state: np.array = self._build_blank_state(size)
        state[:,:] = self.phase_map.get_state_value(bulk_phase)
        center: typing.Tuple[int, int] = (int(size/2), int(size/2))
        state: np.array = self.add_particle_to_state(state, center, radius, particle_phase)
        return self.step_class(state)

    def setup_random_particles(self, size: int, radius: int, num_particles: int, bulk_phase: str, particle_phases: str) -> ReactionStep:
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
        state: np.array = self._build_blank_state(size)
        state[:,:] = self.phase_map.get_state_value(bulk_phase)
        for _ in range(num_particles):
            rand_x: int = np.random.choice(size)
            rand_y: int = np.random.choice(size)
            phase: str = random.choice(particle_phases)
            state: np.array = self.add_particle_to_state(state, (rand_x, rand_y), radius, phase)

        return self.step_class(state)

    def add_particle_to_state(self, state: np.array, center: typing.Tuple[int, int], radius: int, particle_phase: str) -> np.array:
        border_u: int = 0
        border_d: int = state.shape[0]
        border_l: int = 0
        border_r: int = state.shape[1]
        p_ub: int = max(border_u, center[0] - radius)
        p_db: int = min(border_d, center[0] + radius + 1)
        p_lb: int = max(border_l, center[1] - radius)
        p_rb: int = min(border_r, center[1] + radius + 1)
        for i in range(p_ub, p_db):
            for j in range(p_lb, p_rb):
                if np.abs(i - center[0]) + np.abs(j - center[1]) <= radius:
                    state[i][j] = self.phase_map.phase_to_int[particle_phase]
        return state

    def setup_even_mixture(self, num_cells, grain_size, phase1, phase2):
        phase1_cell = np.ones((grain_size, grain_size)) * self.phase_map.get_state_value(phase1)
        phase2_cell = np.ones((grain_size, grain_size)) * self.phase_map.get_state_value(phase2)
        tile = np.concatenate(
            [
                np.concatenate([phase1_cell, phase2_cell]).T.squeeze(),
                np.concatenate([phase2_cell, phase1_cell]).T.squeeze(),
            ]
        )
        result = np.tile(tile, (num_cells, num_cells))
        return ReactionStep(result)

    def setup_noise(self, size, phases):
        blank = self._build_blank_state(size)
        for i in range(size):
            for j in range(size):
                blank[i][j] = self.phase_map.get_state_value(random.choice(phases))
        return ReactionStep(blank)

    def setup_random_mixture(self, side_length, grain_size, phases, weights = None):
        cells = []
        for p in phases:
            cells.append(np.ones((grain_size, grain_size)) * self.phase_map.get_state_value(p))

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

    def setup_random_sites(self, size, num_sites_desired, background_spec, nuc_species, nuc_ratios = None, buffer = 2):
        bg_state = self.phase_map.get_state_value(background_spec)
        blank = self._build_blank_state(size, fill=bg_state)
        nb = MooreNeighborhood(buffer)


        if nuc_ratios is None:
            nuc_ratios = np.ones((len(nuc_species)))

        specie_idxs = np.array(range(0, len(nuc_species)))
        normalized_ratios = nuc_ratios / np.sum(nuc_ratios)

        total_attempts = 0
        num_sites_planted = 0

        while num_sites_planted < num_sites_desired:
            if total_attempts > 100 * num_sites_desired:
                raise RuntimeError(f'Too many nucleation sites at the specified buffer: {total_attempts} made at placing nuclei')

            rand_i = random.randint(0,size - 1)
            rand_j = random.randint(0,size - 1)
            if blank[rand_i][rand_j] != bg_state:
                total_attempts += 1
                continue

            found_existing_nucleus_in_nb = False
            view = nb.get(blank, [rand_i, rand_j])
            for cell, _ in view.iterate(exclude_center=False):
                if cell != bg_state:
                    found_existing_nucleus_in_nb = True
            if not found_existing_nucleus_in_nb:
                chosen_spec = nuc_species[np.random.choice(specie_idxs, p=normalized_ratios)]
                blank[rand_i][rand_j] = self.phase_map.get_state_value(chosen_spec)
                num_sites_planted += 1

            total_attempts += 1

        return self.step_class(blank)
