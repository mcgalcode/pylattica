import typing
import numpy as np
from rxn_ca.core.basic_simulation_step import BasicSimulationStep
from rxn_ca.core.distance_map import EuclideanDistanceMap, ManhattanDistanceMap

from rxn_ca.core.neighborhoods import MooreNeighborhood
from .phase_map import PhaseMap
import random


class DiscreteStateSetup3D():
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
        state: np.array = np.ones((size, size, size))
        if fill is not None:
            state = state * fill
        return state

    def setup_interface(self, size: int, p1: str, p2: str) -> BasicSimulationStep:
        """Generates a starting state that is divided into two phases. One phase
        occupies the left half of the state, and one phase occupies the right half of the state

        Args:
            size (int): The side length of the state
            p1 (str): The name of the left phase
            p2 (str): The name of the right phase

        Returns:
            BasicSimulationStep:
        """
        state: np.array = self._build_blank_state(size)
        half: int = int(size/2)
        state[:, :, 0:half] = self.phase_map.get_state_value(p1)
        state[:, :, half:size] = self.phase_map.get_state_value(p2)
        return self.step_class(state)

    def setup_particle(self, size: int, radius: int, bulk_phase: str, particle_phase: str) -> BasicSimulationStep:
        """Generates a starting state with a bulk phase surrounding a particle in the
        center of the state.

        Args:
            size (int): The side length of the state
            radius (int): The radius of the particle
            bulk_phase (str): The name of the bulk
            particle_phase (str): The name of the particle phase

        Returns:
            BasicSimulationStep:
        """
        state: np.array = self._build_blank_state(size)
        state[:,:, :] = self.phase_map.get_state_value(bulk_phase)
        center: typing.Tuple[int, int] = (int(size/2), int(size/2), int(size/2))
        state: np.array = self.add_particle_to_state(state, center, radius, particle_phase)
        return self.step_class(state)

    def setup_random_particles(self, size: int, radius: int, num_particles: int, bulk_phase: str, particle_phases: str) -> BasicSimulationStep:
        """Generates a starting state with a one phase in the background and num_particles particles distributed
        onto it randomly

        Args:
            size (int): The size of the state
            radius (int): The radius of the particles to drop
            num_particles (int): The number of particles
            bulk_phase (str): The name of the containing phase
            particle_phases (str): The name of the particulate phase

        Returns:
            BasicSimulationStep:
        """
        state: np.array = self._build_blank_state(size)
        state[:,:, :] = self.phase_map.get_state_value(bulk_phase)
        for _ in range(num_particles):
            rand_x: int = np.random.choice(size)
            rand_y: int = np.random.choice(size)
            rand_z: int = np.random.choice(size)
            phase: str = random.choice(particle_phases)
            state: np.array = self.add_particle_to_state(state, (rand_x, rand_y, rand_z), radius, phase)

        return self.step_class(state)

    def add_particle_to_state(self, state: np.array, center: typing.Tuple[int, int], radius: int, particle_phase: str) -> np.array:
        state_size: int = state.shape[0]
        distances = EuclideanDistanceMap(radius * 2 + 1, dimension=3)
        xmin: int = max(0, center[0] - radius)
        xmax: int = min(state_size, center[0] + radius)
        ymin: int = max(0, center[1] - radius)
        ymax: int = min(state_size, center[1] + radius)
        zmin: int = max(0, center[2] - radius)
        zmax: int = min(state_size, center[2] + radius)
        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                for k in range(zmin, zmax):
                    relative_pos = (i - center[0], j - center[1], k - center[2])
                    dist = distances.distances[relative_pos]
                    if dist >= radius:
                        state[(i, j, k)] = self.phase_map.phase_to_int[particle_phase]
        return state

    def setup_even_mixture(self, num_cells, grain_size, phase1, phase2):
        phase1_cell = np.ones((grain_size, grain_size, grain_size)) * self.phase_map.get_state_value(phase1)
        phase2_cell = np.ones((grain_size, grain_size, grain_size)) * self.phase_map.get_state_value(phase2)
        tile = np.concatenate(
            [
                np.concatenate([phase1_cell, phase2_cell]).T.squeeze(),
                np.concatenate([phase2_cell, phase1_cell]).T.squeeze(),
            ]
        )
        result = np.tile(tile, (num_cells, num_cells))
        return BasicSimulationStep(result)


    def setup_coords(self, size, background_state, coordinates):
        state: np.array = self._build_blank_state(size)
        state[:,:,:] = self.phase_map.get_state_value(background_state)
        for phase, coord_list in coordinates:
            for coords in coord_list:
                state[coords] = self.phase_map.get_state_value(phase)

    def setup_noise(self, size, phases):
        blank = self._build_blank_state(size)
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    blank[(i,j,k)] = self.phase_map.get_state_value(random.choice(phases))
        return BasicSimulationStep(blank)

    def setup_random_mixture(self, num_grains, grain_size, phases, weights = None):
        cells = []
        for p in phases:
            cells.append(np.ones((grain_size, grain_size, grain_size)) * self.phase_map.get_state_value(p))

        if weights is None:
            p = np.array([1 for _ in phases]) / len(phases)
        else:
            p = np.array(weights) / np.sum(weights)

        choices = list(range(len(phases)))

        sheets = []
        for _ in range(num_grains):
            rows = []
            for _ in range(num_grains):
                row = np.concatenate([cells[np.random.choice(choices, p = p)] for _ in range(num_grains)])
                rows.append(row)
            sheet = np.concatenate(rows, axis=1)
            sheets.append(sheet)

        result = np.concatenate(sheets, axis=2)
        return BasicSimulationStep(result)

    def setup_random_sites(self, size, num_sites_desired, background_spec, nuc_species, nuc_ratios = None, buffer = 2):
        """_summary_

        Args:
            size (_type_): _description_
            num_sites_desired (_type_): _description_
            background_spec (_type_): _description_
            nuc_species (_type_): _description_
            nuc_ratios (_type_, optional): _description_. Defaults to None.
            buffer (int, optional): _description_. Defaults to 2.

        Raises:
            RuntimeError: _description_

        Returns:
            _type_: _description_
        """
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
            rand_k = random.randint(0,size - 1)
            if blank[rand_i][rand_j][rand_k] != bg_state:
                total_attempts += 1
                continue

            found_existing_nucleus_in_nb = False
            view = nb.get(blank, [rand_i, rand_j, rand_k])
            for cell, _ in view.iterate(exclude_center=False):
                if cell != bg_state:
                    found_existing_nucleus_in_nb = True
            if not found_existing_nucleus_in_nb:
                chosen_spec = nuc_species[np.random.choice(specie_idxs, p=normalized_ratios)]
                blank[rand_i][rand_j][rand_k] = self.phase_map.get_state_value(chosen_spec)
                num_sites_planted += 1

            total_attempts += 1

        return self.step_class(blank)
