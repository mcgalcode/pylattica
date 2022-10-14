import typing
import numpy as np

from ..core.neighborhoods import NeighborGraph
from ..core.periodic_structure import PeriodicStructure
from ..core.simulation_state import SimulationState, SimulationStep
from ..core.constants import SITE_ID, LOCATION
from ..discrete.phase_set import PhaseSet
from ..discrete.state_constants import DISCRETE_OCCUPANCY, VACANT
import random

from ..core.distance_map import distance
from .neighborhoods import MooreNbHoodSpec

class DiscreteGridSetup():
    """A class for setting up states. Provides helper methods for creating starting
    states in specific shapes
    """

    def __init__(self, phase_set: PhaseSet):
        """Initializes a laboratory object by providing a reaction set that is used to
        specify cell states

        Args:
            rxn_set (ScoredReactionSet):
        """
        self.phase_set: PhaseSet = phase_set


    def _build_blank_state(self, structure: PeriodicStructure, fill  = None) -> SimulationStep:
        state = SimulationState()
        for site in structure.sites():
            state.set_site_state(site[SITE_ID], {
                DISCRETE_OCCUPANCY: fill
            })
        return state

    def setup_solid_phase(self, structure: PeriodicStructure, phase_name: str) -> SimulationState:
        state = self._build_blank_state(structure, phase_name)
        return state

    def setup_interface(self, structure: PeriodicStructure, p1: str, p2: str) -> SimulationStep:
        """Generates a starting state that is divided into two phases. One phase
        occupies the left half of the state, and one phase occupies the right half of the state

        Args:
            size (int): The side length of the state
            p1 (str): The name of the left phase
            p2 (str): The name of the right phase

        Returns:
            SimulationState:
        """
        state: SimulationState = self._build_blank_state(structure)
        half: int = int(structure.bounds[0] / 2)
        for site in structure.sites():
            if site[LOCATION][0] <= half:
                state.set_site_state(site[SITE_ID], { DISCRETE_OCCUPANCY: p1 })
            else:
                state.set_site_state(site[SITE_ID], { DISCRETE_OCCUPANCY: p2 })
        return state

    def setup_particle(self, structure: PeriodicStructure, radius: int, bulk_phase: str, particle_phase: str) -> SimulationState:
        """Generates a starting state with a bulk phase surrounding a particle in the
        center of the state.

        Args:
            size (int): The side length of the state
            radius (int): The radius of the particle
            bulk_phase (str): The name of the bulk
            particle_phase (str): The name of the particle phase

        Returns:
            SimulationState:
        """
        state: SimulationState = self.setup_solid_phase(structure, bulk_phase)
        center: typing.Tuple[int, int] = tuple([structure.bounds[0] / 2 for _ in range(structure.dim)])
        state: SimulationState = self.add_particle_to_state(structure, state, center, radius, particle_phase)
        return state

    def setup_random_particles(self, structure: PeriodicStructure, radius: int, num_particles: int, bulk_phase: str, particle_phases: str) -> SimulationState:
        """Generates a starting state with a one phase in the background and num_particles particles distributed
        onto it randomly

        Args:
            size (int): The size of the state
            radius (int): The radius of the particles to drop
            num_particles (int): The number of particles
            bulk_phase (str): The name of the containing phase
            particle_phases (str): The name of the particulate phase

        Returns:
            SimulationState:
        """
        state: SimulationState = self.setup_solid_phase(structure, bulk_phase)
        for _ in range(num_particles):
            rand_coords = tuple([np.random.choice(int(structure.bounds[0])) for _ in range(structure.dim)])
            phase: str = random.choice(particle_phases)
            state: np.array = self.add_particle_to_state(structure, state, rand_coords, radius, phase)

        return state

    def add_particle_to_state(self, structure: PeriodicStructure, state: SimulationState, center: tuple, radius: int, particle_phase: str) -> np.array:
        for site in structure.sites():
            if distance(np.array(site[LOCATION]), np.array(center)) < radius:
                state.set_site_state(site[SITE_ID], {DISCRETE_OCCUPANCY: particle_phase})
        return state

    def setup_coords(self, structure: PeriodicStructure, background_state: str, coordinates: dict):
        state: np.array = self.setup_solid_phase(structure, background_state)
        for phase, coord_list in coordinates:
            for coords in coord_list:
                site_id = structure.site_at(coords)
                state.set_site_state(site_id, { DISCRETE_OCCUPANCY: phase })

    def setup_noise(self, structure: PeriodicStructure, phases: typing.List[str]):
        state: SimulationState = self._build_blank_state(structure)
        for site in structure.sites():
            state.set_site_state(site[SITE_ID], { DISCRETE_OCCUPANCY: random.choice(phases) })
        return state

    def setup_random_sites(self,
                           structure: PeriodicStructure,
                           num_sites_desired: int,
                           background_spec: str,
                           nuc_species: typing.List[str],
                           nuc_ratios: typing.List[float] = None,
                           buffer: int = 2):
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
        state = self.setup_solid_phase(structure, background_spec)
        nb_spec: MooreNbHoodSpec = MooreNbHoodSpec(buffer, dim = structure.dim)
        nb_graph: NeighborGraph = nb_spec.get(structure)
        all_sites = structure.sites()

        if nuc_ratios is None:
            nuc_ratios = np.ones((len(nuc_species)))

        specie_idxs = np.array(range(0, len(nuc_species)))
        normalized_ratios = nuc_ratios / np.sum(nuc_ratios)

        total_attempts = 0
        num_sites_planted = 0

        while num_sites_planted < num_sites_desired:
            if total_attempts > 100 * num_sites_desired:
                raise RuntimeError(f'Too many nucleation sites at the specified buffer: {total_attempts} made at placing nuclei')

            rand_site = random.choice(all_sites)
            rand_site_id = rand_site[SITE_ID]
            if state.get_site_state(rand_site_id)[DISCRETE_OCCUPANCY] != background_spec:
                total_attempts += 1
                continue

            found_existing_nucleus_in_nb = False

            for nb_site_id in nb_graph.neighbors_of(rand_site_id):
                if state.get_site_state(nb_site_id)[DISCRETE_OCCUPANCY] != background_spec:
                    found_existing_nucleus_in_nb = True
            if not found_existing_nucleus_in_nb:
                chosen_spec = nuc_species[np.random.choice(specie_idxs, p=normalized_ratios)]
                state.set_site_state(rand_site_id, { DISCRETE_OCCUPANCY: chosen_spec })
                num_sites_planted += 1

            total_attempts += 1

        return state
