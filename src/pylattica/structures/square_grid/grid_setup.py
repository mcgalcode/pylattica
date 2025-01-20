import random
import typing

import numpy as np

from ...core.constants import LOCATION, SITE_ID
from ...core.distance_map import distance
from ...core.neighborhoods import Neighborhood
from ...core.simulation import Simulation
from ...core.periodic_structure import PeriodicStructure
from ...core.simulation_state import SimulationState
from ...discrete.phase_set import PhaseSet
from ...discrete.state_constants import DISCRETE_OCCUPANCY
from .neighborhoods import MooreNbHoodBuilder
from .structure_builders import (
    SimpleSquare2DStructureBuilder,
    SimpleSquare3DStructureBuilder,
)

from itertools import cycle


class DiscreteGridSetup:
    """A class for setting up states. Provides helper methods for creating starting
    states in specific shapes
    """

    def __init__(self, phase_set: PhaseSet, dim: int = 2):
        """Initializes a DiscreteGridSetup object by providing a set of phases that is used to
        specify cell states

        Parameters
        ----------
        phase_set : PhaseSet
            A PhaseSet instance containing the possible phases that might be assigned to cells.
        dim : int, optional
            The dimension of the simulations to setup, by default 2
        """
        if dim == 2:
            self._builder = SimpleSquare2DStructureBuilder()
        elif dim == 3:
            self._builder = SimpleSquare3DStructureBuilder()
        self.phase_set: PhaseSet = phase_set

    def _build_blank_state(
        self, structure: PeriodicStructure, fill=None
    ) -> SimulationState:
        state = SimulationState()
        for site in structure.sites():
            state.set_site_state(site[SITE_ID], {DISCRETE_OCCUPANCY: fill})
        return state

    def build_structure(self, size: int) -> PeriodicStructure:
        """Constructs a structure without a state of the specified size

        Parameters
        ----------
        size : int
            The size of the structure to build

        Returns
        -------
        PeriodicStructure
            The resulting structure.
        """
        return self._builder.build(size)

    def setup_solid_phase(
        self, structure: PeriodicStructure, phase_name: str
    ) -> SimulationState:
        """Generates a simulation filled with the specified phase.

        Parameters
        ----------
        structure : PeriodicStructure
            The structure to
        phase_name : str
            The name of the phase to fill the structure with.

        Returns
        -------
        SimulationState
            The resulting homogeneous simulation state.
        """
        state = self._build_blank_state(structure, phase_name)
        return state

    def setup_interface(self, size: int, p1: str, p2: str) -> Simulation:
        """Generates a starting state that is divided into two phases. One phase
        occupies the left half of the state, and one phase occupies the right half of the state

        Parameters
        ----------
        size : int
            The side length of the state
        p1 : str
            The name of the left phase
        p2 : str
            The name of the right phase

        Returns
        -------
        Simulation
            The resulting Simulation.
        """
        structure = self._builder.build(size=size)
        state: SimulationState = self._build_blank_state(structure)
        half: int = int(structure.lattice.vec_lengths[0] / 2)
        for site in structure.sites():
            if site[LOCATION][0] < half:
                state.set_site_state(site[SITE_ID], {DISCRETE_OCCUPANCY: p1})
            else:
                state.set_site_state(site[SITE_ID], {DISCRETE_OCCUPANCY: p2})
        return Simulation(state, structure)

    def setup_particle(
        self, size: int, radius: int, bulk_phase: str, particle_phase: str
    ) -> Simulation:
        """Generates a starting state with a bulk phase surrounding a particle in the
        center of the state.

        Parameters
        ----------
        size : int
            The side length of the state
        radius : int
            The radius of the particle
        bulk_phase : str
            The name of the bulk
        particle_phase : str
            The name of the particle phase

        Returns
        -------
        Simulation
            The resulting Simulation
        """
        structure = self._builder.build(size=size)
        state: SimulationState = self.setup_solid_phase(structure, bulk_phase)
        center: typing.Tuple[int, int] = tuple(
            structure.lattice.vec_lengths[0] / 2 for _ in range(structure.dim)
        )
        state: SimulationState = self.add_particle_to_state(
            structure, state, center, radius, particle_phase
        )
        return Simulation(state, structure)

    def setup_random_particles(
        self,
        size: int,
        radius: int,
        num_particles: int,
        bulk_phase: str,
        particle_phases: str,
    ) -> Simulation:
        """Generates a starting state with a one phase in the background and num_particles particles distributed
        onto it randomly

        Parameters
        ----------
        size : int
            The size of the simulation
        radius : int
            The radius of the particles to drop
        num_particles : int
            The number of particles
        bulk_phase : str
            The name of the containing phase
        particle_phases : str
            The name of the particulate phase

        Returns
        -------
        Simulation
            The resulting Simulation.
        """
        structure = self._builder.build(size)
        state: SimulationState = self.setup_solid_phase(structure, bulk_phase)
        for _ in range(num_particles):
            rand_coords = tuple(
                np.random.choice(int(structure.lattice.vec_lengths[0]))
                for _ in range(structure.dim)
            )
            phase: str = random.choice(particle_phases)
            state: np.array = self.add_particle_to_state(
                structure, state, rand_coords, radius, phase
            )

        return Simulation(state, structure)

    def add_particle_to_state(
        self,
        structure: PeriodicStructure,
        state: SimulationState,
        center: tuple,
        radius: int,
        particle_phase: str,
    ) -> SimulationState:
        """Adds a region filled with the specified phase to the point specified in the provided structure.

        Parameters
        ----------
        structure : PeriodicStructure
            The structure in which the particle region should be added.
        state : SimulationState
            The state in which this change should be reflected
        center : tuple
            The coordinates of the center of the desired particle
        radius : int
            The radius of the desired particle
        particle_phase : str
            The phase of the desired particle

        Returns
        -------
        SimulationState
            The adjusted SimulationState
        """
        for site in structure.sites():
            if distance(np.array(site[LOCATION]), np.array(center)) < radius:
                state.set_site_state(
                    site[SITE_ID], {DISCRETE_OCCUPANCY: particle_phase}
                )
        return state

    def setup_coords(
        self, size: int, background_state: str, coordinates: dict
    ) -> Simulation:
        """Generates a simulation filled with a specifed background phase, and specific
        sites filled with specific phases as defined by the coordinates parameter.

        Parameters
        ----------
        size : int
            The size of the simulation to generate
        background_state : str
            The phase with which the background should be filled.
        coordinates : dict
            A map of phase name to list of tuples specifying the coordinates to be filled
            with that phase.

        Returns
        -------
        Simulation
            The resulting Simulation.
        """
        structure = self._builder.build(size)
        state: np.array = self.setup_solid_phase(structure, background_state)
        for phase, coord_list in coordinates.items():
            for coords in coord_list:
                site_id = structure.site_at(coords)[SITE_ID]
                state.set_site_state(site_id, {DISCRETE_OCCUPANCY: phase})
        return Simulation(state, structure)

    def setup_noise(self, size: int, phases: typing.List[str]) -> Simulation:
        """Generates an initial simulation state with sites randomly assigned one
        of the provided list of phases

        Parameters
        ----------
        size : int
            The size of the simulation to generate.
        phases : typing.List[str]
            The phases to be randomly assigned

        Returns
        -------
        Simulation
            The resulting simulation.
        """
        structure = self._builder.build(size)
        state: SimulationState = self._build_blank_state(structure)
        for site in structure.sites():
            state.set_site_state(
                site[SITE_ID], {DISCRETE_OCCUPANCY: random.choice(phases)}
            )
        return Simulation(state, structure)

    def setup_random_sites(
        self,
        size: int,
        num_sites_desired: int,
        background_spec: str,
        nuc_amts: typing.Dict[str, float],
        buffer: int = 2,
    ) -> Simulation:
        """Generates an initial simulation state with a background phase filling space and
        random sites filled in with other phases as specified by the provided ratios.

        Parameters
        ----------
        size : int
            The size of the simulation to generate.
        num_sites_desired : int
            The number of non-background sites to select
        background_spec : str
            The phase of the background
        nuc_amts : typing.Dict[str, float]
            The ratios of the phases to assign to the selected sites
        buffer : int, optional
            The minimum distance between selected sites, by default 2

        Returns
        -------
        Simulation
            The resulting Simulation

        Raises
        ------
        RuntimeError
            If the buffer between sites is too large, and the num_sites_desired cannot fit
            in the specifed simulation size, a RuntimeError is thrown
        """
        structure = self._builder.build(size)
        num_sites_desired = round(num_sites_desired)
        state = self.setup_solid_phase(structure, background_spec)
        if buffer is not None:
            nb_spec: MooreNbHoodBuilder = MooreNbHoodBuilder(buffer, dim=structure.dim)
            nb_graph: Neighborhood = nb_spec.get(structure)
        all_sites = structure.sites()

        nuc_species = []
        nuc_ratios = []

        for spec, amt in nuc_amts.items():
            nuc_species.append(spec)
            nuc_ratios.append(amt)

        normalized_ratios = np.array(nuc_ratios) / np.sum(nuc_ratios)

        total_attempts = 0
        num_sites_planted = 0

        ideal_num_site_list = normalized_ratios * num_sites_desired
        ideal_num_sites = dict(zip(nuc_species, ideal_num_site_list))

        nuc_identities = []

        real_num_sites = {p: 0 for p in nuc_species}

        spec_provider = cycle(nuc_species)
        while len(nuc_identities) < num_sites_desired:
            phase = next(spec_provider)

            if real_num_sites.get(phase) < ideal_num_sites.get(phase):
                nuc_identities.append(phase)
                real_num_sites[phase] += 1

        while num_sites_planted < num_sites_desired:
            if total_attempts > 1000 * num_sites_desired:
                print(f"Only able to place {num_sites_planted} in {total_attempts} attempts")
                break

            rand_site = random.choice(all_sites)
            rand_site_id = rand_site[SITE_ID]
            if (
                state.get_site_state(rand_site_id)[DISCRETE_OCCUPANCY]
                != background_spec
            ):
                total_attempts += 1
                continue

            found_existing_nucleus_in_nb = False

            if buffer is not None:
                for nb_site_id in nb_graph.neighbors_of(rand_site_id):
                    if (
                        state.get_site_state(nb_site_id)[DISCRETE_OCCUPANCY]
                        != background_spec
                    ):
                        found_existing_nucleus_in_nb = True

            if not found_existing_nucleus_in_nb:
                chosen_spec = nuc_identities[num_sites_planted]
                state.set_site_state(rand_site_id, {DISCRETE_OCCUPANCY: chosen_spec})

                num_sites_planted += 1

            total_attempts += 1

        return Simulation(state, structure)
