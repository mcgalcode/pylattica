from pylattica.core.analyzer import StateAnalyzer
from pylattica.core.periodic_state import PeriodicState
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY
from pylattica.core.periodic_structure import PeriodicStructure
import pytest
from typing import Dict

from pylattica.square_grid import DiscreteGridSetup

def test_can_instantiate_grid_setup(simple_phase_set):
    setup = DiscreteGridSetup(simple_phase_set)
    assert setup is not None

def test_setup_interface(square_grid_2D_4x4: PeriodicStructure, grid_setup: DiscreteGridSetup):
    state = grid_setup.setup_interface(square_grid_2D_4x4, 'A', 'B')
    periodic_state = PeriodicState(state, square_grid_2D_4x4)

    A_site = periodic_state.state_at((1,1))
    assert A_site[DISCRETE_OCCUPANCY] == 'A'

    A_site = periodic_state.state_at((1,2))
    assert A_site[DISCRETE_OCCUPANCY] == 'A'

    B_site = periodic_state.state_at((2,2))
    assert B_site[DISCRETE_OCCUPANCY] == 'B'

    B_site = periodic_state.state_at((2,3))
    assert B_site[DISCRETE_OCCUPANCY] == 'B'


def test_setup_random_particles(square_grid_2D_4x4: PeriodicStructure, grid_setup: DiscreteGridSetup):
    state = grid_setup.setup_random_particles(square_grid_2D_4x4, radius = 2, num_particles = 3, bulk_phase = 'A', particle_phases = ['B'])
    assert state is not None

def test_setup_noise(square_grid_2D_4x4: PeriodicStructure, grid_setup: DiscreteGridSetup):
    state = grid_setup.setup_noise(square_grid_2D_4x4, phases = ['A', 'B'])
    assert state is not None

def test_setup_random_sites(square_grid_2D_4x4: PeriodicStructure, grid_setup: DiscreteGridSetup):
    num_sites = 2
    state = grid_setup.setup_random_sites(
        square_grid_2D_4x4,
        num_sites_desired = num_sites,
        background_spec='A',
        nuc_species = ['B', 'C'],
        buffer=1
    )

    analyzer = StateAnalyzer(square_grid_2D_4x4)

    def count_criteria(state: Dict) -> bool:
        return state[DISCRETE_OCCUPANCY] == 'B' or state[DISCRETE_OCCUPANCY] == 'C'

    assert analyzer.get_site_count(state, state_criteria=[count_criteria]) == num_sites

def test_setup_specific_coords(square_grid_2D_4x4: PeriodicStructure, grid_setup: DiscreteGridSetup):
    state = grid_setup.setup_coords(square_grid_2D_4x4, background_state='A', coordinates = {
        'B': [[0, 0]],
        'C': [[1,1]]
    })

    periodic_state = PeriodicState(state, square_grid_2D_4x4)
    
    assert periodic_state.state_at((0,0))[DISCRETE_OCCUPANCY] == 'B'
    assert periodic_state.state_at((1,1))[DISCRETE_OCCUPANCY] == 'C'
