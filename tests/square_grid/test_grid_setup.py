from typing import Dict

from pylattica.core.analyzer import StateAnalyzer
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY
from pylattica.square_grid import DiscreteGridSetup


def test_can_instantiate_grid_setup(simple_phase_set):
    setup = DiscreteGridSetup(simple_phase_set)
    assert setup is not None

def test_setup_interface(grid_setup: DiscreteGridSetup):
    periodic_state = grid_setup.setup_interface(4, 'A', 'B')

    A_site = periodic_state.state_at((1,1))
    assert A_site[DISCRETE_OCCUPANCY] == 'A'

    A_site = periodic_state.state_at((1,2))
    assert A_site[DISCRETE_OCCUPANCY] == 'A'

    B_site = periodic_state.state_at((2,2))
    assert B_site[DISCRETE_OCCUPANCY] == 'B'

    B_site = periodic_state.state_at((2,3))
    assert B_site[DISCRETE_OCCUPANCY] == 'B'


def test_setup_random_particles(grid_setup: DiscreteGridSetup):
    state = grid_setup.setup_random_particles(4, radius = 2, num_particles = 3, bulk_phase = 'A', particle_phases = ['B'])
    assert state is not None

def test_setup_noise(grid_setup: DiscreteGridSetup):
    state = grid_setup.setup_noise(4, phases = ['A', 'B'])
    assert state is not None

def test_setup_random_sites(grid_setup: DiscreteGridSetup):
    num_sites = 2
    periodic_state = grid_setup.setup_random_sites(
        4,
        num_sites_desired = num_sites,
        background_spec='A',
        nuc_species = ['B', 'C'],
        buffer=1
    )

    analyzer = StateAnalyzer(periodic_state.structure)

    def count_criteria(state: Dict) -> bool:
        return state[DISCRETE_OCCUPANCY] == 'B' or state[DISCRETE_OCCUPANCY] == 'C'

    assert analyzer.get_site_count(periodic_state.state, state_criteria=[count_criteria]) == num_sites

def test_setup_specific_coords(grid_setup: DiscreteGridSetup):
    periodic_state = grid_setup.setup_coords(4, background_state='A', coordinates = {
        'B': [[0, 0]],
        'C': [[1,1]]
    })

   
    assert periodic_state.state_at((0,0))[DISCRETE_OCCUPANCY] == 'B'
    assert periodic_state.state_at((1,1))[DISCRETE_OCCUPANCY] == 'C'
    assert periodic_state.state_at((1,0))[DISCRETE_OCCUPANCY] == 'A'

