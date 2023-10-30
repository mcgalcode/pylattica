from typing import Dict
from pylattica.core import StateAnalyzer, PeriodicStructure, SimulationState, analyzer
from pylattica.core.constants import SITE_ID
from pylattica.structures.square_grid import DiscreteGridSetup
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY

import pytest

def test_analyze_get_sites_arb_criteria(square_grid_2D_4x4: PeriodicStructure):
    state = SimulationState()
    for idx, site in enumerate(square_grid_2D_4x4.sites()):
        state.set_site_state(site[SITE_ID], { "trait": idx })
    
    analyzer = StateAnalyzer(square_grid_2D_4x4)

    def _criteria_1(state: Dict) -> bool:
        return state.get('trait') >= 12

    def _criteria_2(state: Dict) -> bool:
        return state.get('trait') <= 15

    sites = analyzer.get_sites(state, state_criteria=[_criteria_1, _criteria_2])
    assert len(sites) == 4

def test_analyze_get_sites_no_criteria(square_grid_2D_4x4: PeriodicStructure):
    state = SimulationState()
    for idx, site in enumerate(square_grid_2D_4x4.sites()):
        state.set_site_state(site[SITE_ID], { "trait": idx })
    
    analyzer = StateAnalyzer(square_grid_2D_4x4) 

    sites = analyzer.get_sites(state)
    assert len(sites) == len(state.all_site_states())   
        

def test_analyze_count_equal(grid_setup: DiscreteGridSetup):
    periodic_state = grid_setup.setup_interface(4, 'A', 'B')
    analyzer = StateAnalyzer(periodic_state.structure)

    assert analyzer.get_site_count_where_equal(periodic_state.state, 
        {
            DISCRETE_OCCUPANCY: "A"
        }
    ) == 8

    assert analyzer.get_site_count_where_equal(periodic_state.state, 
        {
            DISCRETE_OCCUPANCY: "B"
        }
    ) == 8

    assert analyzer.get_site_count_where_equal(periodic_state.state, 
        {
            DISCRETE_OCCUPANCY: "C"
        }
    ) == 0

def test_analyze_get_sites_where_equal(grid_setup: DiscreteGridSetup):
    periodic_state = grid_setup.setup_interface(4, 'A', 'B')

    analyzer = StateAnalyzer(periodic_state.structure)

    a_site_ids = analyzer.get_sites_where_equal(periodic_state.state,
        {
            DISCRETE_OCCUPANCY: "A"
        }
    )

    for site_id in a_site_ids:
        reretrieved_site = periodic_state.state.get_site_state(site_id)
        assert reretrieved_site[DISCRETE_OCCUPANCY] == "A"

        a_site_ids = analyzer.get_sites_where_equal(periodic_state.state,
        {
            DISCRETE_OCCUPANCY: "A"
        }
    )

    b_site_ids = analyzer.get_sites_where_equal(periodic_state.state,
        {
            DISCRETE_OCCUPANCY: "B"
        }
    )

    for site_id in b_site_ids:
        reretrieved_site = periodic_state.state.get_site_state(site_id)
        assert reretrieved_site[DISCRETE_OCCUPANCY] == "B"