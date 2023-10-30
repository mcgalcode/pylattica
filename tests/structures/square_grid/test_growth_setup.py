from pylattica.structures.square_grid.growth_setup import GrowthSetup
from pylattica.structures.square_grid.neighborhoods import MooreNbHoodBuilder
from pylattica.discrete import PhaseSet
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY
from pylattica.core import StateAnalyzer

def test_growth_setup():
    phases = PhaseSet(["A", "B", "C"])

    simulation_side_length = 20
    total_num_sites = 4
    background_phase = "A"
    site_phases = ["B", "C"]
    buffer = 2 # Each site should be at least 2 cells away from any other

    growth_setup = GrowthSetup(phases)
    simulation = growth_setup.grow(
        simulation_side_length,
        background_spec=background_phase,
        num_sites_desired=total_num_sites,
        nuc_species=site_phases,
        buffer=buffer,
        nb_builder=MooreNbHoodBuilder(1)
    )

    analyzer = StateAnalyzer(simulation.structure)
    assert analyzer.get_site_count_where_equal(simulation.state, { DISCRETE_OCCUPANCY: "A" }) == 0


