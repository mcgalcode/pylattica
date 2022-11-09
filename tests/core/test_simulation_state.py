from pylattica.core import Runner
from pylattica.core.analyzer import StateAnalyzer
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY
from pylattica.models.growth import GrowthController
from pylattica.discrete import PhaseSet
from pylattica.square_grid import SimpleSquare2DStructureBuilder, DiscreteGridSetup
from pylattica.square_grid.neighborhoods import CircularNeighborhoodBuilder, PseudoHexagonalNeighborhoodBuilder2D, PseudoPentagonalNeighborhoodBuilder


def test_can_run_growth_sim_series():
    phases = PhaseSet(["A", "B", "C", "D"])
    nb_spec = PseudoHexagonalNeighborhoodBuilder2D()
    setup = DiscreteGridSetup(phases)
    periodic_initial_state = setup.setup_random_sites(20, 20, "A", ["B", "C", "D"], [1, 1, 1])
    assert 'SITES' not in periodic_initial_state.state.site_ids()