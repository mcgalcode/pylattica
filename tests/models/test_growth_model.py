from pylattica.core import Runner
from pylattica.core.analyzer import StateAnalyzer
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY
from pylattica.models.growth import GrowthController
from pylattica.discrete import PhaseSet
from pylattica.square_grid import SimpleSquare2DStructureBuilder, DiscreteGridSetup
from pylattica.square_grid.neighborhoods import CircularNeighborhoodBuilder, PseudoHexagonalNeighborhoodBuilder2D, PseudoPentagonalNeighborhoodBuilder

def test_can_run_growth_sim_parallel():
    phases = PhaseSet(["A", "B", "C", "D"])
    nb_spec = PseudoHexagonalNeighborhoodBuilder2D()
    setup = DiscreteGridSetup(phases)
    periodic_initial_state = setup.setup_random_sites(20, 20, "A", ["B", "C", "D"], [1, 1, 1])
    controller = GrowthController(
        phases,
        periodic_initial_state.structure,
        neighborhood_spec=nb_spec,
        background_phase="A"
    )
    runner = Runner(parallel=True)
    res = runner.run(periodic_initial_state.state, controller, num_steps = 50)

    analyzer = StateAnalyzer(periodic_initial_state.structure)

    assert analyzer.get_site_count_where_equal(res.last_step, {
        DISCRETE_OCCUPANCY: "A"
    }) == 0

def test_can_run_growth_sim_series():
    phases = PhaseSet(["A", "B", "C", "D"])
    nb_spec = PseudoHexagonalNeighborhoodBuilder2D()
    setup = DiscreteGridSetup(phases)
    periodic_initial_state = setup.setup_random_sites(20, 20, "A", ["B", "C", "D"], [1, 1, 1])
    controller = GrowthController(
        phases,
        periodic_initial_state.structure,
        neighborhood_spec=nb_spec,
        background_phase="A"
    )
    runner = Runner(parallel=False)
    res = runner.run(periodic_initial_state.state, controller, num_steps = 50)

    analyzer = StateAnalyzer(periodic_initial_state.structure)

    assert analyzer.get_site_count_where_equal(res.last_step, {
        DISCRETE_OCCUPANCY: "A"
    }) == 0