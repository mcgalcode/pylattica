from pylattica.core import SynchronousRunner
from pylattica.core.analyzer import StateAnalyzer
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY
from pylattica.models.growth import GrowthController
from pylattica.discrete import PhaseSet
from pylattica.structures.square_grid import DiscreteGridSetup
from pylattica.structures.square_grid.neighborhoods import MooreNbHoodBuilder

def test_can_run_growth_sim_parallel():
    phases = PhaseSet(["A", "B", "C", "D"])
    nb_spec = MooreNbHoodBuilder()
    setup = DiscreteGridSetup(phases)
    periodic_initial_state = setup.setup_coords(20, "A", 
        {
            "B": [(10, 10)]
        }
    )
    controller = GrowthController(
        phases,
        periodic_initial_state.structure,
        nb_builder=nb_spec,
        background_phase="A"
    )
    runner = SynchronousRunner(parallel=True)
    res = runner.run(periodic_initial_state.state, controller, num_steps = 3)

    analyzer = StateAnalyzer(periodic_initial_state.structure)

    assert analyzer.get_site_count_where_equal(res.get_step(1), {
        DISCRETE_OCCUPANCY: "B"
    }) == 9

    assert analyzer.get_site_count_where_equal(res.get_step(2), {
        DISCRETE_OCCUPANCY: "B"
    }) == 25

def test_can_run_growth_sim_series():
    phases = PhaseSet(["A", "B", "C", "D"])
    nb_spec = MooreNbHoodBuilder()
    setup = DiscreteGridSetup(phases)
    periodic_initial_state = setup.setup_coords(20, "A", 
        {
            "B": [(10, 10)]
        }
    )
    controller = GrowthController(
        phases,
        periodic_initial_state.structure,
        nb_builder=nb_spec,
        background_phase="A"
    )
    runner = SynchronousRunner()
    res = runner.run(periodic_initial_state.state, controller, num_steps = 3)

    analyzer = StateAnalyzer(periodic_initial_state.structure)

    assert analyzer.get_site_count_where_equal(res.get_step(1), {
        DISCRETE_OCCUPANCY: "B"
    }) == 9

    assert analyzer.get_site_count_where_equal(res.get_step(2), {
        DISCRETE_OCCUPANCY: "B"
    }) == 25