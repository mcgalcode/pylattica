from pylattica.core import Runner
from pylattica.models.growth import GrowthController
from pylattica.discrete import PhaseSet
from pylattica.square_grid import SimpleSquare2DStructureBuilder, DiscreteGridSetup
from pylattica.square_grid.neighborhoods import CircularNeighborhoodBuilder, PseudoHexagonalNeighborhoodBuilder, PseudoPentagonalNeighborhoodBuilder

def test_can_run_growth_sim():
    structure = SimpleSquare2DStructureBuilder().build(20)
    phases = PhaseSet(["A", "B", "C", "D"])
    nb_spec = PseudoHexagonalNeighborhoodBuilder()
    controller = GrowthController(phases, structure, nb_spec, background_phase = "A")
    setup = DiscreteGridSetup(phases)
    initial_state = setup.setup_random_sites(structure, 20, "A", ["B", "C", "D"], [1, 1, 1])
    runner = Runner(parallel=True)
    res = runner.run(initial_state, controller, num_steps = 10)
    assert res is not None