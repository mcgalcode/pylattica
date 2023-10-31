from pylattica.core import SynchronousRunner
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY
from pylattica.models.game_of_life import Maze, Anneal, Diamoeba, Seeds, Life, GameOfLifeController
from pylattica.discrete import PhaseSet
from pylattica.structures.square_grid.grid_setup import DiscreteGridSetup

def test_gol_variants():
    variants = [Life, Maze, Anneal, Diamoeba, Seeds]
    for variant in variants:
        phases = PhaseSet(["dead", "alive"])
        setup = DiscreteGridSetup(phases)
        simulation = setup.setup_noise(10, ["dead", "alive"])
        controller = variant(structure=simulation.structure)
        runner = SynchronousRunner(parallel=True)
        runner.run(simulation.state, controller, 10, verbose=False)

def test_gol_update_rule():
    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases)
    simulation = setup.setup_interface(10, "dead", "alive")    
    controller = GameOfLifeController(structure=simulation.structure)
    controller.pre_run(None)
    site_id = simulation.structure.site_at((4,4))['_site_id']
    update = controller.get_state_update(site_id, simulation.state)
    assert update[DISCRETE_OCCUPANCY] == "alive"
    

