from pylattica.core import Runner
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY
from pylattica.models.game_of_life import Maze, Anneal, Diamoeba, Seeds, Life, GameOfLifeController
from pylattica.discrete import PhaseSet
from pylattica.square_grid.grid_setup import DiscreteGridSetup

def test_gol_variants():
    variants = [Life, Maze, Anneal, Diamoeba, Seeds]
    for variant in variants:
        phases = PhaseSet(["dead", "alive"])
        setup = DiscreteGridSetup(phases)
        starting_state = setup.setup_noise(10, ["dead", "alive"])
        controller = variant()
        runner = Runner(parallel=True)
        runner.run(starting_state.state, controller, 10, structure=starting_state.structure, verbose=False)

def test_gol_update_rule():
    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases)
    simulation = setup.setup_interface(10, "dead", "alive")    
    controller = GameOfLifeController()
    controller.pre_run(None, simulation.structure)
    site_id = simulation.structure.site_at((4,4))['_site_id']
    update = controller.get_state_update(site_id, simulation.state)
    assert update[DISCRETE_OCCUPANCY] == "alive"
    

