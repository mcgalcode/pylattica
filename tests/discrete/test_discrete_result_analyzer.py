from pylattica.core import Runner
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY
from pylattica.models.game_of_life import Maze, Anneal, Diamoeba, Seeds, Life, GameOfLifeController
from pylattica.discrete import PhaseSet, DiscreteResultAnalyzer
from pylattica.square_grid.grid_setup import DiscreteGridSetup

def _get_result():
    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases)
    simulation = setup.setup_noise(10, ["dead", "alive"])
    controller = Life()
    runner = Runner(parallel=True)
    return runner.run(simulation.state, controller, 10, structure=simulation.structure, verbose=False)

def test_plot_phase_fractions():
    result = _get_result()
    analyzer = DiscreteResultAnalyzer(result)
    analyzer.plot_phase_fractions()

def test_final_phase_fractions():
    result = _get_result()
    analyzer = DiscreteResultAnalyzer(result)
    fracs = analyzer.final_phase_fractions()
    for phase, amt in fracs.items():
        assert type(phase) is str
        assert type(amt) is float

