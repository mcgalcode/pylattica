from pylattica.core import Runner
from pylattica.discrete import PhaseSet
from pylattica.square_grid.grid_setup import DiscreteGridSetup
from pylattica.visualization import DiscreteSquareGridArtist2D, DiscreteSquareGridResultArtist
from pylattica.models.game_of_life import Life

import os

def test_step_artist():
    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases)
    artist = DiscreteSquareGridArtist2D()
    starting_state = setup.setup_noise(10, ["dead", "alive"])
    controller = Life()
    runner = Runner(parallel=True)
    result = runner.run(starting_state.state, controller, 10, structure=starting_state.structure, verbose=False)
    artist.get_img(result.last_step, cell_size=5)

def test_result_artist():
    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases)
    step_artist = DiscreteSquareGridArtist2D()
    starting_state = setup.setup_noise(10, ["dead", "alive"])
    controller = Life()
    runner = Runner(parallel=True)
    result = runner.run(starting_state.state, controller, 10, structure=starting_state.structure, verbose=False)
    step_artist.get_img(result.last_step, cell_size=5)
    result_artist = DiscreteSquareGridResultArtist(step_artist, result)
    result_artist.to_gif("out.gif", cell_size=5)
    os.remove("out.gif")