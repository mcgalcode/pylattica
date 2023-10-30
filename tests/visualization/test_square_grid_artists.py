from pylattica.core import Runner, BasicController
from pylattica.core.simulation_state import SimulationState
from pylattica.discrete import PhaseSet
from pylattica.structures.square_grid.grid_setup import DiscreteGridSetup
from pylattica.visualization import DiscreteSquareGridArtist2D, DiscreteSquareGridArtist3D, DiscreteSquareGridResultArtist
from pylattica.models.game_of_life import Life
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY

import os
import random

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

def test_step_artist_3D():

    class SimpleController(BasicController):

        def __init__(self):
            pass

        def get_state_update(self, site_id: int, prev_state: SimulationState):
            return { DISCRETE_OCCUPANCY: random.choice(["dead", "alive"])}


    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases, dim=3)
    artist = DiscreteSquareGridArtist3D()
    starting_state = setup.setup_noise(10, ["dead", "alive"])
    runner = Runner(parallel=True)
    result = runner.run(starting_state.state, SimpleController(), 4, structure=starting_state.structure, verbose=False)
    artist.get_img(result.last_step, cell_size=5)