from pylattica.core import SynchronousRunner, BasicController
from pylattica.core.simulation_state import SimulationState
from pylattica.discrete import PhaseSet
from pylattica.structures.square_grid.grid_setup import DiscreteGridSetup
from pylattica.visualization import SquareGridArtist2D, SquareGridArtist3D, ResultArtist, DiscreteCellArtist
from pylattica.models.game_of_life import Life
from pylattica.discrete.state_constants import DISCRETE_OCCUPANCY

import os
import random

def test_step_artist():
    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases)
    simulation = setup.setup_noise(10, ["dead", "alive"])
    controller = Life(structure = simulation.structure)
    runner = SynchronousRunner(parallel=False)
    result = runner.run(simulation.state, controller, 10, verbose=False)
    cell_artist = DiscreteCellArtist.from_discrete_state(result.last_step)
    artist = SquareGridArtist2D(simulation.structure, cell_artist)
    artist.get_img(result.last_step, cell_size=5, label="test img")
    artist.save_img(result.last_step, "tmp.png")
    os.remove("tmp.png")

def test_result_artist():
    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases)
    simulation = setup.setup_noise(10, ["dead", "alive"])
    controller = Life(structure = simulation.structure)
    runner = SynchronousRunner(parallel=False)
    result = runner.run(simulation.state, controller, 10, verbose=False)
    cell_artist = DiscreteCellArtist.from_discrete_result(result)
    step_artist = SquareGridArtist2D(simulation.structure, cell_artist)
    step_artist.get_img(result.last_step, cell_size=5)
    result_artist = ResultArtist(step_artist, result)
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
    simulation = setup.setup_noise(3, ["dead", "alive"])
    runner = SynchronousRunner(parallel=False)
    result = runner.run(simulation.state, SimpleController(), 4, verbose=False)

    cell_artist = DiscreteCellArtist.from_discrete_state(result.last_step)
    artist = SquareGridArtist3D(simulation.structure, cell_artist)
    artist.get_img(result.last_step, cell_size=5)