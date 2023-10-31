from pylattica.core import SynchronousRunner, BasicController
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
    simulation = setup.setup_noise(10, ["dead", "alive"])
    controller = Life(structure = simulation.structure)
    runner = SynchronousRunner(parallel=True)
    result = runner.run(simulation.state, controller, 10, verbose=False)
    artist.get_img(result.last_step, cell_size=5)

def test_result_artist():
    phases = PhaseSet(["dead", "alive"])
    setup = DiscreteGridSetup(phases)
    step_artist = DiscreteSquareGridArtist2D()
    simulation = setup.setup_noise(10, ["dead", "alive"])
    controller = Life(structure = simulation.structure)
    runner = SynchronousRunner(parallel=True)
    result = runner.run(simulation.state, controller, 10, verbose=False)
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
    simulation = setup.setup_noise(10, ["dead", "alive"])
    runner = SynchronousRunner(parallel=True)
    result = runner.run(simulation.state, SimpleController(), 4, verbose=False)
    artist.get_img(result.last_step, cell_size=5)