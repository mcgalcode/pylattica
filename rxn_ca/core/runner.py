from random import randint

from rxn_ca.core.neighborhoods import Neighborhood, NeighborhoodView
from .basic_controller import BasicController
from .basic_simulation_result import BasicSimulationResult
from .basic_simulation_step import BasicSimulationStep

import numpy as np
from tqdm import tqdm

import multiprocessing as mp

mp_globals = {}

def printif(cond, statement):
    if cond:
        print(statement)

class Runner():
    """Class for orchestrating the running of the simulation. Provide this class a
    set of possible reactions and a BasicSimulationStep that represents the initial system state,
    and it will run a simulation for the prescribed number of steps.
    """

    def __init__(self, parallel = False, workers = None, is_async = False):
        """Initializes a simulation Runner.

        Args:
            parallel (Boolean): Whether or not this simulation should be run in parallel
            workers (int): The number of workers to use in the parallel version of the simulation
        """
        self.parallel = parallel
        self.workers = workers
        self.is_async = is_async

    def run(self, initial_step: BasicSimulationStep, controller: BasicController, num_steps: int, verbose = False) -> BasicSimulationResult:
        """Run the simulation for the prescribed number of steps.

        Args:
            num_steps (int): The number of steps for which the simulation should run.

        Returns:
            BasicSimulationResult:
        """
        printif(verbose, "Initializing run")
        result = controller.instantiate_result()
        printif(verbose, f'Running w/ sim. size {initial_step.size}')

        step = initial_step

        result.add_step(step)

        global mp_globals

        if self.is_async:
            for _ in tqdm(range(num_steps)):
                step = self._take_step_async(step, controller)
                result.add_step(step)
        else:
            if self.parallel:
                mp_globals['controller'] = controller

                if self.workers is None:
                    PROCESSES = mp.cpu_count()
                else:
                    PROCESSES = self.workers

                printif(verbose, f'Running in parallel using {PROCESSES} workers')

                with mp.get_context('fork').Pool(PROCESSES) as pool:
                    for i in tqdm(range(num_steps)):
                        step = self._take_step_parallel(step, pool)
                        result.add_step(step)
                        printif(verbose, f'Finished step {i}')
            else:
                for _ in tqdm(range(num_steps)):
                    step = self._take_step(step, controller)
                    result.add_step(step)

        return result

    def _take_step_parallel(self, step, pool) -> BasicSimulationStep:
        """Given a BasicSimulationStep, advances the system state by one time increment
        and returns a new reaction step.

        Args:
            step (BasicSimulationStep):

        Returns:
            BasicSimulationStep:
        """
        params = []

        for i in range(0, step.size):
            params.append([step, step.size, i])

        results = pool.starmap(step_row_parallel, params)

        new_state = np.array(list(map(lambda x: x[0], results)))
        step_metadata = list(map(lambda x: x[1], results))

        return BasicSimulationStep(new_state, step_metadata)


    def _take_step(self, step: BasicSimulationStep, controller: BasicController) -> BasicSimulationStep:
        results = []

        for i in range(0, step.size):
            results.append(step_row(step, step.size, i, controller))

        new_state = np.array(list(map(lambda x: x[0], results)))
        step_metadata = list(map(lambda x: x[1], results))

        return BasicSimulationStep(new_state, step_metadata)

    def _take_step_async(self, step: BasicSimulationStep, controller: BasicController) -> BasicSimulationStep:

        new_state = np.copy(step.state)

        rand_i = randint(0,step.size - 1)
        rand_j = randint(0,step.size - 1)
        nb_view = controller.neighborhood.get_in_step(step, [rand_i, rand_j])
        new_cell_state, step_metadata = controller.get_new_state(nb_view)

        new_state[rand_i, rand_j] = new_cell_state

        return BasicSimulationStep(new_state, step_metadata)

def step_row_parallel(step: BasicSimulationStep, state_size: int, row_num: int):
    return step_row(
        step,
        state_size,
        row_num,
        mp_globals['controller'],
    )

def step_row(step: BasicSimulationStep, state_size: int, i: int, controller: BasicController):
    new_state = np.zeros(state_size)
    cells_metadata = []
    for j in range(0, state_size):
        view: NeighborhoodView = controller.neighborhood.get_in_step(step, [i, j])
        new_cell_state, cell_update_metadata = controller.get_new_state(view)
        cells_metadata.append(cell_update_metadata)

        new_state[j] = new_cell_state
    return new_state, cells_metadata
