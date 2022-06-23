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

    def __init__(self, parallel = False, workers = None):
        """Initializes a simulation Runner.

        Args:
            parallel (Boolean): Whether or not this simulation should be run in parallel
            workers (int): The number of workers to use in the parallel version of the simulation
        """
        self.parallel = parallel
        self.workers = workers

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

        if self.parallel:
            mp_globals['controller'] = controller

            if self.workers is None:
                PROCESSES = mp.cpu_count()
            else:
                PROCESSES = self.workers

            printif(verbose, f'Running in parallel using {PROCESSES} workers')

            with mp.get_context('fork').Pool(PROCESSES) as pool:
                for i in tqdm(range(num_steps)):
                    step = self._take_step_parallel(step, initial_step.size, pool, controller)
                    result.add_step(step)
                    printif(verbose, f'Finished step {i}')
        else:
            for _ in tqdm(range(num_steps)):
                step = self._take_step(step, initial_step.size, controller)
                result.add_step(step)

        return result

    def _take_step_parallel(self, step, state_size, pool, controller: BasicController) -> BasicSimulationStep:
        """Given a BasicSimulationStep, advances the system state by one time increment
        and returns a new reaction step.

        Args:
            step (BasicSimulationStep):

        Returns:
            BasicSimulationStep:
        """
        params = []
        padded_state = controller.pad_step(step)

        for i in range(0, state_size):
            params.append([padded_state, state_size, i])

        results = pool.starmap(step_row_parallel, params)

        new_state = np.array(list(map(lambda x: x[0], results)))
        step_metadata = list(map(lambda x: x[1], results))

        return BasicSimulationStep(new_state, step_metadata)


    def _take_step(self, step: BasicSimulationStep, state_size: int, controller: BasicController) -> BasicSimulationStep:
        results = []

        padded_state = controller.pad_step(step)
        for i in range(0, state_size):
            results.append(step_row(padded_state, state_size, i, controller))

        new_state = np.array(list(map(lambda x: x[0], results)))
        step_metadata = list(map(lambda x: x[1], results))

        return BasicSimulationStep(new_state, step_metadata)

def step_row_parallel(padded_state, state_size, row_num):
    return step_row(
        padded_state,
        state_size,
        row_num,
        mp_globals['controller'],
    )

def step_row(padded_state, state_size: int, row_num: int, controller: BasicController):
    new_state = np.zeros(state_size)
    cells_metadata = []
    for j in range(0, state_size):
        new_cell_state, cell_update_metadata = controller.get_new_state(padded_state, row_num, j)
        cells_metadata.append(cell_update_metadata)

        new_state[j] = new_cell_state
    return new_state, cells_metadata
