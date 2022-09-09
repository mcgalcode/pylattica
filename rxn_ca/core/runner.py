from random import random
from typing import List


from .basic_controller import BasicController
from .basic_simulation_result import BasicSimulationResult
from .simulation_step import SimulationState

from tqdm import tqdm

import multiprocessing as mp

mp_globals = {}

def printif(cond, statement):
    if cond:
        print(statement)

class Runner():
    """Class for orchestrating the running of the simulation. Provide this class a
    set of possible reactions and a SimulationState that represents the initial system state,
    and it will run a simulation for the prescribed number of steps.
    """

    def __init__(self, parallel = False, workers = None, is_async = False, neighborhood_replace = False):
        """Initializes a simulation Runner.

        Args:
            parallel (Boolean): Whether or not this simulation should be run in parallel
            workers (int): The number of workers to use in the parallel version of the simulation
        """
        self.parallel = parallel
        self.workers = workers
        self.is_async = is_async
        self.neighborhood_replace = neighborhood_replace

    def run(self, initial_step: SimulationState, controller: BasicController, num_steps: int, verbose = False) -> BasicSimulationResult:
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
        elif self.neighborhood_replace:
            for _ in tqdm(range(num_steps)):
                step = self._take_replace_step(step, controller)
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

    def _take_step_parallel(self, prev_state: SimulationState, pool) -> SimulationState:
        """Given a SimulationState, advances the system state by one time increment
        and returns a new reaction step.

        Args:
            step (SimulationState):

        Returns:
            SimulationState:
        """
        params = []
        new_state = prev_state.copy()
        site_ids = prev_state.site_ids()
        chunk_size = 10
        site_batches = [site_ids[i:i + chunk_size] for i in range(0, len(site_ids), chunk_size)]
        for batch in site_batches:
            params.append([batch, prev_state])

        results = pool.starmap(step_batch_parallel, params)

        for batch_update_res in results:
            new_state.batch_update(batch_update_res[0])

        return new_state


    def _take_step(self, prev_state: SimulationState, controller: BasicController) -> SimulationState:
        new_state = []

        new_state = prev_state.copy()
        site_ids = new_state.site_ids()
        chunk_size = 4

        # using list comprehension
        site_batches = [site_ids[i:i + chunk_size] for i in range(0, len(site_ids), chunk_size)]

        for id_batch in site_batches:
            batch_updates = step_batch(id_batch, prev_state, controller)
            new_state.batch_update(batch_updates)

        return new_state

    def _take_step_async(self, prev_state: SimulationState, controller: BasicController) -> SimulationState:
        new_state = prev_state.copy()

        random_site_id = random.choice(prev_state.site_ids())
        state_updates = controller.get_state_update(random_site_id, prev_state)
        new_state.batch_update(state_updates)

        return new_state

def step_batch_parallel(id_batch: List[int], previous_state: SimulationState):
    return step_batch(
        id_batch,
        previous_state,
        mp_globals['controller']
    )

def step_batch(id_batch: List[int], previous_state: SimulationState, controller: BasicController):
    batch_updates = {}
    for site_id in id_batch:
        state_updates = controller.get_state_update(site_id, previous_state)
        batch_updates[site_id] = state_updates
    return batch_updates
