from collections import deque
import math
from mimetypes import init
from random import random, randint
from typing import List

from .utils import printif

from .basic_controller import BasicController
from .simulation_result import SimulationResult
from .simulation_state import GENERAL, SITES, SimulationState

from tqdm import tqdm

import multiprocessing as mp

mp_globals = {}

class Runner():
    """Class for orchestrating the running of the simulation. An automaton simulation
    is run by applying the update rule (as implemented by a Controller) to sites
    in the SimulationState repeatedly. There are two primary modes of accomplishing
    this:

    Normal: In the normal configuration, one simulation step involves applying
        the update rule to every site in the SimulationState. The user should assume
        these applications follow no specific order. This is appropriate if the
        update rule only impacts the cell it is applied to, and no neighboring cells.

        If normal mode is used, the automaton can be parallelized, meaning that the
        update rule can be applied to many cells simultaneously. This can lead to
        dramatic improvements in speed for the simulation. Specify this using
        `parallel = True` during initialization. You can further specify the number
        of workers to use during parallel processing with the `workers` parameter.
        If left unspecified, one worker for each CPU will be created.

    Asynchronous: If a simulation is run asynchronously, one simulation step consists
        of a single site being chosen randomly and applying the update rule there. This
        is appropriate if the update rule impacts the cell it is focused on and it's
        neighbors.

        For instance, if the update rule "moves" an entity from one cell to
        a neighboring cell, it must be applied asynchronously because otherwise it's
        effects could interfere with the effects of neighboring applications. Specify
        that this mode should be used with the is_async initialization parameter.
    """

    def __init__(self,
            parallel: bool = False,
            workers: int = None,
            is_async: bool = False,
        ):
        """Instantiates a simulation Runner.

        Parameters
        ----------
        parallel : bool, optional
            Whether or not the runner should calculate updates in parallel, by default False
        workers : int, optional
            The number of workers to use. Only used if parallel is true, by default None
        is_async : bool, optional
            Whether or not the simulation should be run asynchrounously. If this is true
            then parallel and workers is ignored., by default False
        """

        """Initializes a simulation Runner.

        Args:
            parallel (Boolean): Whether or not this simulation should be run in parallel
            workers (int): The number of workers to use in the parallel version of the simulation
        """
        self.parallel = parallel
        self.workers = workers
        self.is_async = is_async

    def run(self, initial_state: SimulationState, controller: BasicController, num_steps: int, verbose = False) -> SimulationResult:
        """Run the simulation for the prescribed number of steps. Recall that one
        asynchronous simulation step involves one application of the update rule,
        and one normal simulation step applies the update rule to every site.

        Parameters
        ----------
        initial_state : SimulationState
            The starting state for the simulation.
        controller : BasicController
            The controller (a descendent of BasicController) which implements the update rule.
        num_steps : int
            The number of steps for which the simulation should run.
        verbose : bool, optional
            If True, debug information is printed during the run, by default False

        Returns
        -------
        BasicSimulationResult
            The result of the simulation.
        """
        printif(verbose, "Initializing run")
        printif(verbose, f'Running w/ sim. size {initial_state.size}')

        result = controller.instantiate_result(initial_state.copy())

        state = initial_state.copy()

        global mp_globals

        if self.is_async:
            site_queue = deque()
            site_queue.append(controller.get_random_site())

            for _ in tqdm(range(num_steps)):
                site_id = site_queue.popleft()

                controller_response = controller.get_state_update(site_id, state)
                next_sites = []
                # See if controller is specifying which sites to visit next
                if type(controller_response) is tuple:
                    state_updates, next_sites = controller_response
                else:
                    state_updates = controller_response

                state.batch_update(state_updates)
                site_queue.extend(next_sites)
                result.add_step(state_updates)

                if len(site_queue) == 0:
                    site = randint(0, initial_state.size)
                    site_queue.append(site)

        elif self.parallel:
            mp_globals['controller'] = controller
            mp_globals['initial_state'] = initial_state

            if self.workers is None:
                PROCESSES = mp.cpu_count()
            else:
                PROCESSES = self.workers

            printif(verbose, f'Running in parallel using {PROCESSES} workers')
            num_sites = initial_state.size
            chunk_size = math.ceil(num_sites / PROCESSES)
            print(f'Distributing {num_sites} update tasks to {PROCESSES} workers in chunks of {chunk_size}')
            with mp.get_context('fork').Pool(PROCESSES) as pool:
                updates = {}
                for i in tqdm(range(num_steps)):
                    updates = self._take_step_parallel(updates, pool, chunk_size = chunk_size)
                    result.add_step(updates)
                    printif(verbose, f'Finished step {i}')
        else:
            for _ in tqdm(range(num_steps)):
                updates = self._take_step(state, controller)
                state.batch_update(updates)
                result.add_step(updates)

        return result

    def _take_step_parallel(self, updates: dict, pool, chunk_size) -> SimulationState:
        """Given a SimulationState, advances the system state by one time increment
        and returns a new reaction step.

        Args:
            step (SimulationState):

        Returns:
            SimulationState:
        """
        params = []
        site_ids = mp_globals['initial_state'].site_ids()
        num_sites = len(site_ids)
        site_batches = [site_ids[i:i + chunk_size] for i in range(0, num_sites, chunk_size)]
        for batch in site_batches:
            params.append([batch, updates])
        results = pool.starmap(_step_batch_parallel, params)
        all_updates = {
            SITES: {},
            GENERAL: {}
        }
        for batch_update_res in results:
            all_updates[SITES].update(batch_update_res.get(SITES, {}))
            all_updates[GENERAL].update(batch_update_res.get(GENERAL, {}))

        return all_updates


    def _take_step(self, state: SimulationState, controller: BasicController) -> SimulationState:
        site_ids = state.site_ids()
        updates = _step_batch(site_ids, state, controller)

        return updates


def _step_batch_parallel(id_batch: List[int], last_updates: dict):
    """Here we are in a subprodcess

    Args:
        id_batch (List[int]): _description_
        previous_state (SimulationState): _description_

    Returns:
        _type_: _description_
    """
    state = mp_globals['initial_state']
    state.batch_update(last_updates)
    return _step_batch(
        id_batch,
        state,
        mp_globals['controller']
    )

def _step_batch(id_batch: List[int], previous_state: SimulationState, controller: BasicController):
    batch_updates = {
        SITES: {}
    }
    for site_id in id_batch:
        state_updates = controller.get_state_update(site_id, previous_state)
        batch_updates[SITES][site_id] = state_updates

    return batch_updates
