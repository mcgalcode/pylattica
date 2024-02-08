import math
import multiprocessing as mp
from typing import List

from tqdm import tqdm

from ..basic_controller import BasicController
from ..simulation_result import SimulationResult
from ..simulation_state import SimulationState
from ..utils import printif

from .base_runner import Runner
from .common import merge_updates

mp_globals = {}


class SynchronousRunner(Runner):
    """Class for orchestrating the running of the simulation. An automaton simulation
    is run by applying the update rule (as implemented by a Controller) to sites
    in the SimulationState repeatedly. This class implements the synchronous update
    strategy.

    In the normal configuration, one simulation step involves applying
    the update rule to every site in the SimulationState. The user should assume
    these applications follow no specific order. This is appropriate if the
    update rule only impacts the cell it is applied to, and no neighboring cells.

    If normal mode is used, the automaton can be parallelized, meaning that the
    update rule can be applied to many cells simultaneously. This can lead to
    dramatic improvements in speed for the simulation. Specify this using
    `parallel = True` during initialization. You can further specify the number
    of workers to use during parallel processing with the `workers` parameter.
    If left unspecified, one worker for each CPU will be created.
    """

    def __init__(self, parallel: bool = False, workers: int = None) -> None:
        self.parallel = parallel
        self.workers = workers

    def _run(
        self,
        initial_state: SimulationState,
        result: SimulationResult,
        live_state: SimulationState,
        controller: BasicController,
        num_steps: int,
        verbose: bool = False,
    ):
        if self.parallel:
            global mp_globals  # pylint: disable=global-variable-not-assigned
            mp_globals["controller"] = controller
            mp_globals["initial_state"] = initial_state

            if self.workers is None:
                PROCESSES = mp.cpu_count()
            else:
                PROCESSES = self.workers  # pragma: no cover

            printif(verbose, f"Running in parallel using {PROCESSES} workers")
            num_sites = initial_state.size
            chunk_size = math.ceil(num_sites / PROCESSES)
            printif(
                verbose,
                f"Distributing {num_sites} update tasks to {PROCESSES} workers in chunks of {chunk_size}",
            )
            with mp.get_context("fork").Pool(PROCESSES) as pool:
                updates = {}
                for _ in tqdm(range(num_steps)):
                    updates = self._take_step_parallel(
                        updates, pool, chunk_size=chunk_size
                    )
                    result.add_step(updates)
        else:
            printif(verbose, "Running in series.")
            for _ in tqdm(range(num_steps)):
                updates = self._take_step(live_state, controller)
                live_state.batch_update(updates)
                result.add_step(updates)

        result.set_output(live_state)
        return result

    def _take_step_parallel(self, updates: dict, pool, chunk_size) -> SimulationState:
        params = []
        site_ids = mp_globals["initial_state"].site_ids()
        num_sites = len(site_ids)
        site_batches = [
            site_ids[i : i + chunk_size] for i in range(0, num_sites, chunk_size)
        ]

        for batch in site_batches:
            params.append([batch, updates])

        results = pool.starmap(_step_batch_parallel, params)
        all_updates = None
        for batch_update_res in results:
            all_updates = merge_updates(all_updates, batch_update_res)

        return all_updates

    def _take_step(
        self, state: SimulationState, controller: BasicController
    ) -> SimulationState:
        site_ids = state.site_ids()
        updates = _step_batch(site_ids, state, controller)

        return updates


def _step_batch_parallel(id_batch: List[int], last_updates: dict):  # pragma: no cover
    state = mp_globals["initial_state"]
    state.batch_update(last_updates)
    return _step_batch(id_batch, state, mp_globals["controller"])


def _step_batch(
    id_batch: List[int], previous_state: SimulationState, controller: BasicController
):
    batch_updates = None
    for site_id in id_batch:
        site_updates = controller.get_state_update(site_id, previous_state)
        batch_updates = merge_updates(site_updates, batch_updates, site_id)

    return batch_updates
