from collections import deque

from tqdm import tqdm

from ..basic_controller import BasicController
from ..simulation_result import SimulationResult
from ..simulation_state import SimulationState

from .base_runner import Runner
from .common import merge_updates


class AsynchronousRunner(Runner):
    """Class for orchestrating the running of the simulation. An automaton simulation
    is run by applying the update rule (as implemented by a Controller) to sites
    in the SimulationState repeatedly. This Runner implements the Asynchronous evolution
    strategy.

    If a simulation is run asynchronously, one simulation step consists
    of a single site being chosen randomly and applying the update rule there. This
    is appropriate if the update rule impacts the cell it is focused on and it's
    neighbors.

    For instance, if the update rule "moves" an entity from one cell to
    a neighboring cell, it must be applied asynchronously because otherwise it's
    effects could interfere with the effects of neighboring applications. Specify
    that this mode should be used with the is_async initialization parameter.
    """

    def _run(
        self,
        _: SimulationState,
        result: SimulationResult,
        live_state: SimulationState,
        controller: BasicController,
        num_steps: int,
        verbose: bool = False,
    ) -> SimulationResult:
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

        site_queue = deque()

        def _add_sites_to_queue():
            next_site = controller.get_random_site(live_state)
            if isinstance(next_site, int):
                site_queue.append(next_site)
            elif isinstance(next_site, list):
                site_queue.extend(next_site)

        _add_sites_to_queue()

        if len(site_queue) == 0:
            raise RuntimeError("Controller provided no sites to update, ABORTING")

        for _ in tqdm(range(num_steps), disable=(not verbose)):
            site_id = site_queue.popleft()

            controller_response = controller.get_state_update(site_id, live_state)
            next_sites = []

            # See if controller is specifying which sites to visit next
            if isinstance(controller_response, tuple):
                state_updates, next_sites = controller_response
            else:
                state_updates = controller_response

            state_updates = merge_updates(state_updates, site_id=site_id)
            live_state.batch_update(state_updates)
            site_queue.extend(next_sites)

            result.add_step(state_updates)

            if len(site_queue) == 0:
                _add_sites_to_queue()

            if len(site_queue) == 0:
                break

        result.set_output(live_state)
        return result
