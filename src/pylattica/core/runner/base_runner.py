from ..basic_controller import BasicController
from ..simulation_result import SimulationResult
from ..simulation_state import SimulationState


class Runner:
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

    def run(
        self,
        initial_state: SimulationState,
        controller: BasicController,
        num_steps: int,
        verbose=False,
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

        result = controller.instantiate_result(initial_state.copy())
        controller.pre_run(initial_state)
        live_state = initial_state.copy()

        self._run(initial_state, result, live_state, controller, num_steps, verbose)

        result.set_output(live_state)
        return result
