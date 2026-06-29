from typing import List, Optional

from ..basic_controller import BasicController
from ..observed_result import ObservedResult, Observer
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
        compress_freq: int = None,
        num_frames: int = None,
        retain_history: bool = True,
        observers: Optional[List[Observer]] = None,
        observe_freq: int = None,
        observe_num_frames: int = None,
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
        compress_freq : int, optional
            If set, the result stores a full state snapshot every compress_freq
            steps instead of per-step diffs. Cannot be combined with num_frames.
        num_frames : int, optional
            If set, the result stores roughly this many full state snapshots over
            the course of the run. Cannot be combined with compress_freq.
        retain_history : bool, optional
            If False, the result retains only the current (live) state -- no
            diffs and no frames. This is the minimum-memory mode, intended for
            use alongside observers. Cannot be combined with compress_freq or
            num_frames. Defaults to True.
        observers : list, optional
            Reducers to accumulate a lightweight, derived view of the trajectory
            alongside (or instead of) the full result. Each entry may be a bare
            StateObserver/callable -- which is wrapped into an ObservedResult
            using the shared observe_freq/observe_num_frames cadence -- or an
            already-constructed ObservedResult, which is attached as-is (allowing
            per-observer cadence). The filled accumulators are returned on
            result.observed_results, in the same order.
        observe_freq : int, optional
            The sampling cadence applied to bare reducers in observers. Cannot be
            combined with observe_num_frames. Ignored for entries that are
            already ObservedResults.
        observe_num_frames : int, optional
            Target observation count applied to bare reducers in observers. Cannot
            be combined with observe_freq. Ignored for entries that are already
            ObservedResults.

        Returns
        -------
        BasicSimulationResult
            The result of the simulation.
        """

        result = controller.instantiate_result(initial_state.copy())
        # Apply minimum-memory mode before any steps are recorded. Validation of
        # the (retain_history vs compression) combination lives in the result.
        result.set_retain_history(retain_history)
        # Configure live compression after instantiation when run-time options
        # are provided. This keeps storage policy out of the controller while
        # leaving instantiate_result overrides untouched.
        if compress_freq is not None or num_frames is not None:
            result.configure_compression(
                compress_freq=compress_freq, num_frames=num_frames
            )

        # Attach observers. Bare reducers are wrapped into ObservedResults using
        # the shared observe cadence; pre-built ObservedResults pass through so
        # callers can control per-observer cadence.
        for observer in observers or []:
            if isinstance(observer, ObservedResult):
                observed_result = observer
            else:
                observed_result = ObservedResult(
                    observer=observer,
                    compress_freq=observe_freq,
                    num_frames=observe_num_frames,
                )
            result.add_observed_result(observed_result)

        # Let the result derive its frame-sampling strategy now that the total
        # step count is known (used when configured with num_frames). This also
        # fans out to each attached observer.
        result.prepare_for_steps(num_steps)
        controller.pre_run(initial_state)

        self._run(initial_state, result, controller, num_steps, verbose)

        # Ensure the final state is captured (e.g. as the last frame in live
        # compression mode) and observers record the final step.
        result.finalize()

        return result
