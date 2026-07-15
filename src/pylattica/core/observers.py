from abc import ABC, abstractmethod
from typing import Any

from .simulation_state import SimulationState


class StateObserver(ABC):
    """Reduces a SimulationState to a small, serializable observation.

    Observers are the mechanism for recording a lightweight, derived view of
    the simulation (for example, per-phase amounts) at a configurable cadence,
    instead of retaining full simulation states or diffs. An ObservedResult
    drives a StateObserver during a run and stores its outputs.

    A plain callable with the signature ``(state, step_no) -> Any`` may be used
    anywhere a StateObserver is accepted.
    """

    @abstractmethod
    def observe(self, state: SimulationState, step_no: int) -> Any:
        """Compute the observation for the given state.

        Parameters
        ----------
        state : SimulationState
            The current (live) simulation state.
        step_no : int
            The step number at which this observation is being taken.

        Returns
        -------
        Any
            A small, serializable value summarizing the state.
        """
