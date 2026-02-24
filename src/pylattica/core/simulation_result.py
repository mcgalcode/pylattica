import tqdm

from typing import Dict, List

from monty.serialization import dumpfn, loadfn
import datetime
from .simulation_state import SimulationState
from .constants import GENERAL, SITES

import copy


class SimulationResult:
    """A class that stores the result of running a simulation.

    Attributes
    ----------
    initial_state : SimulationState
        The state with which the simulation started.
    max_history : int, optional
        Maximum number of diffs to keep in memory. When exceeded, older diffs
        are dropped and a checkpoint is created. Set to None for unlimited
        history (default, but may cause memory issues for long simulations).
    """

    @classmethod
    def from_file(cls, fpath):
        return loadfn(fpath)

    @classmethod
    def from_dict(cls, res_dict):
        diffs = res_dict["diffs"]
        compress_freq = res_dict.get("compress_freq", 1)
        max_history = res_dict.get("max_history", None)
        res = cls(
            SimulationState.from_dict(res_dict["initial_state"]),
            compress_freq=compress_freq,
            max_history=max_history,
        )
        # Restore checkpoint if present
        if "checkpoint_state" in res_dict and res_dict["checkpoint_state"] is not None:
            res._checkpoint_state = SimulationState.from_dict(res_dict["checkpoint_state"])
            res._checkpoint_step = res_dict.get("checkpoint_step", 0)

        for diff in diffs:
            if SITES in diff:
                diff[SITES] = {int(k): v for k, v in diff[SITES].items()}
            if GENERAL not in diff and SITES not in diff:
                diff = {int(k): v for k, v in diff.items()}
            res._diffs.append(diff)  # Bypass add_step to avoid re-checkpointing

        # Restore total_steps from serialized data, or compute from diffs + checkpoint
        res._total_steps = res_dict.get("total_steps", res._checkpoint_step + len(diffs))

        return res

    def __init__(self, starting_state: SimulationState, compress_freq: int = 1, max_history: int = None):
        """Initializes a SimulationResult with the specified starting_state.

        Parameters
        ----------
        starting_state : SimulationState
            The state with which the simulation started.
        compress_freq : int, optional
            Compression frequency for sampling, by default 1.
        max_history : int, optional
            Maximum number of diffs to keep in memory. When exceeded, a
            checkpoint is created and old diffs are dropped. This prevents
            unbounded memory growth during long simulations. Set to None
            (default) for unlimited history. Recommended: 1000-10000 for
            long simulations.
        """
        self.initial_state = starting_state
        self.compress_freq = compress_freq
        self.max_history = max_history
        self._diffs: list[dict] = []
        self._stored_states = {}
        # Checkpoint support for bounded history
        self._checkpoint_state: SimulationState = None
        self._checkpoint_step: int = 0
        self._total_steps: int = 0

    def add_step(self, updates: Dict[int, Dict]) -> None:
        """Takes a set of updates as a dictionary mapping site IDs
        to the new values for various state parameters. For instance, if at the
        new step, my_state_attribute at site 23 changed to 12, updates would look
        like this:

        {
            23: {
                "my_state_attribute": 12
            }
        }

        Parameters
        ----------
        updates : dict
            The changes associated with a new simulation step.
        """
        self._diffs.append(updates)
        self._total_steps += 1

        # Check if we need to create a checkpoint and drop old diffs
        if self.max_history is not None and len(self._diffs) > self.max_history:
            self._create_checkpoint()

    def _create_checkpoint(self) -> None:
        """Creates a checkpoint by computing the current state and dropping old diffs.

        This is called automatically when max_history is exceeded. It computes
        the state at the midpoint of the current diffs, stores it as a checkpoint,
        and drops all diffs before that point.
        """
        # Compute checkpoint at half the current diffs (keeps half the history)
        checkpoint_offset = len(self._diffs) // 2

        # Compute the state at the checkpoint
        if self._checkpoint_state is not None:
            state = self._checkpoint_state.copy()
        else:
            state = self.initial_state.copy()

        for i in range(checkpoint_offset):
            state.batch_update(self._diffs[i])

        # Update checkpoint
        self._checkpoint_state = state
        self._checkpoint_step += checkpoint_offset

        # Drop old diffs
        self._diffs = self._diffs[checkpoint_offset:]

        # Clear stored states cache (indices are now invalid)
        self._stored_states.clear()

    @property
    def earliest_available_step(self) -> int:
        """Returns the earliest step number that can be reconstructed.

        When max_history is set and checkpoints have been created, early
        steps are no longer available.
        """
        return self._checkpoint_step

    @property
    def original_length(self) -> int:
        return int(len(self) * self.compress_freq)

    def __len__(self) -> int:
        # Total steps = checkpoint step + remaining diffs + 1 (for initial state)
        return self._total_steps + 1

    def steps(self) -> List[SimulationState]:
        """Yields all available steps from this simulation.

        Note: When max_history is set, only steps from the checkpoint onward
        are available. Use earliest_available_step to check.

        Yields
        ------
        SimulationState
            Each step's state (as a copy to avoid mutation issues).
        """
        # Start from checkpoint or initial state
        if self._checkpoint_state is not None:
            live_state = self._checkpoint_state.copy()
        else:
            live_state = self.initial_state.copy()

        yield live_state.copy()  # Yield a copy to avoid mutation issues
        for diff in self._diffs:
            live_state.batch_update(diff)
            yield live_state.copy()

    @property
    def last_step(self) -> SimulationState:
        """The last step of the simulation.

        Returns
        -------
        SimulationState
            The last step of the simulation
        """
        return self.get_step(len(self) - 1)

    @property
    def first_step(self):
        return self.get_step(0)

    def set_output(self, step: SimulationState):
        self.output = step

    def load_steps(self, interval=1):
        """Pre-loads steps into memory at the specified interval for faster access.

        Parameters
        ----------
        interval : int, optional
            Store every Nth step in memory, by default 1.
        """
        # Clear old cache first
        self._stored_states.clear()

        # Start from checkpoint or initial state
        if self._checkpoint_state is not None:
            live_state = self._checkpoint_state.copy()
            start_step = self._checkpoint_step
        else:
            live_state = self.initial_state.copy()
            start_step = 0

        self._stored_states[start_step] = live_state.copy()

        for ud_idx in tqdm.tqdm(
            range(0, len(self._diffs)), desc="Constructing result from diffs"
        ):
            step_no = start_step + ud_idx + 1
            live_state.batch_update(self._diffs[ud_idx])
            if step_no % interval == 0 and self._stored_states.get(step_no) is None:
                stored_state = live_state.copy()
                self._stored_states[step_no] = stored_state

    def get_step(self, step_no) -> SimulationState:
        """Retrieves the step at the provided number.

        Parameters
        ----------
        step_no : int
            The number of the step to return.

        Returns
        -------
        SimulationState
            The simulation state at the requested step.

        Raises
        ------
        ValueError
            If step_no is before the earliest available step (when using max_history).
        """
        if step_no < self._checkpoint_step:
            raise ValueError(
                f"Cannot retrieve step {step_no}. Earliest available step is "
                f"{self._checkpoint_step} (earlier steps were dropped due to max_history={self.max_history})."
            )

        stored = self._stored_states.get(step_no)
        if stored is not None:
            return stored

        # Start from checkpoint (or initial state if no checkpoint)
        if self._checkpoint_state is not None:
            state = self._checkpoint_state.copy()
            start_idx = 0
        else:
            state = self.initial_state.copy()
            start_idx = 0

        # Apply diffs from checkpoint to requested step
        diffs_to_apply = step_no - self._checkpoint_step
        for ud_idx in range(start_idx, diffs_to_apply):
            state.batch_update(self._diffs[ud_idx])

        return state

    def as_dict(self):
        result = {
            "initial_state": self.initial_state.as_dict(),
            "diffs": self._diffs,
            "compress_freq": self.compress_freq,
            "max_history": self.max_history,
            "total_steps": self._total_steps,
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }
        # Include checkpoint if present
        if self._checkpoint_state is not None:
            result["checkpoint_state"] = self._checkpoint_state.as_dict()
            result["checkpoint_step"] = self._checkpoint_step
        else:
            result["checkpoint_state"] = None
            result["checkpoint_step"] = 0
        return result

    def to_file(self, fpath: str = None) -> None:
        """Serializes this result to the specified filepath.

        Parameters
        ----------
        fpath : str
            The filepath at which to save the serialized simulation result.
        """
        if fpath is None:
            now = datetime.datetime.now()
            date_string = now.strftime("%m-%d-%Y-%H-%M")
            fpath = f"{date_string}.json"

        dumpfn(self, fpath)
        return fpath


def compress_result(result: SimulationResult, num_steps: int):
    """Compress a simulation result by sampling fewer steps.

    Parameters
    ----------
    result : SimulationResult
        The result to compress.
    num_steps : int
        Target number of steps in the compressed result.

    Returns
    -------
    SimulationResult
        A new result with fewer steps.
    """
    # Use earliest available step as the starting point
    i_state = result.get_step(result.earliest_available_step)
    available_steps = len(result) - result.earliest_available_step
    if num_steps >= available_steps:
        raise ValueError(
            f"Cannot compress SimulationResult with {available_steps} available steps to {num_steps} steps."
        )

    exact_sample_freq = available_steps / num_steps
    total_compress_freq = exact_sample_freq * result.compress_freq
    compressed_result = SimulationResult(i_state, compress_freq=total_compress_freq)

    live_state = SimulationState(copy.deepcopy(i_state._state))
    next_sample_step = exact_sample_freq
    for i, diff in enumerate(result._diffs):
        curr_step = i + 1
        live_state.batch_update(diff)
        if curr_step > next_sample_step:
            compressed_result.add_step(live_state.as_state_update())
            next_sample_step += exact_sample_freq
    return compressed_result
