import math
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
    live_compress : bool, optional
        If True, store full state snapshots at compress_freq intervals during
        simulation instead of diffs. This avoids the expensive O(n) reconstruction
        in load_steps() at the cost of more memory per frame. When enabled,
        load_steps() becomes a no-op since frames are already stored. Specifying
        either compress_freq or num_frames implies live_compress=True.
    num_frames : int, optional
        Target total number of frames to store, as an alternative to specifying
        compress_freq directly. The sampling frequency is derived from the planned
        number of steps once it is known (see prepare_for_steps). The initial and
        final frames are always included, so the stored count is approximately
        num_frames. Cannot be combined with compress_freq.
    """

    @classmethod
    def from_file(cls, fpath):
        return loadfn(fpath)

    @classmethod
    def from_dict(cls, res_dict):
        diffs = res_dict.get("diffs", [])
        compress_freq = res_dict.get("compress_freq", 1)
        max_history = res_dict.get("max_history", None)
        # Read the (derived) live_compress flag for back-compat with older
        # serialized results that predate compress_freq/num_frames implying it.
        live_compress = res_dict.get("live_compress", False)
        num_frames = res_dict.get("num_frames", None)
        # Reconstruct using the already-derived compress_freq so we don't
        # re-derive it (and to sidestep the mutually-exclusive validation).
        res = cls(
            SimulationState.from_dict(res_dict["initial_state"]),
            compress_freq=compress_freq if live_compress else None,
            max_history=max_history,
        )
        # Preserve the original num_frames target for round-tripping.
        res.num_frames = num_frames
        # Restore checkpoint if present
        if "checkpoint_state" in res_dict and res_dict["checkpoint_state"] is not None:
            res._checkpoint_state = SimulationState.from_dict(
                res_dict["checkpoint_state"]
            )
            res._checkpoint_step = res_dict.get("checkpoint_step", 0)

        # Restore frames if present (for live_compress mode)
        if "frames" in res_dict and res_dict["frames"]:
            for step_str, state_dict in res_dict["frames"].items():
                res._frames[int(step_str)] = SimulationState.from_dict(state_dict)

        for diff in diffs:
            if SITES in diff:
                diff[SITES] = {int(k): v for k, v in diff[SITES].items()}
            if GENERAL not in diff and SITES not in diff:
                diff = {int(k): v for k, v in diff.items()}
            res._diffs.append(diff)  # Bypass add_step to avoid re-checkpointing

        # Restore total_steps from serialized data, or compute from diffs + checkpoint
        res._total_steps = res_dict.get(
            "total_steps", res._checkpoint_step + len(diffs)
        )

        # Reconstruct live_state to reflect the final state
        if res._frames:
            # In live_compress mode, use the last frame
            last_step = max(res._frames.keys())
            res._live_state = res._frames[last_step].copy()
        elif res._diffs:
            # Replay all diffs to get final state
            if res._checkpoint_state is not None:
                res._live_state = res._checkpoint_state.copy()
            for diff in res._diffs:
                res._live_state.batch_update(diff)

        return res

    def __init__(
        self,
        starting_state: SimulationState,
        compress_freq: int = None,
        max_history: int = None,
        num_frames: int = None,
    ):
        """Initializes a SimulationResult with the specified starting_state.

        By default the result stores per-step diffs and reconstructs states on
        demand. Specifying either compress_freq or num_frames switches it into
        "live compression" mode, where full state snapshots (frames) are stored
        during the run instead of diffs.

        Parameters
        ----------
        starting_state : SimulationState
            The state with which the simulation started.
        compress_freq : int, optional
            Compression frequency for sampling. When set, a full state snapshot
            is stored every compress_freq steps. Cannot be combined with
            num_frames. Defaults to None (diff mode).
        max_history : int, optional
            Maximum number of diffs to keep in memory. When exceeded, a
            checkpoint is created and old diffs are dropped. This prevents
            unbounded memory growth during long simulations. Set to None
            (default) for unlimited history. Recommended: 1000-10000 for
            long simulations. Ignored in live compression mode.
        num_frames : int, optional
            Target total number of frames to store, as an alternative to
            compress_freq. The sampling frequency is derived from the planned
            number of steps via prepare_for_steps(). The initial and final
            frames are always included. Cannot be combined with compress_freq.

        Raises
        ------
        ValueError
            If both compress_freq and num_frames are specified, or if either is
            not a positive integer.
        """
        self.initial_state = starting_state
        self.max_history = max_history
        self._diffs: list[dict] = []
        self._stored_states = {}
        self._frames: Dict[int, SimulationState] = {}  # For live_compress mode
        # Checkpoint support for bounded history
        self._checkpoint_state: SimulationState = None
        self._checkpoint_step: int = 0
        self._total_steps: int = 0

        # Live state that gets updated with each step
        self._live_state: SimulationState = starting_state.copy()

        # Set up the frame-sampling strategy (also stores frame 0 if enabled).
        self.configure_compression(compress_freq=compress_freq, num_frames=num_frames)

    def configure_compression(
        self, compress_freq: int = None, num_frames: int = None
    ) -> None:
        """Configures the frame-sampling strategy.

        Called from __init__, and also by the runner after instantiation once
        run-time options are known. Must be called before any steps have been
        recorded. Passing neither argument leaves the result in diff mode.

        Parameters
        ----------
        compress_freq : int, optional
            Store a full state snapshot every compress_freq steps. Cannot be
            combined with num_frames.
        num_frames : int, optional
            Target total number of frames; the interval is derived later via
            prepare_for_steps(). Cannot be combined with compress_freq.

        Raises
        ------
        ValueError
            If both compress_freq and num_frames are specified, or if either is
            not a positive integer.
        RuntimeError
            If called after steps have already been recorded.
        """
        if self._total_steps > 0 or self._diffs:
            raise RuntimeError(
                "configure_compression must be called before recording steps."
            )
        if compress_freq is not None and num_frames is not None:
            raise ValueError(
                "Specify either compress_freq or num_frames, not both."
            )
        if compress_freq is not None and compress_freq < 1:
            raise ValueError(f"compress_freq must be >= 1, got {compress_freq}.")
        if num_frames is not None and num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}.")

        # Specifying a sampling strategy (frequency or frame count) enables
        # live compression (storing frames instead of diffs).
        self.live_compress = compress_freq is not None or num_frames is not None
        self.num_frames = num_frames
        # Effective sampling interval. When num_frames is set this is derived
        # later (prepare_for_steps); until then default to every step.
        self.compress_freq = compress_freq if compress_freq is not None else 1

        # Store the initial state as frame 0 when live compression is enabled.
        self._frames = {}
        if self.live_compress:
            self._frames[0] = self._live_state.copy()

    def get_diffs(self) -> list[dict]:
        """Returns the list of diffs.

        Returns
        -------
        list[dict]
            The list of state diffs.
        """
        return self._diffs

    @property
    def live_state(self) -> SimulationState:
        """The current live state of the simulation.

        This state is updated with each call to add_step(). Use this to access
        the current simulation state during a run.
        """
        return self._live_state

    def prepare_for_steps(self, num_steps: int) -> None:
        """Finalizes the frame sampling strategy now that the total number of
        steps is known.

        When the result was configured with num_frames, the sampling frequency
        cannot be determined at construction time (the total step count is only
        known by the runner). This method derives compress_freq so that roughly
        num_frames frames are stored over the course of the run. It is a no-op
        for diff mode or when compress_freq was specified directly.

        Parameters
        ----------
        num_steps : int
            The total number of steps the simulation will run.
        """
        if self.num_frames is None or num_steps is None:
            return

        # Derive the interval so that ~num_frames snapshots are stored. The
        # initial (and final, via finalize) frames are always included.
        self.compress_freq = max(1, math.ceil(num_steps / self.num_frames))

    def finalize(self) -> None:
        """Ensures the final state is stored as a frame.

        In live compression mode the last step may not land on a compress_freq
        boundary, so it would otherwise be omitted. This stores the current
        live state as the frame for the final step. It is a no-op in diff mode
        or before any steps have been taken.
        """
        if self.live_compress and self._total_steps > 0:
            self._frames[self._total_steps] = self._live_state.copy()

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

        This method:
        1. Applies the updates to the internal live_state
        2. Increments the step counter
        3. In live_compress mode: stores frames at compress_freq intervals
        4. In normal mode: stores the diff for later reconstruction

        Parameters
        ----------
        updates : dict
            The changes associated with a new simulation step.
        """
        # Update the live state
        self._live_state.batch_update(updates)
        self._total_steps += 1

        # In live_compress mode, store frames at intervals instead of diffs
        if self.live_compress:
            if self._total_steps % self.compress_freq == 0:
                self._frames[self._total_steps] = self._live_state.copy()
            return

        # Normal mode: store diffs
        self._diffs.append(updates)

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
        are available. When live_compress is set, only frames at compress_freq
        intervals are available. Use earliest_available_step to check.

        Yields
        ------
        SimulationState
            Each step's state (as a copy to avoid mutation issues).
        """
        # If frames exist (live_compress mode), yield them in order
        if self._frames:
            for step_no in sorted(self._frames.keys()):
                yield self._frames[step_no].copy()
            return

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

    @property
    def output(self) -> SimulationState:
        """The final output state of the simulation (alias for live_state)."""
        return self._live_state

    def load_steps(self, interval=1):
        """Pre-loads steps into memory at the specified interval for faster access.

        When live_compress is enabled, this is a no-op since frames are already
        stored during simulation. If a different interval is requested than what
        was used during simulation (compress_freq), an error is raised.

        Parameters
        ----------
        interval : int, optional
            Store every Nth step in memory, by default 1.

        Raises
        ------
        ValueError
            If live_compress was used and requested interval doesn't match compress_freq.
        """
        # If frames already exist (live_compress mode), no reconstruction needed
        if self._frames:
            if interval != self.compress_freq:
                raise ValueError(
                    f"Cannot load steps with interval={interval}. This result was "
                    f"created with live_compress=True and compress_freq={self.compress_freq}. "
                    f"Only interval={self.compress_freq} is available."
                )
            return

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
            If step_no is before the earliest available step (when using max_history),
            or if step_no is not available in live_compress mode.
        """
        # Check frames first (live_compress mode)
        if self._frames:
            if step_no in self._frames:
                return self._frames[step_no]
            # In live_compress mode, only frames at compress_freq intervals exist
            raise ValueError(
                f"Cannot retrieve step {step_no}. This result was created with "
                f"live_compress=True and compress_freq={self.compress_freq}. "
                f"Available steps: {sorted(self._frames.keys())}"
            )

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
            "live_compress": self.live_compress,
            "num_frames": self.num_frames,
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

        # Include frames if present (live_compress mode)
        if self._frames:
            result["frames"] = {
                str(step): state.as_dict() for step, state in self._frames.items()
            }
        else:
            result["frames"] = {}

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

    live_state = SimulationState(copy.deepcopy(i_state.get_state()))
    next_sample_step = exact_sample_freq
    for i, diff in enumerate(result.get_diffs()):
        curr_step = i + 1
        live_state.batch_update(diff)
        if curr_step > next_sample_step:
            compressed_result.add_step(live_state.as_state_update())
            next_sample_step += exact_sample_freq
    return compressed_result
