from typing import Any, Callable, Dict, Optional, Union

from monty.serialization import dumpfn, loadfn
import datetime

from .frame_sampler import FrameSampler
from .observers import StateObserver
from .simulation_state import SimulationState

Observer = Union[StateObserver, Callable[[SimulationState, int], Any]]


class ObservedResult:
    """A lightweight accumulation of reduced observations over a simulation.

    Rather than retaining full simulation states or diffs, an ObservedResult
    stores the output of a reducer (a StateObserver or callable) applied to the
    live state at a configurable cadence. This is dramatically lighter than a
    full SimulationResult when only a derived view of the trajectory is needed
    (for example, per-phase amounts at each sampled step).

    An ObservedResult does not reconstruct states; get_step is not supported.

    During a run it is driven by a SimulationResult: the result fans out its
    live state to record() at each step, and finalize() captures the final
    step. Each ObservedResult keeps its own sampling cadence (independent from
    the SimulationResult's frame cadence), so observations can be sampled more
    or less frequently than stored frames.

    Attributes
    ----------
    observer : StateObserver or callable, optional
        The reducer applied to each sampled state. None on a deserialized
        result (the observations are already computed).
    compress_freq : int
        The effective observation interval (delegated to FrameSampler).
    num_frames : int, optional
        The target number of observations, if configured by frame count.
    """

    @classmethod
    def from_file(cls, fpath):
        return loadfn(fpath)

    @classmethod
    def from_dict(cls, res_dict):
        compress_freq = res_dict.get("compress_freq", 1)
        num_frames = res_dict.get("num_frames", None)
        # Reconstruct with the already-derived compress_freq (avoid re-deriving
        # and sidestep the mutually-exclusive validation). No observer is needed
        # since the observations have already been computed.
        res = cls(observer=None, compress_freq=compress_freq)
        res._sampler.num_frames = num_frames
        res._total_steps = res_dict.get("total_steps", 0)

        observations = res_dict.get("observations", {})
        res._observations = {int(step): obs for step, obs in observations.items()}
        return res

    def __init__(
        self,
        observer: Optional[Observer] = None,
        compress_freq: int = None,
        num_frames: int = None,
    ):
        """Initialize an ObservedResult.

        Parameters
        ----------
        observer : StateObserver or callable, optional
            The reducer applied to each sampled state, with signature
            ``(state, step_no) -> Any``. May be None for a result that only
            holds previously computed observations (e.g. after deserialization).
        compress_freq : int, optional
            Take an observation every compress_freq steps. Cannot be combined
            with num_frames. Defaults to every step.
        num_frames : int, optional
            Target total number of observations; the interval is derived once
            the step count is known via prepare_for_steps(). Cannot be combined
            with compress_freq.

        Raises
        ------
        ValueError
            If both compress_freq and num_frames are specified, or if either is
            not a positive integer.
        """
        self.observer = observer
        self._sampler = FrameSampler(compress_freq=compress_freq, num_frames=num_frames)
        self._observations: Dict[int, Any] = {}
        self._total_steps: int = 0

    @property
    def compress_freq(self) -> int:
        """The effective observation interval (delegated to FrameSampler)."""
        return self._sampler.compress_freq

    @property
    def num_frames(self) -> int:
        """The target observation count, if configured by frame count."""
        return self._sampler.num_frames

    @property
    def observations(self) -> Dict[int, Any]:
        """The recorded observations, keyed by step number."""
        return self._observations

    @property
    def observation_steps(self) -> list:
        """The sorted step numbers at which observations were recorded."""
        return sorted(self._observations.keys())

    def observation_at(self, step_no: int) -> Any:
        """Return the observation recorded at the given step.

        Raises
        ------
        KeyError
            If no observation was recorded at that step.
        """
        return self._observations[step_no]

    def _reduce(self, state: SimulationState, step_no: int) -> Any:
        if isinstance(self.observer, StateObserver):
            return self.observer.observe(state, step_no)
        return self.observer(state, step_no)

    def prepare_for_steps(self, num_steps: int) -> None:
        """Derive the observation interval now that the step count is known.

        A no-op unless configured with num_frames.

        Parameters
        ----------
        num_steps : int
            The total number of steps the simulation will run.
        """
        self._sampler.prepare_for_steps(num_steps)

    def record(self, state: SimulationState, step_no: int) -> None:
        """Record an observation for the given step if it falls on the cadence.

        Step 0 always records (the initial observation).

        Parameters
        ----------
        state : SimulationState
            The current (live) simulation state.
        step_no : int
            The step number associated with this state.
        """
        self._total_steps = step_no
        if self.observer is None:
            return
        if self._sampler.should_sample(step_no):
            self._observations[step_no] = self._reduce(state, step_no)

    def finalize(self, state: SimulationState, step_no: int) -> None:
        """Ensure the final step is recorded as an observation.

        The last step may not land on the sampling cadence, so it would
        otherwise be omitted. A no-op before any steps, or if the final step was
        already recorded.

        Parameters
        ----------
        state : SimulationState
            The final live simulation state.
        step_no : int
            The final step number.
        """
        self._total_steps = step_no
        if self.observer is None:
            return
        if step_no > 0 and step_no not in self._observations:
            self._observations[step_no] = self._reduce(state, step_no)

    def __len__(self) -> int:
        return len(self._observations)

    def as_dict(self):
        return {
            "observations": {
                str(step): obs for step, obs in self._observations.items()
            },
            "compress_freq": self.compress_freq,
            "num_frames": self.num_frames,
            "total_steps": self._total_steps,
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

    def to_file(self, fpath: str = None) -> str:
        """Serialize this result to the specified filepath."""
        if fpath is None:
            now = datetime.datetime.now()
            date_string = now.strftime("%m-%d-%Y-%H-%M")
            fpath = f"{date_string}.json"

        dumpfn(self, fpath)
        return fpath
