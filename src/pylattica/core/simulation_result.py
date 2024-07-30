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
    """

    @classmethod
    def from_file(cls, fpath):
        return loadfn(fpath)

    @classmethod
    def from_dict(cls, res_dict):
        diffs = res_dict["diffs"]
        compress_freq = res_dict.get("compress_freq", 1)
        res = cls(
            SimulationState.from_dict(res_dict["initial_state"]),
            compress_freq=compress_freq,
        )
        for diff in diffs:
            if SITES in diff:
                diff[SITES] = {int(k): v for k, v in diff[SITES].items()}
            if GENERAL not in diff and SITES not in diff:
                diff = {int(k): v for k, v in diff.items()}
            res.add_step(diff)

        return res

    def __init__(self, starting_state: SimulationState, compress_freq: int = 1):
        """Initializes a SimulationResult with the specified starting_state.

        Parameters
        ----------
        starting_state : SimulationState
            The state with which the simulation started.
        """
        self.initial_state = starting_state
        self.compress_freq = compress_freq
        self._diffs: list[dict] = []
        self._stored_states = {}

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

    @property
    def original_length(self) -> int:
        return int(len(self) * self.compress_freq)

    def __len__(self) -> int:
        return len(self._diffs) + 1

    def steps(self) -> List[SimulationState]:
        """Returns a list of all the steps from this simulation.

        Returns
        -------
        List[SimulationState]
            The list of steps
        """
        live_state = self.initial_state.copy()
        for diff in self._diffs:
            yield live_state
            live_state.batch_update(diff)

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
        live_state = self.initial_state.copy()
        self._stored_states[0] = self.initial_state.copy()
        for ud_idx in tqdm.tqdm(
            range(0, len(self._diffs)), desc="Constructing result from diffs"
        ):
            step_no = ud_idx + 1
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
        """

        # if step_no % self.compress_freq != 0:
        #     raise ValueError(f"Cannot retrieve step no {step_no} because this result has been compressed with sampling frequency {self.compress_freq}")

        stored = self._stored_states.get(step_no)
        if stored is not None:
            return stored
        else:
            state = self.initial_state.copy()
            for ud_idx in range(0, step_no):
                state.batch_update(self._diffs[ud_idx])
            return state

    def as_dict(self):
        return {
            "initial_state": self.initial_state.as_dict(),
            "diffs": self._diffs,
            "compress_freq": self.compress_freq,
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

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
    i_state = result.first_step
    # total steps is the actual number of diffs stored, not the number of original simulation steps taken
    total_steps = len(result)
    if num_steps >= total_steps:
        raise ValueError(
            f"Cannot upsample SimulationResult of length {total_steps} to size {num_steps}."
        )

    exact_sample_freq = total_steps / (num_steps)
    # print(total_steps, current_sample_freq)
    total_compress_freq = exact_sample_freq * result.compress_freq
    compressed_result = SimulationResult(i_state, compress_freq=total_compress_freq)

    live_state = SimulationState(copy.deepcopy(i_state._state))
    added = 0
    next_sample_step = exact_sample_freq
    for i, diff in enumerate(result._diffs):
        curr_step = i + 1
        live_state.batch_update(diff)
        # if curr_step % current_sample_freq == 0:
        if curr_step > next_sample_step:
            # print(curr_step)
            added += 1
            compressed_result.add_step(live_state.as_state_update())
            next_sample_step += exact_sample_freq
    return compressed_result
