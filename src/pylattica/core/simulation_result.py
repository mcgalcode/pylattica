from typing import Dict, List

from monty.serialization import dumpfn, loadfn

from .simulation_state import SimulationState


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
        steps = res_dict["steps"]
        res = cls({})
        for step in steps:
            res.add_step(step)
        return res

    def __init__(self, starting_state: SimulationState):
        """Initializes a SimulationResult with the specified starting_state.

        Parameters
        ----------
        starting_state : SimulationState
            The state with which the simulation started.
        """
        self.initial_state = starting_state
        self._steps: list[SimulationState] = [starting_state]
        self._diffs: list[dict] = []

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
        new_step = self._steps[-1].copy()
        new_step.batch_update(updates)
        self._steps.append(new_step)
        self._diffs.append(updates)

    def __len__(self) -> int:
        return len(self._diffs) + 1

    @property
    def steps(self) -> List[SimulationState]:
        """Returns a list of all the steps from this simulation.

        Returns
        -------
        List[SimulationState]
            The list of steps
        """
        return self._steps

    @property
    def last_step(self) -> SimulationState:
        """The last step of the simulation.

        Returns
        -------
        SimulationState
            The last step of the simulation
        """
        return self.get_step(len(self))

    @property
    def first_step(self):
        return self.get_step(1)

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
        return self._steps[step_no - 1]

    def as_dict(self):
        return {
            "steps": [s.as_dict() for s in self._steps],
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

    def to_file(self, fpath: str) -> None:
        """Serializes this result to the specified filepath.

        Parameters
        ----------
        fpath : str
            The filepath at which to save the serialized simulation result.
        """
        dumpfn(self, fpath)
