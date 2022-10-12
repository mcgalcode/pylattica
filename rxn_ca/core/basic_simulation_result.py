from monty.serialization import dumpfn, loadfn

from rxn_ca.core.simulation_step import SimulationState

class BasicSimulationResult():
    """A class that stores the result of running a simulation. Keeps track of all
    the steps that the simulation proceeded through, and the set of reactions that
    was used in the simulation.
    """

    @classmethod
    def from_file(cls, fpath):
        return loadfn(fpath)

    @classmethod
    def from_dict(cls, res_dict):
        steps = res_dict["steps"]
        res = cls()
        for step in steps:
            res.add_step(step)
        return res

    def __init__(self, starting_state: SimulationState):
        """Initializes a ReactionResult with the reaction set used in the simulation

        Args:
            rxn_set (ScoredReactionSet):
        """
        self.initial_state = starting_state
        self._steps: list[SimulationState] = [starting_state]
        self._diffs: list[dict] = []

    def add_step(self, updates: dict) -> None:
        """Adds a step to the reaction result. Using this method allows
        the accumulation of all the steps in a given simulation.

        Args:
            step (dict): _description_
        """
        new_step = self._steps[-1].copy()
        new_step.batch_update(updates)
        self._steps.append(new_step)
        self._diffs.append(updates)

    def __len__(self) -> int:
        return len(self._diffs) + 1

    @property
    def last_step(self):
        return self.get_step(-1)

    @property
    def first_step(self):
        return self.get_step(0)

    def get_step(self, step):
        return self._steps[step]

    def as_dict(self):
        return {
            "steps": [s.as_dict() for s in self._steps],
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

    def to_file(self, fpath):
        dumpfn(self, fpath)