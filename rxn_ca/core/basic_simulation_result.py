from monty.serialization import dumpfn, loadfn
from .basic_simulation_step import BasicSimulationStep

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

    def __init__(self):
        """Initializes a ReactionResult with the reaction set used in the simulation

        Args:
            rxn_set (ScoredReactionSet):
        """
        self.steps: list[BasicSimulationStep] = []

    def add_step(self, step: BasicSimulationStep) -> None:
        """Adds a step to the reaction result. Using this method allows
        the accumulation of all the steps in a given simulation.

        Args:
            step (BasicSimulationStep): _description_
        """
        self.steps.append(step)

    def get_metadata_at(self, step_no: int) -> None:
        return self.steps[step_no].metadata

    @property
    def last_step(self):
        return self.steps[-1]

    @property
    def first_step(self):
        return self.steps[0]

    def as_dict(self):
        return {
            "steps": [s.as_dict() for s in self.steps],
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

    def to_file(self, fpath):
        dumpfn(self, fpath)