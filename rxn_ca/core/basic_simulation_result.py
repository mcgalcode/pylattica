import json

from .basic_simulation_step import BasicSimulationStep

class BasicSimulationResult():
    """A class that stores the result of running a simulation. Keeps track of all
    the steps that the simulation proceeded through, and the set of reactions that
    was used in the simulation.
    """

    @classmethod
    def from_file(cls, fpath):
        with open(fpath, 'r') as f:
            return cls.from_dict(json.loads(f.read()))

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

    def to_dict(self):
        return {
            "steps": [s.to_dict() for s in self.steps]
        }

    def to_file(self, fpath):
        with open(fpath, 'w+') as f:
            f.write(json.dumps(self.to_dict()))
