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
        self.steps: list[SimulationState] = [starting_state]

    def add_step(self, updates: dict) -> None:
        """Adds a step to the reaction result. Using this method allows
        the accumulation of all the steps in a given simulation.

        Args:
            step (dict): _description_
        """
        new_step = self.steps[-1].copy()
        new_step.batch_update(updates)
        self.steps.append(new_step)
        # self.steps.append(updates)

    # def _rebuild_until(self, step_no = None):
    #     if step_no is None:
    #         step_no = len(self.steps)

    #     rebuilt_state = self.initial_state.copy()
    #     for i in range(step_no):
    #         rebuilt_state.batch_update(self.steps[i])
    #     return rebuilt_state

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