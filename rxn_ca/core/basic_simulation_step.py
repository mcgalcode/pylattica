import numpy as np

class BasicSimulationStep():
    """A class representing the system state during a simulation. This state keeps track
    of which phase is in which position on the grid, and handles padding the state to accommodate
    whatever the chosen filter size is. It is responsible for most of the calculation in the
    simulation procedure.
    """

    @classmethod
    def from_dict(cls, step_dict):
        return cls(np.array(step_dict["state"]).astype(int))

    def __init__(self, state: np.array, metadata = []):
        """Intializes a ReactionStep.

        Args:
            state (np.array): A 2D np.array of integers representing the distribution of phases.
                The value of the integers corresponds to phases in the reaction_set's int_to_phase mapping.
        """
        self.size: int = state.shape[0]
        self.shape: tuple(int, int) = state.shape
        self.state: np.array = state
        self.metadata = metadata

    def to_dict(self):
        return {
            "state": self.state.tolist(),
            "metadata": self.metadata_to_dict()
        }

    def metadata_to_dict(self):
        return self.metadata