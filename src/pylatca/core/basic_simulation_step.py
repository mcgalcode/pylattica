import numpy as np

class BasicSimulationStep():
    """A class representing the system state during a simulation. This state keeps track
    of which phase is in which position on the grid, and handles padding the state to accommodate
    whatever the chosen filter size is. It is responsible for most of the calculation in the
    simulation procedure.
    """

    @classmethod
    def from_dict(cls, step_dict):
        return cls(np.array(step_dict["state"]).astype(int), step_dict["metadata"])

    def __init__(self, state: np.array, metadata = []):
        """Intializes a ReactionStep.

        Args:
            state (np.array): A 3D array
        """
        self.size: int = state.shape[0]
        self.shape: tuple = state.shape
        self.state: np.array = state
        self.metadata = metadata
        self.dim = len(state.shape)

    def as_dict(self):
        return {
            "state": self.state.tolist(),
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }
