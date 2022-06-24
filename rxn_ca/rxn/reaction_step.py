import numpy as np

from ..core import BasicSimulationStep

class ReactionStep(BasicSimulationStep):
    """A class representing the system state during a simulation. This state keeps track
    of which phase is in which position on the grid, and handles padding the state to accommodate
    whatever the chosen filter size is. It is responsible for most of the calculation in the
    simulation procedure.
    """

    def __init__(self, unpadded_state: np.array, metadata = []):
        """Intializes a ReactionStep.

        Args:
            state (np.array): A 2D np.array of integers representing the distribution of phases.
                The value of the integers corresponds to phases in the reaction_set's int_to_phase mapping.
        """
        super().__init__(unpadded_state, metadata)
        self.reaction_choices = {}

        if len(metadata) > 0:

            for chosen_rxn in metadata:
                rxn_str = str(chosen_rxn)
                if rxn_str in self.reaction_choices:
                    self.reaction_choices[rxn_str] += 1
                else:
                    self.reaction_choices[rxn_str] = 1