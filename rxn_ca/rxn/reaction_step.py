import numpy as np

from ..core import BasicSimulationStep

def get_filter_size_from_side_length(side_length: int) -> int:
    """Provided the side length of the simulation stage, generate the
    appropriate filter size for the simulation. This routine chooses the smaller of:
        1. The filter size that will cover the entire reaction stage
        2. 25, which is a performance compromise
    If the filter size is 25, then we are looking at reactant pairs up to 12 squares
    away from eachother. Given that probability of proceeding scales as 1/d^2, this is equivalent
    to cutting off possibility after the scaling drops below 1/144.

    Args:
        side_length (int): The side length of the simulation stage.

    Returns:
        int: The side length of the filter used in the convolution step.
    """
    return min((side_length - 1) * 2 + 1, 21)

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