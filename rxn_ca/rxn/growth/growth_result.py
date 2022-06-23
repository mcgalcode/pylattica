import numpy as np

from rxn_ca.discrete import DiscreteStateResult
from rxn_ca.rxn.solid_phase_map import SolidPhaseMap

class GrowthResult(DiscreteStateResult):
    """A class that stores the result of running a simulation. Keeps track of all
    the steps that the simulation proceeded through, and the set of reactions that
    was used in the simulation.
    """


    def __init__(self, phase_map: SolidPhaseMap):
        """Initializes a ReactionResult with the reaction set used in the simulation

        Args:
            rxn_set (ScoredReactionSet):
        """
        super().__init__(phase_map)

