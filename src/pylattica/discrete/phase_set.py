from typing import List


class PhaseSet:
    """A lightweight class for representing the set of phases possible in
    a simulation with discrete state occupancies.
    """

    def __init__(self, phases: List[str]):
        """Instantiates the PhaseSet.

        Parameters
        ----------
        phases : List[str]
            The phases that exist in this simulation.
        """
        self.phases = list(set(phases))
