from typing import Dict, Tuple

from pylattica.core.constants import SITE_ID

from .periodic_structure import PeriodicStructure
from .simulation_state import SimulationState


class Simulation:
    """A wrapper class for binding a SimulationState to the structure
    with which it belongs. Simplifies actions like retrieving simulation
    state based on site location.
    """

    def __init__(self, state: SimulationState, structure: PeriodicStructure):
        """Instantiates a Simulation with the provided state and structure.

        Parameters
        ----------
        state : SimulationState
            The SimulationState
        structure : PeriodicStructure
            The structure to which the SimulationState site IDs refer.
        """
        self.state = state
        self.structure = structure

    def state_at(self, location: Tuple[float]) -> Dict:
        """Retrieves the state of the site at the requested location.

        Parameters
        ----------
        location : Tuple[float]
            The location of the desired site

        Returns
        -------
        Dict
            The state of the found site.
        """
        site = self.structure.site_at(location)

        if site is None:
            return None

        site_state = self.state.get_site_state(site[SITE_ID])
        return site_state
