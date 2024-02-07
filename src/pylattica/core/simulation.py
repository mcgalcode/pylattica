from typing import Dict, Tuple

from .constants import SITE_ID
from .periodic_structure import PeriodicStructure
from .simulation_state import SimulationState

import json


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

    def as_dict(self):
        res = {"state": self.state.as_dict(), "structure": self.structure.as_dict()}
        return res

    def to_file(self, fname):
        with open(fname, "w+", encoding="utf-8") as f:
            json.dump(self.as_dict(), f)

    @classmethod
    def from_dict(cls, d):
        state = SimulationState.from_dict(d["state"])
        structure = PeriodicStructure.from_dict(d["structure"])
        return cls(state, structure)

    @classmethod
    def from_file(cls, fname):
        with open(fname, "r+", encoding="utf-8") as f:
            d = json.load(f)
            return cls.from_dict(d)

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
