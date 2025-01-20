from __future__ import annotations

import copy
from typing import Dict, List

from .constants import SITE_ID, SITES, GENERAL
from .periodic_structure import PeriodicStructure


class SimulationState:
    """Representation of the state during a single step of the simulation. This is essentially
    a dictionary that maps the IDs of sites in the simulation structure to dictionaries with
    arbitrary keys and values that can store whatever state is relevant for the simulation.

    Additionally, there is a concept of general simulation state that is separate from the state
    of any specific site in the simulation.
    """

    def as_dict(self):
        return {
            "state": self._state,
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, state_dict):
        state = state_dict["state"]
        state[SITES] = {int(k): v for k, v in state[SITES].items()}
        return cls(state)

    @classmethod
    def from_struct(cls, struct: PeriodicStructure):
        state = cls()

        for sid in struct.site_ids:
            state.set_site_state(sid, {})

        return state

    def __init__(self, state: Dict = None):
        """Initializes the SimulationState.

        Parameters
        ----------
        state : dict, optional
            A state to store. should be a map with keys "GENERAL" and "SITES", by default None
        """
        if state is None:
            self._state = {
                SITES: {},
                GENERAL: {},
            }
        else:
            self._state = copy.deepcopy(state)

    @property
    def size(self) -> int:
        """Gives the number of sites for which state information is stored.

        Returns
        -------
        int
            The number of sites for which state information is stored.
        """
        return len(self.site_ids())

    def site_ids(self) -> List[int]:
        """A list of site IDs for which some state is stored.

        Returns
        -------
        List[int]
        """
        return list(self._state[SITES].keys())

    def all_site_states(self) -> List[Dict]:
        """Returns a list of dictionaries representing the site
        state values.

        Returns
        -------
        List[Dict]
            The state dictionaries for every site in this state.
        """
        return list(self._state[SITES].values())

    def get_site_state(self, site_id: int) -> Dict:
        """Returns the state stored for the specified site ID, if any.

        Parameters
        ----------
        site_id : int
            The ID of the site for which state information should be retrieved.

        Returns
        -------
        Dict
            The state of that site. Returns None if no state is stored under that site ID.
        """
        return self._state[SITES].get(site_id)

    def get_general_state(self, key: str = None, default=None) -> Dict:
        """Returns the general state.

        Returns
        -------
        Dict
            The general state.
        """
        if key is None:
            return copy.deepcopy(self._state.get(GENERAL))
        else:
            return copy.deepcopy(self._state.get(GENERAL)).get(key, default)

    def set_general_state(self, updates: Dict) -> None:
        """Updates the general state with the keys and values provided by the updates parameter.

        Parameters
        ----------
        updates : Dict
            The updates to apply. Note that this overwrites values in the old state, but unspecified
            values are left unchanged.
        """
        old_state = self.get_general_state()
        self._state[GENERAL] = {**old_state, **updates}

    def set_site_state(self, site_id: int, updates: dict) -> None:
        """Updates the state stored for site with ID site_id.

        Parameters
        ----------
        site_id : int
            The ID of the site for which the state should be updated.
        updates : dict
            The updates to the state that should be performed.
        """
        old_state = self._state[SITES].get(site_id)
        if old_state is None:
            old_state = {SITE_ID: site_id}

        self._state[SITES][site_id] = {**old_state, **updates}

    def batch_update(self, update_batch: Dict) -> None:
        """Applies a batch update to many sites and the general state. Takes a dictionary
        formatted like this:

        {
            "GENERAL": {...},
            "SITES": {
                1: {...}
            }
        }

        Parameters
        ----------
        update_batch : Dict
            The updates to apply as a batch.
        """

        if GENERAL in update_batch:
            for site_id, updates in update_batch.get(SITES, {}).items():
                self.set_site_state(site_id, updates)

            self.set_general_state(update_batch[GENERAL])
        else:
            for site_id, updates in update_batch.items():
                self.set_site_state(site_id, updates)

    def copy(self) -> SimulationState:
        """Creates a new simulation state identical to this one. This is a deepcopy
        operation, so changing the copy will not change the original.

        Returns
        -------
        SimulationState
            The copy of this SimulationState
        """
        return SimulationState(self._state)

    def as_state_update(self) -> Dict:
        return copy.deepcopy(self._state)

    def __eq__(self, other: SimulationState) -> bool:
        return self._state == other._state
