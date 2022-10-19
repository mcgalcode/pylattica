import copy
from turtle import update

from pylattica.core.constants import SITE_ID
from .periodic_structure import PeriodicStructure

GENERAL = "GENERAL"
SITES = "SITES"

class SimulationState():

    def as_dict(self):
        return {
            "state": self._state,
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

    def __init__(self, state = None):
        if state is None:
            self._state = {
                SITES: {},
                GENERAL: {},
            }
        else:
            self._state = copy.deepcopy(state)

    @property
    def size(self):
        return len(self.site_ids())

    def site_ids(self):
        return list(self._state[SITES].keys())

    def all_site_states(self):
        return [site_state for site_state in self._state[SITES].values()]

    def get_site_state(self, site_id: int):
        return self._state[SITES].get(site_id)

    def get_general_state(self):
        return self._state.get(GENERAL)

    def set_general_state(self, updates):
        old_state = self.get_general_state()
        self._state[GENERAL] = { **old_state, **updates }

    def set_site_state(self, site_id: int, updates: dict):
        old_state = self._state[SITES].get(site_id)
        if old_state is None:
            old_state = {
                SITE_ID: site_id
            }

        self._state[SITES][site_id] = { **old_state, **updates }

    def batch_update(self, update_batch):
        if SITES in update_batch:
            for site_id, updates in update_batch[SITES].items():
                self.set_site_state(site_id, updates)

        if GENERAL in update_batch:
            self.set_general_state(update_batch[GENERAL])



    def copy(self):
        return SimulationState(self._state)

class SimulationStep():

    def __init__(self, state: SimulationState, site_structure: PeriodicStructure):
        self._state = state
        self._site_structure = site_structure

    @property
    def struct(self):
        return self._site_structure

    @property
    def state(self):
        return self._state

