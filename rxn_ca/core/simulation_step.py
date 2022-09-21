import copy
from rxn_ca.core.periodic_structure import PeriodicStructure

GENERAL = "GENERAL"

class SimulationState():

    def as_dict(self):
        return {
            "state": self._state,
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
        }

    def __init__(self, state = None):
        if state is None:
            self._state = {}
        else:
            self._state = copy.deepcopy(state)

    @property
    def size(self):
        return len(self.site_ids())

    def site_ids(self):
        return list(self._state.keys())

    def all_site_states(self):
        return list(self._state.values())

    def get_site_state(self, site_id: int):
        return self._state.get(site_id)

    def get_general_state(self):
        return self._state.get(GENERAL)

    def set_general_state(self, updates):
        old_state = self.get_general_state()
        if old_state is None:
            old_state = {}

        self._state[GENERAL] = { **old_state, **updates }

    def set_site_state(self, site_id: int, updates: dict):
        old_state = self._state.get(site_id)
        if old_state is None:
            old_state = {}

        self._state[site_id] = { **old_state, **updates }

    def batch_update(self, update_batch):
        for site_id, updates in update_batch.items():
            self.set_site_state(site_id, updates)

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

