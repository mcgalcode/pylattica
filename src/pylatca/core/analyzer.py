from .periodic_structure import PeriodicStructure
from .simulation_state import SimulationState

class StateAnalyzer():

    def __init__(self, structure: PeriodicStructure):
        self._structure = structure

    def get_sites(self, state: SimulationState, site_class = None, state_true_criteria = None, state_false_criteria = None):
        if site_class is None:
            sites = self._structure.sites()
        else:
            sites = self._structure.sites(site_class)

        if state_true_criteria is None:
            state_true_criteria = {}

        if state_false_criteria is None:
            state_false_criteria = {}

        matching_sites = []
        for site in sites:
            site_state = state.get_site_state(site['id'])
            meets_criteria = True

            for crit_key, crit_val in state_true_criteria.items():
                if site_state.get(crit_key) != crit_val:
                    meets_criteria = False

            for crit_key, crit_val in state_false_criteria.items():
                if site_state.get(crit_key) == crit_val:
                    meets_criteria = False

            if meets_criteria:
                matching_sites.append(site['id'])

        return matching_sites

    def get_site_count(self, state, site_class = None, state_true_criteria = None, state_false_criteria = None):
        return len(self.get_sites(state, site_class, state_true_criteria, state_false_criteria))