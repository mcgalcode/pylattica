from ...core.analyzer import StateAnalyzer

from .consts import OCT_SITE, TET_SITE, Occupancy, Vacant

class DeltaAnalyzer(StateAnalyzer):


    def vacant_oct_sites(self, state):
        return self.get_sites(state, OCT_SITE, { Occupancy: Vacant })

    def num_vacant_oct_sites(self, state):
        return self.get_site_count(state, OCT_SITE, { Occupancy: Vacant })


    def occupied_tet_sites(self, state):
        return self.get_sites(state, TET_SITE, state_false_criteria = { Occupancy: Vacant })

    def num_occupied_tet_sites(self, state):
        return self.get_site_count(state, TET_SITE, state_false_criteria = { Occupancy: Vacant })

    def summary(self, state):
        summary = {
            TET_SITE: {},
            OCT_SITE: {}
        }
        for site in self._structure.sites():

            occupancy = state.get_site_state(site['id'])['occupancy']

            if occupancy in summary[site['site_class']]:
                summary[site['site_class']][occupancy] += 1
            else:
                summary[site['site_class']][occupancy] = 1

        return summary

    def print_ratios(self, state):
        counts = {}

        for site in state.all_site_states():
            occ = site.get(Occupancy)
            if counts.get(occ) is not None:
                counts[occ] += 1
            else:
                counts[occ] = 1

        for spec, count in counts.items():
            print(f'{count} {spec}')