from collections import deque
from math import ceil
from rxn_ca.core import SimulationState

import numpy as np

from rxn_ca.delta.analyzer import DeltaAnalyzer
from .consts import InitialAtomCount, InitialLiCount, Li, Mn, Mn2Count, RemovedLiCount, Ti, Vacant, Occupancy, OCT_SITE, TET_SITE

class DeltaSetup():


    def setup(self, struct, ratios):
        state = SimulationState()

        analyzer = DeltaAnalyzer(struct)

        species = [Li, Ti, Mn]

        site_counts = np.array(ratios) / sum(ratios) * len(struct.sites(OCT_SITE))
        spec_assignments = [spec for count, spec in zip(site_counts, species) for _ in range(ceil(count))]
        np.random.shuffle(spec_assignments)
        spec_assignments_queue = deque()
        for a in spec_assignments:
            spec_assignments_queue.append(a)

        print(len(spec_assignments_queue), len(struct.sites(OCT_SITE)))
        for site in struct.sites(OCT_SITE):

            state.set_site_state(site["id"], {
                Occupancy: spec_assignments_queue.popleft()
            })

        for site in struct.sites(TET_SITE):
            state.set_site_state(site["id"], {
                Occupancy: Vacant
            })

        state.set_general_state({
            Mn2Count: 0,
            InitialLiCount: analyzer.get_site_count(state, state_true_criteria={ Occupancy: Li }),
            InitialAtomCount: analyzer.get_site_count(state, state_false_criteria={ Occupancy: Vacant }),
            RemovedLiCount: 0
        })

        return state