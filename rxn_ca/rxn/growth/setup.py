from random import randint
import numpy as np

from rxn_ca.core import BasicSimulationStep
from rxn_ca.core.neighborhoods import MooreNeighborhood, Neighborhood
from rxn_ca.rxn.solid_phase_map import SolidPhaseMap

class GrowthSetup():

    def __init__(self, phase_map: SolidPhaseMap, buffer = 1):
        self.phase_map = phase_map
        self.nb = MooreNeighborhood(buffer)

    def _build_empty_state(self, size: int) -> np.array:
        state: np.array = np.ones((size, size)) * self.phase_map.free_space_id
        return state

    def setup(self, size, num_sites_desired, nuc_species, nuc_ratios = None):
        blank = self._build_empty_state(size)
        if nuc_ratios is None:
            nuc_ratios = np.ones((len(nuc_species)))

        specie_idxs = np.array(range(0, len(nuc_species)))
        normalized_ratios = nuc_ratios / np.sum(nuc_ratios)

        total_attempts = 0
        num_sites_planted = 0

        while num_sites_planted < num_sites_desired:
            if total_attempts > 100 * num_sites_desired:
                raise RuntimeError(f'Too many nucleation sites at the specified buffer: {total_attempts} made at placing nuclei')

            rand_i = randint(0,size - 1)
            rand_j = randint(0,size - 1)
            if blank[rand_i][rand_j] != self.phase_map.free_space_id:
                total_attempts += 1
                continue

            found_existing_nucleus_in_nb = False
            for cell, _ in self.nb.iterate_state(blank, rand_i, rand_j, exclude_center=False):
                if cell != Neighborhood.PADDING_VAL and cell != self.phase_map.free_space_id:
                    found_existing_nucleus_in_nb = True
            if not found_existing_nucleus_in_nb:
                chosen_spec = nuc_species[np.random.choice(specie_idxs, p=normalized_ratios)]
                blank[rand_i][rand_j] = self.phase_map.phase_to_int[chosen_spec]
                num_sites_planted += 1

            total_attempts += 1
        print(total_attempts)
        return BasicSimulationStep(blank)