from rxn_ca.core import BasicController
from rxn_ca.core.neighborhoods import MooreNeighborhood, Neighborhood, NeighborhoodView
from rxn_ca.rxn.solid_phase_map import SolidPhaseMap
from .growth_result import GrowthResult

class GrowthController(BasicController):

    def __init__(self, phase_map: SolidPhaseMap, neighborhood: Neighborhood = None) -> None:
        self.phase_map: SolidPhaseMap = phase_map

        if neighborhood is None:
            self.neighborhood = MooreNeighborhood(1)
        else:
            self.neighborhood = neighborhood


    def instantiate_result(self):
        return GrowthResult(self.phase_map)

    def get_new_state(self, nb_view: NeighborhoodView):
        if nb_view.center_value == self.phase_map.free_space_id:
            counts = {}
            for cell, _ in nb_view.iterate():
                if cell != self.phase_map.free_space_id and cell != Neighborhood.PADDING_VAL:
                    if cell not in counts:
                        counts[cell] = 1
                    else:
                        counts[cell] += 1

            if len(counts) > 0:
                max_count = 0
                max_spec = None
                for cell, count in counts.items():
                    if count > max_count:
                        max_spec = cell
                        max_count = count

                return int(max_spec), None

            else:
                return int(nb_view.center_value), None
        else:
            return int(nb_view.center_value), None

