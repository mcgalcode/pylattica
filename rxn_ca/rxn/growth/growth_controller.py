from rxn_ca.core import BasicController
from rxn_ca.core.neighborhoods import StructureNeighborhoodSpec
from rxn_ca.core.periodic_structure import PeriodicStructure
from rxn_ca.core.simulation_step import SimulationState
from rxn_ca.discrete import PhaseSet
from rxn_ca.grid2d.neighborhoods import MooreNbHoodSpec
from rxn_ca.rxn.solid_phase_set import SolidPhaseSet

class GrowthController(BasicController):

    def __init__(self, phase_set: PhaseSet, periodic_struct: PeriodicStructure, neighborhood_spec: StructureNeighborhoodSpec = None) -> None:
        self.phase_set: PhaseSet = phase_set
        self.struct = periodic_struct

        if neighborhood_spec is None:
            self.neighborhood_spec = MooreNbHoodSpec(1, dim = periodic_struct.dim)
        else:
            self.neighborhood_spec = neighborhood_spec

        self.nb_graph = self.neighborhood_spec.get(periodic_struct)

    def get_state_update(self, site_id: int, prev_state: SimulationState):
        curr_state = prev_state.get_site_state(site_id)
        curr_phase = prev_state.get_site_state(site_id)['_disc_occupancy']
        if curr_state['_disc_occupancy'] == SolidPhaseSet.FREE_SPACE:
            counts = {}
            for nb_id in self.nb_graph.neighbors_of(site_id):
                nb_phase = prev_state.get_site_state(nb_id)['_disc_occupancy']
                if nb_phase != SolidPhaseSet.FREE_SPACE:
                    if nb_phase not in counts:
                        counts[nb_phase] = 1
                    else:
                        counts[nb_phase] += 1

            if len(counts) > 0:
                max_count = 0
                max_spec = None
                for phase, count in counts.items():
                    if count > max_count:
                        max_spec = phase
                        max_count = count

                return { '_disc_occupancy': max_spec }

            else:
                return { '_disc_occupancy': curr_phase }
        else:
            return { '_disc_occupancy': curr_phase }

