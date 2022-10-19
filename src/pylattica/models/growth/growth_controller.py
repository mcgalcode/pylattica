from ...core import BasicController
from ...core.neighborhood_builders import NeighborhoodBuilder
from ...core.periodic_structure import PeriodicStructure
from ...core.simulation_state import SimulationState
from ...discrete import PhaseSet
from ...discrete.state_constants import DISCRETE_OCCUPANCY, VACANT
from ...square_grid.neighborhoods import MooreNbHoodBuilder

class GrowthController(BasicController):

    def __init__(self, phase_set: PhaseSet, 
                       periodic_struct: PeriodicStructure,
                       background_phase: str = VACANT,
                       neighborhood_spec: NeighborhoodBuilder = None
    ) -> None:
        self.background_phase = background_phase
        self.phase_set: PhaseSet = phase_set

        if neighborhood_spec is None:
            self.neighborhood_spec = MooreNbHoodBuilder(1, dim = periodic_struct.dim)
        else:
            self.neighborhood_spec = neighborhood_spec

        self.nb_graph = self.neighborhood_spec.get(periodic_struct)

    def get_state_update(self, site_id: int, prev_state: SimulationState):
        curr_state = prev_state.get_site_state(site_id)
        if curr_state[DISCRETE_OCCUPANCY] == self.background_phase:
            counts = {}
            for nb_id in self.nb_graph.neighbors_of(site_id):
                nb_phase = prev_state.get_site_state(nb_id)[DISCRETE_OCCUPANCY]
                if nb_phase != self.background_phase:
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

                return { DISCRETE_OCCUPANCY: max_spec }

            else:
                return {}
        else:
            return {}

