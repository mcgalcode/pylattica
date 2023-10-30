from ...core import BasicController
from ...core.neighborhood_builders import NeighborhoodBuilder
from ...core.periodic_structure import PeriodicStructure
from ...core.simulation_state import SimulationState
from ...discrete import PhaseSet
from ...discrete.state_constants import DISCRETE_OCCUPANCY, VACANT
from ...structures.square_grid.neighborhoods import MooreNbHoodBuilder


class GrowthController(BasicController):
    def __init__(
        self,
        phase_set: PhaseSet,
        periodic_struct: PeriodicStructure,
        background_phase: str = VACANT,
        nb_builder: NeighborhoodBuilder = None,
    ) -> None:
        self.background_phase = background_phase
        self.phase_set: PhaseSet = phase_set

        if nb_builder is None:
            self.nb_builder = MooreNbHoodBuilder(1, dim=periodic_struct.dim)
        else:
            self.nb_builder = nb_builder

        self.nb_graph = self.nb_builder.get(periodic_struct)

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

                return {DISCRETE_OCCUPANCY: max_spec}

            else:
                return {}
        else:
            return {}
