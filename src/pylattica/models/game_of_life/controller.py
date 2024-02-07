from ...core import BasicController, SimulationState, PeriodicStructure
from ...structures.square_grid import MooreNbHoodBuilder
from ...discrete.state_constants import DISCRETE_OCCUPANCY


def process_variant_string(v_string):
    halves = v_string.split("/")
    born = [int(el) for el in list(halves[0][1:])]
    survive = [int(el) for el in list(halves[1][1:])]
    return born, survive


class GameOfLifeController(BasicController):
    variant = None

    def __init__(self, structure: PeriodicStructure, variant="B3/S23"):
        if self.variant is None:
            self.variant = variant
        self.born, self.survive = process_variant_string(self.variant)
        self.structure = structure

    def pre_run(self, _):
        self.neighborhood = MooreNbHoodBuilder().get(self.structure)

    def get_state_update(self, site_id, curr_state: SimulationState):
        alive_neighbor_count = 0
        dead_neighbor_count = 0

        neighbor_site_ids = self.neighborhood.neighbors_of(site_id)

        for nb_id in neighbor_site_ids:
            neighbor_state = curr_state.get_site_state(nb_id)[DISCRETE_OCCUPANCY]
            if neighbor_state == "alive":
                alive_neighbor_count += 1
            else:
                dead_neighbor_count += 1

        current_state = curr_state.get_site_state(site_id)[DISCRETE_OCCUPANCY]

        if current_state == "alive" and alive_neighbor_count in self.survive:
            new_state = "alive"
        elif current_state == "dead" and alive_neighbor_count in self.born:
            new_state = "alive"
        else:
            new_state = "dead"

        updates = {DISCRETE_OCCUPANCY: new_state}
        return updates


Life = "B3/S23"
Anneal = "B4678/S35678"
Diamoeba = "B35678/S5678"
Seeds = "B2/S"
Maze = "B3/S12345"
