from abc import ABC, abstractmethod
from rxn_ca.core.basic_simulation_result import BasicSimulationResult

from rxn_ca.core.simulation_step import SimulationState


class BasicController(ABC):

    @abstractmethod
    def get_state_update(self, site_id: int, prev_state: SimulationState):
        pass

    def instantiate_result(self, starting_state):
        return BasicSimulationResult(starting_state = starting_state)
