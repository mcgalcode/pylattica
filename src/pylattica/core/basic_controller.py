from abc import ABC, abstractmethod
import random

from .simulation_result import SimulationResult
from .simulation_state import SimulationState


class BasicController(ABC):
    """The base class which all Controllers extend. Every new type of
    simulation will involve creating a new Controller class. The controller
    class has a single responsibility, which is to implement the update
    rule for the simulation.

    To do this, implement the get_state_update method. The entire current
    SimulationState will be passed to this method, along with the ID of
    the site at which the update rule should be applied. It is up to the
    user to decide what updates should be produced using this information.
    """

    @abstractmethod
    def get_state_update(self, site_id: int, prev_state: SimulationState):
        pass  # pragma: no cover

    def pre_run(self, initial_state: SimulationState) -> None:
        pass

    def get_random_site(self, state: SimulationState):
        return random.randint(0, len(state.site_ids()) - 1)

    def instantiate_result(self, starting_state: SimulationState):
        return SimulationResult(starting_state=starting_state)
