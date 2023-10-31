from typing import List

from ..core import SimulationState, StateAnalyzer
from .state_constants import DISCRETE_OCCUPANCY


class DiscreteStepAnalyzer(StateAnalyzer):
    """Implements simple utilities for analyzing simulation states which are
    specified by categorical occupancies for each site in the simulation
    """

    def cell_fraction(self, state: SimulationState, phase_name: str) -> float:
        """Returns the fraction of sites in the provided state which are occupied
        by the specified phase.

        Parameters
        ----------
        state : SimulationState
            The state to analyze.
        phase_name : str
            The name of the phase for which a fraction of sites should be calculated

        Returns
        -------
        float
            The fraction of sites occupied by the specified phase.
        """
        phase_count = self.cell_count(state, phase_name)
        total_occupied_cells = state.size
        return phase_count / total_occupied_cells

    def cell_count(self, state: SimulationState, phase_name: str) -> int:
        """Retrieves the number of cells in the given simulation state occupied by
        the specified phase.

        Parameters
        ----------
        state : SimulationState
            The state to analyze.
        phase_name : str
            The name of the phase to count.

        Returns
        -------
        int
            The number of sites occupied by the specified phase.
        """
        return self.get_site_count_where_equal(state, {DISCRETE_OCCUPANCY: phase_name})

    def cell_ratio(self, step: SimulationState, p1: str, p2: str) -> float:
        """Returns the occupancy ratio between two phases in the provided simulation state.

        Parameters
        ----------
        step : SimulationState
            The state to analyze.
        p1 : str
            The name of the first phase.
        p2 : str
            The name of the second phase

        Returns
        -------
        float
            The ratio of the occupancies of the two phases.
        """
        return self.cell_count(step, p1) / self.cell_count(step, p2)

    def phase_count(self, step: SimulationState) -> int:
        """The number of phases present in the specified simulation state.

        Parameters
        ----------
        step : SimulationState
            The state to analyze.

        Returns
        -------
        int
            The number of phases identified.
        """
        return len(self.phases_present(step))

    def phases_present(self, state: SimulationState) -> List[str]:
        """Returns a list of the phases present in the specified state.

        Parameters
        ----------
        state : SimulationState
            The state to analyze.

        Returns
        -------
        List[str]
            A list of the phases identified.
        """
        return list(
            set(
                site_state[DISCRETE_OCCUPANCY] for site_state in state.all_site_states()
            )
        )
