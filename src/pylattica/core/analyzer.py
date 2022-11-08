from typing import Callable, Dict, List

from .constants import SITE_ID
from .periodic_structure import PeriodicStructure
from .simulation_state import SimulationState


class StateAnalyzer:
    """Provides basic functionality for analyzing a SimulationState object."""

    def __init__(self, structure: PeriodicStructure = None):
        """Ininitializes the StateAnalyzer with the structure provided.
        The structure is used to filter sites down by site class, if desired.

        Parameters
        ----------
        structure : PeriodicStructure
            The structure to use as the source for site class information.
        """
        self._structure = structure

    def get_sites(
        self,
        state: SimulationState,
        site_class: str = None,
        state_criteria: List[Callable[[Dict], bool]] = None,
    ) -> List[int]:
        """Retrieves a list of site states matching the specified criteria.
        Criteria are expressed as functions or lambdas that take a site state dictionary
        and return a boolean indicating whether or not the site satisfies the criteria.

        Parameters
        ----------
        state : SimulationState
            The simulation state to analyze
        site_class : str, optional
            If desired, a site class to filter down on, by default None
        state_criteria : List[Callable[[Dict], bool]], optional
            A list of functions which must return true for a site to be included in the returned sites, by default None

        Returns
        -------
        List[Dict]
            A list of site IDs satisfying the specified criteria and belonging to the specified site class.
        """
        if state_criteria is None:
            state_criteria = []

        if self._structure is not None:
            sites = self._structure.sites(site_class)
            sites = [state.get_site_state(site[SITE_ID]) for site in sites]
        else:
            sites = state.all_site_states()

        matching_sites = []
        for site in sites:
            site_state = state.get_site_state(site[SITE_ID])
            meets_criteria = True

            for crit in state_criteria:
                if not crit(site_state):
                    meets_criteria = False
                    break

            if meets_criteria:
                matching_sites.append(site[SITE_ID])

        return matching_sites

    def get_sites_where_equal(
        self, state: SimulationState, search_pairs: Dict, site_class: str = None
    ) -> List[int]:
        """Returns sites whose state dictionaries contain values matching the
        search_pairs parameters passed here. For instance, if you wanted every site
        with had a state value for property "B" equal to 2, search_pairs would be:

        {
            "B": 2
        }

        Also supports filtering sites based on site class.

        Parameters
        ----------
        state : SimulationState
            The state to filter sites from.
        search_pairs : Dict
            The dictionary containing key value pairs that must match the site state.
        site_class : str, optional
            The class of sites to consider, by default None

        Returns
        -------
        List[int]
            A list of site IDs corresponding to the matching sites
        """

        def _criteria(site_state: Dict) -> bool:
            for state_key, state_val in search_pairs.items():
                if site_state.get(state_key) != state_val:
                    return False
            return True

        return self.get_sites(state, site_class=site_class, state_criteria=[_criteria])

    def get_site_count(
        self,
        state: SimulationState,
        site_class: str = None,
        state_criteria: List[Callable[[Dict], bool]] = None,
    ) -> int:
        """Counts the sites in the state matching the specified criteria. See documentation
        for get_sites for more details.

        Parameters
        ----------
        state : SimulationState
            The simulation state to search
        site_class : str, optional
            The class of sites to consider, by default None
        state_criteria : List[Callable[[Dict], bool]], optional
            The criteria by which a site should be assessed for filtering, by default None

        Returns
        -------
        int
            The number of sites matching the specified criteria
        """
        return len(
            self.get_sites(state, site_class=site_class, state_criteria=state_criteria)
        )

    def get_site_count_where_equal(
        self, state: SimulationState, search_pairs: Dict, site_class: str = None
    ) -> int:
        """Counts the sites which have state values equal to those specified by search_pairs.
        See get_sites_equal for a specification of the search_pairs parameter.

        Parameters
        ----------
        state : SimulationState
            The state to search for sites.
        search_pairs : Dict
            The state values that must be matched
        site_class : str, optional
            The class of sites to consider, by default None

        Returns
        -------
        int
            The number of sites matching the criteria
        """
        return len(
            self.get_sites_where_equal(
                state, site_class=site_class, search_pairs=search_pairs
            )
        )
