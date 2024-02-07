from __future__ import annotations
from abc import abstractmethod

from .colors import COLORS
from ..discrete.state_constants import DISCRETE_OCCUPANCY
from ..discrete.discrete_step_analyzer import DiscreteStepAnalyzer
from ..discrete.discrete_state_result_analyzer import DiscreteResultAnalyzer
from ..core import SimulationResult

from ..core import SimulationState
from typing import Dict, List, Tuple


class CellArtist:
    """A strategy class for implementing methods of coloring cells in a
    simulation visualization."""

    @abstractmethod
    def get_color_from_cell_state(self, cell_state: Dict):
        pass  # pragma: no cover

    @abstractmethod
    def get_cell_legend_label(self, cell_state: Dict):
        pass  # pragma: no cover

    def get_legend(self, simulation_state: SimulationState):
        legend = {}
        for state in simulation_state.all_site_states():
            label = self.get_cell_legend_label(state)
            if label not in legend:
                color = self.get_color_from_cell_state(state)
                legend[label] = color

        return legend


class DiscreteCellArtist(CellArtist):
    """A class for coloring cells based on their discrete phase occupancy."""

    @classmethod
    def from_phase_list(cls, phases: List[str]) -> DiscreteCellArtist:
        """Generates a DiscreteCellArtist from a list of phases.

        Parameters
        ----------
        phases : List[str]
            The phases expected in the Simulation.

        Returns
        -------
        DiscreteCellArtist
            The resulting artist.
        """
        color_map = cls.build_color_map_from_phase_list(phases)
        return cls(color_map)

    @classmethod
    def from_discrete_state(cls, state: SimulationState) -> DiscreteCellArtist:
        """Generates a DiscreteCellArtist from the phases present in the
        provided simulation state.

        Parameters
        ----------
        state : SimulationState
            The state to generate the artist from.

        Returns
        -------
        DiscreteCellArtist
            The resulting artist.
        """
        analyzer = DiscreteStepAnalyzer()
        phases = analyzer.phases_present(state)
        return cls.from_phase_list(phases)

    @classmethod
    def from_discrete_result(cls, result: SimulationResult) -> DiscreteCellArtist:
        """Generates a DiscreteCellArtist from the phases present
        in a SimulationResult.

        Parameters
        ----------
        result : SimulationResult
            The result from which the artist should be generated.

        Returns
        -------
        DiscreteCellArtist
            The resulting artist.
        """

        analyzer = DiscreteResultAnalyzer(result)
        return cls.from_phase_list(analyzer.all_phases())

    @classmethod
    def build_color_map_from_phase_list(
        cls, phases: List[str]
    ) -> Dict[str, Tuple[int, int, int]]:
        """Generates a map of phase name to RGB color from a list of phase names.

        Parameters
        ----------
        phases : List[str]
            The phases to include in the color mapping.

        Returns
        -------
        Dict[str, Tuple[int, int, int]]
            A dictionary mapping the phase names to unique color values.
        """
        display_phases: Dict[str, Tuple[int, int, int]] = {}
        c_idx: int = 0

        for p in phases:
            display_phases[p] = COLORS[c_idx % len(COLORS)]
            c_idx += 1

        return display_phases

    def __init__(self, color_map: Dict[str, Tuple[int, int, int]]):
        """Instantiates the DiscreteCellArtist

        Parameters
        ----------
        color_map : Dict[str, Tuple[int, int, int]]
            A mapping of phase name to color.
        """
        self.color_map = color_map

    def get_color_from_cell_state(self, cell_state: Dict):
        """Returns the color associated with a particular cell state.

        Parameters
        ----------
        cell_state : Dict
            The cell state to assess for color.

        Returns
        -------
        Tuple[int, int, int]
            The color associated with the specified state.
        """
        phase_name = cell_state[DISCRETE_OCCUPANCY]
        return self.color_map[phase_name]

    def get_cell_legend_label(self, cell_state: Dict) -> str:
        """Get the legend label associated with a particular cell state.

        Parameters
        ----------
        cell_state : Dict
            The state for which the legend label should be returned.

        Returns
        -------
        str
            The legend label for the provided state.
        """
        return cell_state[DISCRETE_OCCUPANCY]
