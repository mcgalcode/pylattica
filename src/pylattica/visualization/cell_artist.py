from abc import abstractmethod

import typing
from .colors import COLORS
from ..discrete.state_constants import DISCRETE_OCCUPANCY
from ..discrete.discrete_step_analyzer import DiscreteStepAnalyzer
from ..discrete.discrete_state_result_analyzer import DiscreteResultAnalyzer
from ..core import SimulationResult

from ..core import SimulationState
from typing import Dict


class CellArtist:
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
    @classmethod
    def from_phase_list(cls, phases):
        color_map = cls.build_color_map_from_phase_list(phases)
        return cls(color_map)

    @classmethod
    def from_discrete_state(cls, state: SimulationState):
        analyzer = DiscreteStepAnalyzer()
        phases = analyzer.phases_present(state)
        return cls.from_phase_list(phases)

    @classmethod
    def from_discrete_result(
        cls, result: SimulationResult
    ) -> typing.Dict[str, typing.Tuple[int, int, int]]:
        """Returns a map of phases to colors that can be used to visualize the phases

        Returns:
            typing.Dict[str, typing.Tuple[int, int ,int]]: A mapping of phase name to RGB
            color values
        """
        analyzer = DiscreteResultAnalyzer(result)
        return cls.from_phase_list(analyzer.all_phases())

    @classmethod
    def build_color_map_from_phase_list(cls, phases):
        """Returns a map of phases to colors that can be used to visualize the phases

        Returns:
            typing.Dict[str, typing.Tuple[int, int ,int]]: A mapping of phase name to RGB
            color values
        """
        display_phases: typing.Dict[str, typing.Tuple[int, int, int]] = {}
        c_idx: int = 0

        for p in phases:
            display_phases[p] = COLORS[c_idx % len(COLORS)]
            c_idx += 1

        return display_phases

    def __init__(self, color_map=None):
        self.color_map = color_map

    def get_color_from_cell_state(self, cell_state: Dict):
        phase_name = cell_state[DISCRETE_OCCUPANCY]
        return self.color_map[phase_name]

    def get_cell_legend_label(self, cell_state: Dict):
        return cell_state[DISCRETE_OCCUPANCY]
