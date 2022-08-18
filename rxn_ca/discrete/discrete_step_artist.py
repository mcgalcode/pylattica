import typing
from rxn_ca.core import BasicSimulationStep
import numpy as np

from rxn_ca.core.neighborhoods import Neighborhood
from rxn_ca.discrete.discrete_step_analyzer import DiscreteStepAnalyzer
from ..core import BasicStepArtist
from ..core import COLORS

from .phase_map import PhaseMap

class DiscreteStepArtist(BasicStepArtist):

    @classmethod
    def build_legend_from_phase_list(cls, phases):
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

    def __init__(self, phase_map: PhaseMap, legend = None):
        self.phase_map = phase_map
        self.legend = legend
        super().__init__()

    def set_legend(self, new_leg):
        self.legend = new_leg

    def get_color_by_cell_state(self, cell_state):
        if int(cell_state) == Neighborhood.PADDING_VAL:
            return (0,0,0)
        phase_name = self.phase_map.get_state_name(cell_state)
        return self.legend[phase_name]

    def get_legend(self, state):
        if self.legend is None:
            analyzer = DiscreteStepAnalyzer(self.phase_map)
            phases = analyzer.phases_present(BasicSimulationStep(state))
            return DiscreteStepArtist.build_legend_from_phase_list(phases)
        else:
            return self.legend

    def _draw_image(self, state: np.array, **kwargs):
        self.legend = self.get_legend(state)
        return super()._draw_image(state, **kwargs)


