import io
import math
import typing

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from ..core import COLORS
from ..core.constants import LOCATION, SITE_ID
from ..core.simulation_state import SimulationState
from ..discrete.discrete_step_analyzer import DiscreteStepAnalyzer
from ..discrete.state_constants import DISCRETE_OCCUPANCY
from ..square_grid.structure_builders import (
    SimpleSquare2DStructureBuilder,
    SimpleSquare3DStructureBuilder,
)


class SquareGridArtist():

    def __init__(self, legend = None):
        self.legend = legend

    def get_legend(self, state: SimulationState):
        return {}

    def jupyter_show_state(self, state: SimulationState, **kwargs):
        img = self._draw_image(state, **kwargs)
        display(img)

    def get_img_state(self, state: SimulationState, **kwargs):
        return self._draw_image(state, **kwargs)

    def jupyter_show(self, state: SimulationState, **kwargs):
        img = self.get_img(state, **kwargs)
        display(img)

    def get_img(self, state: SimulationState, **kwargs):
        return self._draw_image(state, **kwargs)

class DiscreteSquareGridArtist(SquareGridArtist):

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

    def __init__(self, legend = None):
        self.legend = legend

    def get_color_by_cell_state(self, cell_state):
        phase_name = cell_state[DISCRETE_OCCUPANCY]
        return self.get_legend()[phase_name]

    def get_legend(self, state):
        if self.legend is None:
            analyzer = DiscreteStepAnalyzer()
            phases = analyzer.phases_present(state)
            return DiscreteSquareGridArtist.build_legend_from_phase_list(phases)
        else:
            return self.legend

class DiscreteSquareGridArtist2D(DiscreteSquareGridArtist):

    def _draw_image(self, state: SimulationState, **kwargs):
        label = kwargs.get('label', None)
        cell_size = kwargs.get('cell_size', 20)
        
        size = int(math.sqrt(state.size))
        struct = SimpleSquare2DStructureBuilder().build(size)

        legend = self.get_legend(state)
        state_size = int(struct.bounds[0])
        width = state_size + 6

        legend_border_width = 5
        height = max(state_size, len(legend) + 1)
        img = Image.new('RGB', (width * cell_size + legend_border_width, height * cell_size), "black") # Create a new black image
        pixels = img.load()
        draw = ImageDraw.Draw(img)

        for site in struct.sites():
            loc = site[LOCATION]
            cell_state = state.get_site_state(site[SITE_ID])

            p_x_start = int((loc[0]) * cell_size)
            p_y_start = int((state_size - 1 - loc[1]) * cell_size)
            for p_x in range(p_x_start, p_x_start + cell_size):
                for p_y in range(p_y_start, p_y_start + cell_size):
                    pixels[p_x, p_y] = legend[cell_state[DISCRETE_OCCUPANCY]]

        count = 0
        legend_hoffset = int(cell_size / 4)
        legend_voffset = int(cell_size / 4)

        for p_y in range(height * cell_size):
            for p_x in range(0, legend_border_width):
                x = state_size * cell_size + p_x
                pixels[x, p_y] = (255,255,255)

        for phase, color in legend.items():
            p_col_start = state_size * cell_size + legend_border_width + legend_hoffset
            p_row_start = count * cell_size + legend_voffset
            for p_x in range(p_col_start, p_col_start + cell_size):
                for p_y in range(p_row_start, p_row_start + cell_size):
                    pixels[p_x, p_y] = color

            legend_label_loc = (int(p_col_start + cell_size + cell_size / 4), int(p_row_start + cell_size / 4))
            draw.text(legend_label_loc, phase, (255, 255, 255))
            count += 1

        if label is not None:
            draw.text((5,5),label,(255,255,255))

        return img

class DiscreteSquareGridArtist3D(DiscreteSquareGridArtist):

    def _draw_image(self, state: SimulationState, **kwargs):
        shell_only = kwargs.get('shell_only', False)
        include_phases = kwargs.get('include_phases', None)

        size = round(state.size ** (1/3))
        struct = SimpleSquare3DStructureBuilder().build(size)

        legend = self.get_legend(state)

        shape = [size for _ in range(struct.dim)]
        dataset = {}

        if include_phases is None:
            analyzer = DiscreteStepAnalyzer(struct)
            include_phases = analyzer.phases_present(state)

        dataset['empty'] = np.ones(shape)
        for phase in include_phases:
            phase_data = np.zeros(shape)
            for site in struct.sites():
                loc = site[LOCATION]
                if not shell_only or (loc[1] == 0 or loc[0] == size or loc[2] == size):
                    site_id = site[SITE_ID]
                    site_state = state.get_site_state(site_id)
                    if site_state[DISCRETE_OCCUPANCY] == phase:
                        shifted_loc = tuple(int(i) for i in loc)
                        phase_data[shifted_loc] = 1
                        dataset['empty'][shifted_loc] = 0
    
            if phase_data.sum() > 0:
                dataset[phase] = phase_data

        ax = plt.figure(figsize=(12,12)).add_subplot(projection='3d')

        for phase, data in dataset.items():
            if phase == 'empty':
                colors = [0.8, 0.8, 0.8, 0.2]
                ax.voxels(data, facecolors=colors, edgecolor='k', linewidth=0)
            else:
                colors = np.array(legend[phase]) / 255
                ax.voxels(data, facecolors=colors, edgecolor='k', linewidth=0.25)


        plt.axis('off')
        fig = ax.get_figure()
        buf = io.BytesIO()
        fig.savefig(buf)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return img