from PIL import Image, ImageDraw

from ..core.periodic_structure import PeriodicStructure
from ..core.simulation_state import SimulationState
import numpy as np
import io

import matplotlib.pyplot as plt
from ..discrete.discrete_step_analyzer import DiscreteStepAnalyzer

from ..discrete.phase_set import PhaseSet

class BasicGridArtist():

    def __init__(self, phase_set: PhaseSet, struct: PeriodicStructure, legend = None):
        self.phase_set = phase_set
        self.legend = legend
        self._struct = struct

    def get_legend(self, state: SimulationState):
        return {}

    def jupyter_show_state(self, state: SimulationState, **kwargs):
        img = self._draw_image(state, **kwargs)
        display(img)

    # def jupyter_show_view(self, view: NeighborhoodView, **kwargs):
    #     img = self._draw_image(view.view_state, **kwargs)
    #     display(img)

    def get_img_state(self, state: SimulationState, **kwargs):
        return self._draw_image(state, **kwargs)

    def jupyter_show(self, state: SimulationState, **kwargs):
        img = self.get_img(state, **kwargs)
        display(img)

    def get_img(self, state: SimulationState, **kwargs):
        return self._draw_image(state, **kwargs)

    def _draw_image(self, state: SimulationState, **kwargs):
        if self._struct.dim == 2:
            return self._draw_image_2D(state, **kwargs)
        else:
            return self._draw_image_3D(state, **kwargs)

    def _draw_image_2D(self, state: SimulationState, **kwargs):
        label = kwargs.get('label', None)
        cell_size = kwargs.get('cell_size', 20)

        legend = self.get_legend(state)
        state_size = self._struct.size
        width = state_size + 6

        legend_border_width = 5
        height = max(state_size, len(legend) + 1)
        img = Image.new('RGB', (width * cell_size + legend_border_width, height * cell_size), "black") # Create a new black image
        pixels = img.load()
        draw = ImageDraw.Draw(img)

        for site in self._struct.sites():
            loc = site['location']
            cell_state = state.get_site_state(site['id'])

            p_x_start = int((loc[0] - 0.5) * cell_size)
            p_y_start = int((loc[1] - 0.5) * cell_size)
            for p_x in range(p_x_start, p_x_start + cell_size):
                for p_y in range(p_y_start, p_y_start + cell_size):
                    pixels[p_x, p_y] = self.get_color_by_cell_state(cell_state)

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

    def _draw_image_3D(self, state: SimulationState, **kwargs):
        shell_only = kwargs.get('shell_only', False)
        include_phases = kwargs.get('include_phases', None)

        legend = self.get_legend(state)
        shape = [self._struct.size for _ in range(self._struct.dim)]
        size = self._struct.size
        dataset = {}

        if include_phases is None:
            include_phases = self.phase_set.phases

        magic_offset = 0.5

        dataset['empty'] = np.ones(shape)
        for phase in include_phases:
            phase_data = np.zeros(shape)
            for site in self._struct.sites():
                loc = site['location']
                if not shell_only or (loc[1] == magic_offset or loc[0] == (size - magic_offset) or loc[2] == (size - magic_offset)):
                    if state.get_site_state(site['id'])['_disc_occupancy'] == phase:
                        shifted_loc = tuple(int(i - magic_offset) for i in loc)
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

    def draw_step_3D(self, step, legend, shell_only = False, include_phases = None):
        self._draw_image_3D(step.state, legend, shell_only = shell_only, include_phases = include_phases)


import typing
import numpy as np

from pylatca.core import COLORS

class DiscreteSquareGridArtist(BasicGridArtist):

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

    def __init__(self, phase_set: PhaseSet, struct: PeriodicStructure, legend = None):
        self.phase_set = phase_set
        self.legend = legend
        self._struct = struct

    def set_legend(self, new_leg):
        self.legend = new_leg

    def get_color_by_cell_state(self, cell_state):
        phase_name = cell_state['_disc_occupancy']
        return self.legend[phase_name]

    def get_legend(self, state):
        if self.legend is None:
            analyzer = DiscreteStepAnalyzer()
            phases = analyzer.phases_present(state)
            return DiscreteSquareGridArtist.build_legend_from_phase_list(phases)
        else:
            return self.legend

    def _draw_image(self, state: SimulationState, **kwargs):
        self.legend = self.get_legend(state)
        return super()._draw_image(state, **kwargs)