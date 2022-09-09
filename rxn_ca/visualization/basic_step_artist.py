from PIL import Image, ImageDraw

from rxn_ca.core import NeighborhoodView
from rxn_ca.core import BasicSimulationStep
import numpy as np
import io

import matplotlib.pyplot as plt

class BasicStepArtist():

    def get_legend(self, state: np.array):
        return {}

    def jupyter_show_state(self, state: np.array, **kwargs):
        img = self._draw_image(state, **kwargs)
        display(img)

    def jupyter_show_view(self, view: NeighborhoodView, **kwargs):
        img = self._draw_image(view.view_state, **kwargs)
        display(img)

    def get_img_state(self, state: np.array, **kwargs):
        return self._draw_image(state, **kwargs)

    def jupyter_show(self, simulation_step: BasicSimulationStep, **kwargs):
        img = self.get_img(simulation_step, **kwargs)
        display(img)

    def get_img(self, simulation_step: BasicSimulationStep, **kwargs):
        return self._draw_image(simulation_step.state, **kwargs)

    def _draw_image(self, state, **kwargs):
        dim = len(state.shape)
        if dim == 2:
            return self._draw_image_2D(state, **kwargs)
        elif dim == 3:
            return self._draw_image_3D(state, **kwargs)

    def _draw_image_2D(self, state: np.array, **kwargs):
        label = kwargs.get('label', None)
        cell_size = kwargs.get('cell_size', 20)

        legend = self.get_legend(state)
        state_size = state.shape[0]
        width = state_size + 6

        legend_border_width = 5
        height = max(state_size, len(legend) + 1)
        img = Image.new( 'RGB', (width * cell_size + legend_border_width, height * cell_size), "black") # Create a new black image
        pixels = img.load()
        draw = ImageDraw.Draw(img)

        for row in range(0, state_size):
            for col in range(0, state_size):

                cell_state = state[row, col]

                p_col_start = col * cell_size
                p_row_start = row * cell_size
                for p_x in range(p_col_start, p_col_start + cell_size):
                    for p_y in range(p_row_start, p_row_start + cell_size):
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

    def _draw_image_3D(self, state, **kwargs):
        shell_only = kwargs.get('shell_only', False)
        include_phases = kwargs.get('include_phases', None)

        legend = self.get_legend(state)
        shape = state.shape
        size = state.shape[0]
        dataset = {}
        if include_phases is None:
            include_phases = self.phase_map.phases

        dataset['empty'] = np.ones(shape)
        for phase in include_phases:
            phase_data = np.zeros(shape)
            for xloc in range(0, size):
                for yloc in range(0, size):
                    for zloc in range(0, size):
                        if not shell_only or (yloc == 0 or xloc == (size - 1) or zloc == (size - 1)):
                            if state[xloc][yloc][zloc] == self.phase_map.phase_to_int[phase]:
                                phase_data[xloc][yloc][zloc] = 1
                                dataset['empty'][xloc][yloc][zloc] = 0
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

