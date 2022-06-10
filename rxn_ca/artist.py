from PIL import Image, ImageDraw

from reaction_model.phase_map import PhaseMap
from .colors import COLORS

class StepArtist():

    def __init__(self, phase_map: PhaseMap, display_phases = None):
        self.display_phases = display_phases
        self.phase_map = phase_map

    def show(self, reaction_step, label = None, cell_size = 20):
        img = self._draw_image(reaction_step, label, cell_size)
        img.show()

    def jupyter_show(self, reaction_step, label = None, cell_size = 20):
        img = self.get_img(reaction_step, label, cell_size)
        display(img)

    def get_img(self, reaction_step, label = None, cell_size = 20):
        return self._draw_image(reaction_step, label, cell_size)

    def _draw_image(self, reaction_step, label=None, cell_size = 20):

        if self.display_phases is None:
            display_phases = {}
        else:
            display_phases = self.display_phases
        color_ct = 0

        phase_map = self.phase_map.step_as_phase_name_array_no_padding(reaction_step)
        width = reaction_step.size + 6
        height = max(reaction_step.size, len(display_phases) + 1)
        img = Image.new( 'RGB', (width * cell_size, height * cell_size), "black") # Create a new black image
        pixels = img.load()
        draw = ImageDraw.Draw(img)
        for row in range(0, reaction_step.size):
            for col in range(0, reaction_step.size):
                phase_name = str(phase_map[row, col])
                if phase_name not in display_phases:
                    display_phases[phase_name] = COLORS[color_ct]
                    color_ct += 1

                p_col_start = col * cell_size
                p_row_start = row * cell_size
                for p_x in range(p_col_start, p_col_start + cell_size):
                    for p_y in range(p_row_start, p_row_start + cell_size):
                        pixels[p_x, p_y] = display_phases[phase_name]

        count = 0
        legend_hoffset = int(cell_size / 4)
        legend_voffset = int(cell_size / 4)

        for phase, color in display_phases.items():
            p_col_start = reaction_step.size * cell_size + legend_hoffset
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
