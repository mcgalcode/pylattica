from PIL import Image, ImageDraw
from .basic_simulation_step import BasicSimulationStep

class BasicStepArtist():

    def show(self, simulation_step: BasicSimulationStep, label = None, cell_size = 20):
        img = self._draw_image(simulation_step, label, cell_size)
        img.show()

    def jupyter_show(self, simulation_step: BasicSimulationStep, label = None, cell_size = 20):
        img = self.get_img(simulation_step, label, cell_size)
        display(img)

    def get_img(self, simulation_step: BasicSimulationStep, label = None, cell_size = 20):
        return self._draw_image(simulation_step, label, cell_size)

    def get_legend(self):
        return {}

    def _draw_image(self, simulation_step: BasicSimulationStep, label=None, cell_size = 20):

        legend = self.get_legend()

        width = simulation_step.size + 6
        height = max(simulation_step.size, len(legend) + 1)
        img = Image.new( 'RGB', (width * cell_size, height * cell_size), "black") # Create a new black image
        pixels = img.load()
        draw = ImageDraw.Draw(img)

        for row in range(0, simulation_step.size):
            for col in range(0, simulation_step.size):

                cell_state = simulation_step.state[row, col]

                p_col_start = col * cell_size
                p_row_start = row * cell_size
                for p_x in range(p_col_start, p_col_start + cell_size):
                    for p_y in range(p_row_start, p_row_start + cell_size):
                        pixels[p_x, p_y] = self.get_color_by_cell_state(cell_state)

        count = 0
        legend_hoffset = int(cell_size / 4)
        legend_voffset = int(cell_size / 4)

        for phase, color in legend.items():
            p_col_start = simulation_step.size * cell_size + legend_hoffset
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
