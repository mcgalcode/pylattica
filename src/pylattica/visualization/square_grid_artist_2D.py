from PIL import Image, ImageDraw

from ..core.constants import LOCATION, SITE_ID
from ..core.simulation_state import SimulationState

from .structure_artist import StructureArtist


class SquareGridArtist2D(StructureArtist):
    """A helper StructureArtist class for rendering 2D square grids."""

    def _draw_image(self, state: SimulationState, **kwargs):
        label = kwargs.get("label", None)
        cell_size = kwargs.get("cell_size", 20)

        show_legend = kwargs.get("show_legend", True)

        legend = self.cell_artist.get_legend(state)
        legend_order = sorted(legend.keys())
        state_size = int(self.structure.lattice.vec_lengths[0])

        if show_legend:
            width = state_size + 6
            legend_border_width = 5
            height = max(state_size, len(legend) + 1)
            img_width = width * cell_size + legend_border_width
            img_height = height * cell_size
        else:
            width = state_size
            height = state_size
            img_width = width * cell_size
            img_height = height * cell_size

        img = Image.new(
            "RGB",
            (img_width, img_height),
            "black",
        )  # Create a new black image

        pixels = img.load()
        draw = ImageDraw.Draw(img)

        for site in self.structure.sites():
            loc = site[LOCATION]
            cell_state = state.get_site_state(site[SITE_ID])
            cell_color = self.cell_artist.get_color_from_cell_state(cell_state)
            p_x_start = int((loc[0]) * cell_size)
            p_y_start = int((state_size - 1 - loc[1]) * cell_size)
            for p_x in range(p_x_start, p_x_start + cell_size):
                for p_y in range(p_y_start, p_y_start + cell_size):
                    pixels[p_x, p_y] = cell_color

        if show_legend:
            count = 0
            legend_hoffset = int(cell_size / 4)
            legend_voffset = int(cell_size / 4)

            for p_y in range(height * cell_size):
                for p_x in range(0, legend_border_width):
                    x = state_size * cell_size + p_x
                    pixels[x, p_y] = (255, 255, 255)

            for phase in legend_order:
                color = legend.get(phase)
                p_col_start = state_size * cell_size + legend_border_width + legend_hoffset
                p_row_start = count * cell_size + legend_voffset
                for p_x in range(p_col_start, p_col_start + cell_size):
                    for p_y in range(p_row_start, p_row_start + cell_size):
                        pixels[p_x, p_y] = color

                legend_label_loc = (
                    int(p_col_start + cell_size + cell_size / 4),
                    int(p_row_start + cell_size / 4),
                )
                draw.text(legend_label_loc, phase, (255, 255, 255))
                count += 1

        if label is not None:
            draw.text((5, 5), label, (255, 255, 255))

        return img
