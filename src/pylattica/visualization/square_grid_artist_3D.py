from .structure_artist import StructureArtist
from ..core import SimulationState
from ..core.periodic_structure import LOCATION, SITE_ID

import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image


class SquareGridArtist3D(StructureArtist):
    """A helper StructureArtist class for rendering 3D square grids."""

    def _draw_image(self, state: SimulationState, **kwargs):
        shell_only = kwargs.get("shell_only", False)

        size = round(state.size ** (1 / 3))

        shape = [size for _ in range(self.structure.dim)]
        dataset = {}

        dataset["empty"] = np.ones(shape)
        color_cache = {}

        for site in self.structure.sites():
            loc = site[LOCATION]
            if not shell_only or (loc[1] == 0 or loc[0] == size or loc[2] == size):
                site_id = site[SITE_ID]
                site_state = state.get_site_state(site_id)
                color = self.cell_artist.get_color_from_cell_state(site_state)
                color_str = str(color)
                if color_str not in color_cache:
                    color_cache[color_str] = color

                if color_str not in dataset:
                    dataset[color_str] = np.zeros(shape)

                shifted_loc = tuple(int(i) for i in loc)
                dataset[color_str][shifted_loc] = 1
                dataset["empty"][shifted_loc] = 0

        ax = plt.figure(figsize=(12, 12)).add_subplot(projection="3d")

        for color, data in dataset.items():
            if color == "empty":
                colors = [0.8, 0.8, 0.8, 0.2]
                ax.voxels(data, facecolors=colors, edgecolor="k", linewidth=0)
            else:
                colors = np.array(color_cache[color]) / 255
                ax.voxels(data, facecolors=colors, edgecolor="k", linewidth=0.25)

        ax.legend()
        plt.axis("off")
        fig = ax.get_figure()
        buf = io.BytesIO()
        fig.savefig(buf)
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return img
