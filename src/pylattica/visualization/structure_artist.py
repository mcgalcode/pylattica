from abc import abstractmethod

from pylattica.core import SimulationState, PeriodicStructure
from .cell_artist import CellArtist


class StructureArtist:
    def __init__(self, structure: PeriodicStructure, cell_artist: CellArtist):
        self.structure = structure
        self.cell_artist = cell_artist

    def jupyter_show(self, state: SimulationState, **kwargs):
        from IPython.display import display  # pragma: no cover

        img = self.get_img(state, **kwargs)  # pragma: no cover
        display(img)  # pragma: no cover

    def get_img(self, state: SimulationState, **kwargs):
        return self._draw_image(state, **kwargs)

    def save_img(self, state: SimulationState, filename: str, **kwargs) -> None:
        """Saves the areaction result as an animated GIF.

        Args:
            filename (str): The name of the output GIF. Must end in .gif.
            color_map (_type_, optional): Defaults to None.
            cell_size (int, optional): The side length of a grid cell in pixels. Defaults to 20.
            wait (float, optional): The time in seconds between each frame. Defaults to 0.8.
        """
        img = self.get_img(state, **kwargs)
        img.save(filename)
        return filename

    @abstractmethod
    def _draw_image(self, state: SimulationState, **kwargs):
        pass  # pragma: no cover
