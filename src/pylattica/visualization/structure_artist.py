from abc import abstractmethod

from ..core import SimulationState, PeriodicStructure
from .cell_artist import CellArtist


class StructureArtist:
    """A parent class for specifying strategies for visualizing structures."""

    def __init__(self, structure: PeriodicStructure, cell_artist: CellArtist):
        """Instantiates the StructureArtist with a structure and the artist used to
        render it's sites.

        Parameters
        ----------
        structure : PeriodicStructure
            The structure to render.
        cell_artist : CellArtist
            The artist for rendering each of the structure's sites.
        """
        self.structure = structure
        self.cell_artist = cell_artist

    def jupyter_show(self, state: SimulationState, **kwargs):
        """Shows the rendering of the provided state in a jupyter notebook environment.

        Parameters
        ----------
        state : SimulationState
            The simulation state to display.
        """
        from IPython.display import display  # pragma: no cover

        img = self.get_img(state, **kwargs)  # pragma: no cover
        display(img)  # pragma: no cover

    def get_img(self, state: SimulationState, **kwargs):
        """Returns a PIL Image object representing the rendered state

        Parameters
        ----------
        state : SimulationState
            The simulation state to render.

        Returns
        -------
        PIL.Image
            The rendered simulation state image.
        """
        return self._draw_image(state, **kwargs)

    def save_img(self, state: SimulationState, filename: str, **kwargs) -> None:
        """Saves the rendering of the provided simulation state to a file.

        Parameters
        ----------
        state : SimulationState
            The state to render.
        filename : str
            The name of the file to store the rendering.

        Returns
        -------
        None
        """
        img = self.get_img(state, **kwargs)
        img.save(filename)
        return filename

    @abstractmethod
    def _draw_image(self, state: SimulationState, **kwargs):
        pass  # pragma: no cover
