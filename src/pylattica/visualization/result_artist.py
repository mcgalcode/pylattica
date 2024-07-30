import multiprocessing as mp
import os
import time
import sys

from .structure_artist import StructureArtist
from ..core import SimulationResult

from PIL import Image

from typing import Callable

_dsr_globals = {}


def default_annotation_builder(step, step_no):
    return f"Step {step_no}"


class ResultArtist:
    """A class for rendering simulation results as animated GIFs."""

    def __init__(
        self,
        step_artist: StructureArtist,
        result: SimulationResult,
        annotation_builder: Callable = default_annotation_builder,
    ):
        """Instantiates the ResultArtist class.

        Parameters
        ----------
        step_artist : StructureArtist
            The artist that should be used to render each step of the simulation.
        result : SimulationResult
            The result to render.
        """
        self._step_artist = step_artist
        self.result = result
        self.annotation_builder = annotation_builder

    def _get_images(self, **kwargs):
        draw_freq = kwargs.get("draw_freq", 1)
        indices = list(range(0, len(self.result), draw_freq))
        imgs = []

        if sys.platform.startswith("win"):
            for idx in indices:
                label = f"Step {idx}"
                step_kwargs = {**kwargs, "label": label}
                step = self.result.get_step(idx)
                img = self._step_artist.get_img(step, **step_kwargs)
                imgs.append(img)
        else:
            PROCESSES = mp.cpu_count()
            global _dsr_globals  # pylint: disable=global-variable-not-assigned
            _dsr_globals["artist"] = self._step_artist

            with mp.get_context("fork").Pool(PROCESSES) as pool:
                params = []
                for idx in indices:
                    step = self.result.get_step(idx)
                    label = self.annotation_builder(step, idx)
                    step_kwargs = {**kwargs, "label": label}

                    params.append([step, step_kwargs])

                for img in pool.starmap(_get_img_parallel, params):
                    imgs.append(img)

        return imgs

    def jupyter_show_step(self, step_no: int, cell_size=20, **kwargs) -> None:
        """In a jupyter notebook environment, visualizes the step as a color coded phase grid.

        Parameters
        ----------
        step_no : int
            The step of the simulation to visualize
        cell_size : int, optional
            The size of each simulation cell, in pixels, by default 20
        """
        step = self.result.get_step(step_no)  # pragma: no cover
        label = self.annotation_builder(step, step_no)
        self._step_artist.jupyter_show(
            step, label=label, cell_size=cell_size, **kwargs
        )  # pragma: no cover

    def jupyter_play(self, cell_size: int = 20, wait: int = 1, **kwargs):
        """In a jupyter notebook environment, plays the simulation visualization back by showing a
        series of images with {wait} seconds between each one.

        Parameters
        ----------
        cell_size : int, optional
            The sidelength of a grid cell in pixels. Defaults to 20., by default 20
        wait : int, optional
            The time duration between frames in the animation. Defaults to 1., by default 1
        """
        from IPython.display import clear_output, display  # pragma: no cover

        imgs = self._get_images(cell_size=cell_size, **kwargs)  # pragma: no cover
        for img in imgs:  # pragma: no cover
            clear_output()  # pragma: no cover
            display(img)  # pragma: no cover
            time.sleep(wait)  # pragma: no cover

    def to_gif(self, filename: str, **kwargs) -> None:
        """Saves the simulation result result as an animated GIF.

        Parameters
        ----------
        filename : str
            The filename for the resulting file.
        """
        wait = kwargs.get("wait", 0.8)
        imgs = self._get_images(**kwargs)
        img_names = []
        for idx, img in enumerate(imgs):
            fname = f"tmp_pylat_step_{idx}.png"
            img.save(fname)
            img_names.append(fname)

        reloaded_imgs = []
        for fname in img_names:
            reloaded_imgs.append(Image.open(fname))

        reloaded_imgs[0].save(
            filename,
            save_all=True,
            append_images=reloaded_imgs[1:],
            duration=wait * 1000,
            loop=0,
        )

        for fname in img_names:
            os.remove(fname)


def _get_img_parallel(step, step_kwargs):
    return _dsr_globals["artist"].get_img(step, **step_kwargs)
