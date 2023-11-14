import multiprocessing as mp
import os
import time

from PIL import Image

_dsr_globals = {}


class ResultArtist:
    """A class that stores the result of running a simulation. Keeps track of all
    the steps that the simulation proceeded through, and the set of reactions that
    was used in the simulation.
    """

    def __init__(self, step_artist, result):
        self._step_artist = step_artist
        self.result = result

    def _get_images(self, **kwargs):
        draw_freq = kwargs.get("draw_freq", 1)
        indices = list(range(0, len(self.result), draw_freq))

        global _dsr_globals  # pylint: disable=global-variable-not-assigned
        _dsr_globals["artist"] = self._step_artist
        imgs = []

        PROCESSES = mp.cpu_count()

        with mp.get_context("fork").Pool(PROCESSES) as pool:
            params = []
            for idx in indices:
                label = f"Step {idx}"
                step_kwargs = {**kwargs, "label": label}
                step = self.result.get_step(idx)
                params.append([step, step_kwargs])

            for img in pool.starmap(get_img_parallel, params):
                imgs.append(img)

        return imgs

    def jupyter_show_step(
        self,
        step_no: int,
        cell_size=20,
    ) -> None:
        """In a jupyter notebook environment, visualizes the step as a color coded phase grid.

        Args:
            step_no (int): The step of the simulation to visualize
        """
        label = f"Step {step_no}"  # pragma: no cover
        step = self.result.get_step(step_no)  # pragma: no cover
        self._step_artist.jupyter_show(
            step, label=label, cell_size=cell_size
        )  # pragma: no cover

    def jupyter_play(
        self,
        cell_size: int = 20,
        wait: int = 1,
    ):
        """In a jupyter notebook environment, plays the simulation visualization back by showing a
        series of images with {wait} seconds between each one.

        Args:
            cell_size (int, optional): The sidelength of a grid cell in pixels. Defaults to 20.
            wait (int, optional): The time duration between frames in the animation. Defaults to 1.
        """
        from IPython.display import clear_output, display  # pragma: no cover

        imgs = self._get_images(cell_size=cell_size)  # pragma: no cover
        for img in imgs:  # pragma: no cover
            clear_output()  # pragma: no cover
            display(img)  # pragma: no cover
            time.sleep(wait)  # pragma: no cover

    def to_gif(self, filename: str, **kwargs) -> None:
        """Saves the areaction result as an animated GIF.

        Args:
            filename (str): The name of the output GIF. Must end in .gif.
            cell_size (int, optional): The side length of a grid cell in pixels. Defaults to 20.
            wait (float, optional): The time in seconds between each frame. Defaults to 0.8.
        """
        wait = kwargs.get("wait", 0.8)
        imgs = self._get_images(**kwargs)
        for idx, img in enumerate(imgs):
            img.save(f"tmp_rxn_ca_step_{idx}.png")

        reloaded_imgs = []
        for idx in range(len(imgs)):
            fname = f"tmp_rxn_ca_step_{idx}.png"
            reloaded_imgs.append(Image.open(fname))
            os.remove(fname)

        reloaded_imgs[0].save(
            filename,
            save_all=True,
            append_images=reloaded_imgs[1:],
            duration=wait * 1000,
            loop=0,
        )


def get_img_parallel(step, step_kwargs):
    return _dsr_globals["artist"].get_img(step, **step_kwargs)
