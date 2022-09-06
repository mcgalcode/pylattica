import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from rxn_ca.core import BasicSimulationResult

from .phase_map import PhaseMap
from .discrete_step_analyzer import DiscreteStepAnalyzer
from .discrete_step_artist import DiscreteStepArtist

import numpy as np
from ..core import COLORS
import time
import typing
import multiprocessing as mp

from PIL import Image


_dsr_globals = {}

def color_map(phases):
    color_map: typing.Dict[str, typing.Tuple[int, int, int]] = {}
    c_idx: int = 0
    for p in phases:
        color_map[p] = COLORS[c_idx]
        c_idx += 1

    return color_map

class DiscreteStateResult(BasicSimulationResult):
    """A class that stores the result of running a simulation. Keeps track of all
    the steps that the simulation proceeded through, and the set of reactions that
    was used in the simulation.
    """

    def __init__(self, phase_map: PhaseMap):
        """Initializes a ReactionResult with the reaction set used in the simulation

        Args:
            rxn_set (ScoredReactionSet):
        """
        super().__init__()
        self.phase_map: PhaseMap = phase_map

    @property
    def all_phases(self) -> list[str]:
        """A list of all the phases that appeared during this simulation. Note that
        this is distinct from the list of phases that _could_ appear according to the
        reaction set used during the simulation.

        Returns:
            list[str]:
        """
        analyzer = DiscreteStepAnalyzer(self.phase_map)
        phases: list[str] = []
        for step in self.steps:
            phases = phases + analyzer.phases_present(step)

        return list(set(phases))

    @property
    def phase_color_map(self) -> typing.Dict[str, typing.Tuple[int, int ,int]]:
        """Returns a map of phases to colors that can be used to visualize the phases

        Returns:
            typing.Dict[str, typing.Tuple[int, int ,int]]: A mapping of phase name to RGB
            color values
        """
        return color_map(self.all_phases)

    def _get_images(self, **kwargs):
        color_map = kwargs.get('color_map', self.phase_color_map)
        if color_map is None:
            color_map = self.phase_color_map

        global _dsr_globals
        _dsr_globals['artist'] = DiscreteStepArtist(self.phase_map, color_map)
        imgs = []
        PROCESSES = mp.cpu_count()
        with mp.get_context('fork').Pool(PROCESSES) as pool:
            params = []
            for idx, step in enumerate(self.steps):
                label = f'Step {idx}'
                step_kwargs = { **kwargs, 'label': label }
                params.append([step, step_kwargs])

            for img in pool.starmap(get_img_parallel, params):
                imgs.append(img)

                # img = artist.get_img(step, label, cell_size)
                # imgs.append(img)

        return imgs

    def jupyter_show_step(self, step_no: int, color_map: typing.Dict[str, typing.Tuple[int, int ,int]] = None, cell_size = 20) -> None:
        """In a jupyter notebook environment, visualizes the step as a color coded phase grid.

        Args:
            step_no (int): The step of the simulation to visualize
            color_map (typing.Dict[str, typing.Tuple[int, int ,int]], optional): Defaults to None.
        """

        if color_map is None:
            color_map = self.phase_color_map

        label = f'Step {step_no}'
        artist = DiscreteStepArtist(self.phase_map, color_map)
        step = self.steps[step_no]
        artist.jupyter_show(step, label, cell_size=cell_size)

    def jupyter_play(self, color_map: typing.Dict[str, typing.Tuple[int, int ,int]] = None, cell_size: int = 20, wait: int = 1):
        """In a jupyter notebook environment, plays the simulation visualization back by showing a
        series of images with {wait} seconds between each one.

        Args:
            color_map (typing.Dict[str, typing.Tuple[int, int ,int]], optional): Defaults to None.
            cell_size (int, optional): The sidelength of a grid cell in pixels. Defaults to 20.
            wait (int, optional): The time duration between frames in the animation. Defaults to 1.
        """
        from IPython.display import clear_output

        imgs = self._get_images(color_map = color_map, cell_size = cell_size)
        for img in imgs:
            clear_output()
            display(img)
            time.sleep(wait)

    def to_gif(self, filename: str, **kwargs) -> None:
        """Saves the areaction result as an animated GIF.

        Args:
            filename (str): The name of the output GIF. Must end in .gif.
            color_map (_type_, optional): Defaults to None.
            cell_size (int, optional): The side length of a grid cell in pixels. Defaults to 20.
            wait (float, optional): The time in seconds between each frame. Defaults to 0.8.
        """
        wait = kwargs.get('wait', 0.8)
        imgs = self._get_images(**kwargs)
        for idx, img in enumerate(imgs):
            img.save(f'tmp_rxn_ca_step_{idx}.png')

        reloaded_imgs = []
        for idx in range(len(imgs)):
            fname = f'tmp_rxn_ca_step_{idx}.png'
            reloaded_imgs.append(Image.open(fname))
            os.remove(fname)

        reloaded_imgs[0].save(filename, save_all=True, append_images=reloaded_imgs[1:], duration=wait * 1000, loop=0)

    def plot_phase_fractions(self, min_prevalence=0.01) -> None:
        """In a Jupyter Notebook environment, plots the phase prevalence traces for the simulation.

        Returns:
            None:
        """

        fig = go.Figure()
        fig.update_layout(width=800, height=800)
        fig.update_yaxes(range=[-0.05,1.05], title="Volume Fraction")
        fig.update_xaxes(range=[0, len(self.steps) - 1], title="Simulation Step")

        analyzer = DiscreteStepAnalyzer(self.phase_map)
        traces = []
        for phase in self.all_phases:
            if phase != self.phase_map.FREE_SPACE:
                xs = np.arange(len(self.steps))
                ys = [analyzer.cell_fraction(step, phase) for step in self.steps]
                traces.append((xs, ys, phase))

        filtered_traces = [t for t in traces if max(t[1]) > min_prevalence]

        for t in filtered_traces:
            fig.add_trace(go.Scatter(name=t[2], x=t[0], y=t[1], mode='lines'))

        fig.show()

    def final_phase_fractions(self):
        analyzer = DiscreteStepAnalyzer(self.phase_map)
        fracs = {}
        for phase in self.all_phases:
            if phase != self.phase_map.FREE_SPACE:
                fracs[phase] = analyzer.cell_fraction(self.steps[-1], phase)

        return fracs

    def print_final_phase_fracs(self):
        for phase, frac in self.final_phase_fractions().items():
            print(f'{phase}: {frac}')

    def phase_fraction_at(self, step, phase):
        analyzer = DiscreteStepAnalyzer(self.phase_map)
        return analyzer.cell_fraction(self.steps[step - 1], phase)


    def plot_phase_counts(self):
        """In a jupyter notebook environment, plots the number of phases at each
        time step.
        """
        xs = np.arange(len(self.steps))
        ys = [step.phase_count for step in self.steps]
        plt.plot(xs, ys)

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "steps": [s.as_dict() for s in self.steps],
            "phase_map": self.phase_map.as_dict()
        }

def get_img_parallel(step, step_kwargs):
    return _dsr_globals['artist'].get_img(step, **step_kwargs)