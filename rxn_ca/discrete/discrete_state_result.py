import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from rxn_ca.core import BasicSimulationResult

from .phase_map import PhaseMap
from .discrete_step_analyzer import DiscreteStepAnalyzer
from .discrete_step_artist import DiscreteStepArtist

import numpy as np
from ..core import COLORS
import time
import typing

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
        display_phases: typing.Dict[str, typing.Tuple[int, int, int]] = {}
        c_idx: int = 0
        for p in self.all_phases:
            display_phases[p] = COLORS[c_idx]
            c_idx += 1

        return display_phases

    def _get_images(self, display_phases = None, cell_size = 20):
        if display_phases is None:
            display_phases = self.phase_color_map

        artist = DiscreteStepArtist(self.phase_map, display_phases)
        imgs = []
        for idx, step in tqdm(enumerate(self.steps), total = len(self.steps)):
            label = f'Step {idx}'
            img = artist.get_img(step, label, cell_size)
            imgs.append(img)

        return imgs

    def jupyter_show_step(self, step_no: int, display_phases: typing.Dict[str, typing.Tuple[int, int ,int]] = None, cell_size = 20) -> None:
        """In a jupyter notebook environment, visualizes the step as a color coded phase grid.

        Args:
            step_no (int): The step of the simulation to visualize
            display_phases (typing.Dict[str, typing.Tuple[int, int ,int]], optional): Defaults to None.
        """

        if display_phases is None:
            display_phases = self.phase_color_map

        label = f'Step {step_no}'
        artist = DiscreteStepArtist(self.phase_map, display_phases)
        step = self.steps[step_no]
        artist.jupyter_show(step, label, cell_size=cell_size)

    def jupyter_play(self, display_phases: typing.Dict[str, typing.Tuple[int, int ,int]] = None, cell_size: int = 20, wait: int = 1):
        """In a jupyter notebook environment, plays the simulation visualization back by showing a
        series of images with {wait} seconds between each one.

        Args:
            display_phases (typing.Dict[str, typing.Tuple[int, int ,int]], optional): Defaults to None.
            cell_size (int, optional): The sidelength of a grid cell in pixels. Defaults to 20.
            wait (int, optional): The time duration between frames in the animation. Defaults to 1.
        """
        from IPython.display import clear_output

        imgs = self._get_images(display_phases, cell_size)
        for img in imgs:
            clear_output()
            display(img)
            time.sleep(wait)

    def to_gif(self, filename: str, display_phases = None, cell_size: int = 20, wait: float = 0.8) -> None:
        """Saves the areaction result as an animated GIF.

        Args:
            filename (str): The name of the output GIF. Must end in .gif.
            display_phases (_type_, optional): Defaults to None.
            cell_size (int, optional): The side length of a grid cell in pixels. Defaults to 20.
            wait (float, optional): The time in seconds between each frame. Defaults to 0.8.
        """

        imgs = self._get_images(display_phases, cell_size)
        imgs[0].save(filename, save_all=True, append_images=imgs[1:], duration=wait * 1000, loop=0)

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

    def to_dict(self):
        return {
            "steps": [s.to_dict() for s in self.steps],
            "phase_map": self.phase_map.to_dict()
        }

