import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from rxn_ca.phase_map import PhaseMap

from rxn_ca.scored_reaction_set import ScoredReactionSet
from rxn_ca.reaction_step import ReactionStep
from rxn_ca.step_analyzer import StepAnalyzer
from .artist import StepArtist
import numpy as np
from .colors import COLORS
import time
import typing

class ReactionResult():
    """A class that stores the result of running a simulation. Keeps track of all
    the steps that the simulation proceeded through, and the set of reactions that
    was used in the simulation.
    """

    @classmethod
    def from_dict(cls, res_dict):
        res = ReactionResult(
            ScoredReactionSet.from_dict(res_dict["rxn_set"]),
            PhaseMap.from_dict(res_dict["phase_map"])
        )
        for step_dict in res_dict["steps"]:
            res.add_step(ReactionStep.from_dict(step_dict))

        for choices in res_dict["choices"]:
            res.add_choices(choices)

        return res

    @classmethod
    def from_file(cls, fpath):
        with open(fpath, 'r') as f:
            return cls.from_dict(json.loads(f.read()))

    def __init__(self, rxn_set: ScoredReactionSet, phase_map: PhaseMap):
        """Initializes a ReactionResult with the reaction set used in the simulation

        Args:
            rxn_set (ScoredReactionSet):
        """
        self.steps: list[ReactionStep] = []
        self.mole_steps: list[np.array] = []
        self.choices: list[dict] = []
        self.rxn_set: ScoredReactionSet = rxn_set
        self.phase_map: PhaseMap = phase_map
        self.analyzer = StepAnalyzer(self.phase_map, self.rxn_set)

    def add_step(self, step: ReactionStep) -> None:
        """Adds a step to the reaction result. Using this method allows
        the accumulation of all the steps in a given simulation.

        Args:
            step (ReactionStep): _description_
        """
        self.steps.append(step)

    def add_mol_step(self, mol_step: np.array) -> None:
        self.mole_steps.append(mol_step)

    def add_choices(self, choices: dict) -> None:
        self.choices.append(choices)

    def get_choices_at(self, step_no: int, top: int = None, exclude_ids = True) -> None:

        data = self.choices[step_no]
        names = list(data.keys())
        values = list(data.values())
        zipped = list(zip(names, values))
        zipped.sort(key = lambda x: -x[1])

        if exclude_ids:
            filtered = list(filter(lambda item: not self.rxn_set.get_rxn_by_str(item[0]).is_identity, zipped))
        else:
            filtered = zipped

        if top is None:
            top = len(filtered)

        return filtered[0:top]

    def show_choices_at(self, step_no: int, top: int = None, exclude_ids = True) -> None:
        choices = self.get_choices_at(step_no, top, exclude_ids)
        for choice in choices:
            print(choice[0])
            print(f'Competitiveness: {self.rxn_set.get_rxn_by_str(choice[0]).competitiveness}')
            if self.rxn_set.get_rxn_by_str(choice[0]).original_rxn is not None:
                print(f'eV / atom: {self.rxn_set.get_rxn_by_str(choice[0]).original_rxn.energy_per_atom}')
            print(f'Count: {choice[1]}')
            print('--------------')

    def plot_choices_at(self, step_no: int, top: int = None, exclude_ids = True) -> None:
        choices = self.get_choices_at(step_no, top, exclude_ids)
        sorted_names = list(map(lambda x: x[1], choices[0:top]))
        sorted_values = list(map(lambda x: x[0], choices[0:top]))
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.bar(sorted_names, sorted_values)
        fig.suptitle(f'Chosen Reactions @ Step {step_no}')

        axs.set_xticklabels(sorted_names, rotation = 60)

        fig.show()

    @property
    def all_phases(self) -> list[str]:
        """A list of all the phases that appeared during this simulation. Note that
        this is distinct from the list of phases that _could_ appear according to the
        reaction set used during the simulation.

        Returns:
            list[str]:
        """

        phases: list[str] = []
        for step in self.steps:
            phases = phases + self.analyzer.phases_present(step)

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

    def jupyter_show_step(self, step_no: int, display_phases: typing.Dict[str, typing.Tuple[int, int ,int]] = None, cell_size = 20) -> None:
        """In a jupyter notebook environment, visualizes the step as a color coded phase grid.

        Args:
            step_no (int): The step of the simulation to visualize
            display_phases (typing.Dict[str, typing.Tuple[int, int ,int]], optional): Defaults to None.
        """

        if display_phases is None:
            display_phases = self.phase_color_map

        label = f'Step {step_no}'
        artist = StepArtist(self.phase_map, display_phases)
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

    def _get_images(self, display_phases = None, cell_size = 20):
        if display_phases is None:
            display_phases = self.phase_color_map

        artist = StepArtist(self.phase_map, display_phases)
        imgs = []
        for idx, step in enumerate(self.steps):
            label = f'Step {idx}'
            img = artist.get_img(step, label, cell_size)
            imgs.append(img)

        return imgs

    def plot_volume_fractions(self, min_prevalence=0.01) -> None:
        """In a Jupyter Notebook environment, plots the phase prevalence traces for the simulation.

        Returns:
            None:
        """

        fig = go.Figure()
        fig.update_layout(width=800, height=800)
        fig.update_yaxes(range=[-0.05,1.05], title="Volume Fraction")
        fig.update_xaxes(range=[0, len(self.steps) - 1], title="Simulation Step")

        analyzer = StepAnalyzer(self.phase_map, self.rxn_set)
        traces = []
        for phase in self.all_phases:
            if phase != self.phase_map.FREE_SPACE:
                xs = np.arange(len(self.steps))
                ys = [analyzer.volume_fraction(step, phase) for step in self.steps]
                traces.append((xs, ys, phase))

        filtered_traces = [t for t in traces if max(t[1]) > min_prevalence]

        for t in filtered_traces:
            fig.add_trace(go.Scatter(name=t[2], x=t[0], y=t[1], mode='lines'))

        fig.show()

    def plot_mole_fractions(self, min_prevalence=0.01) -> None:
        """In a Jupyter Notebook environment, plots the phase prevalence traces for the simulation.

        Returns:
            None:
        """

        fig = go.Figure()
        fig.update_layout(width=800, height=800)
        fig.update_yaxes(range=[-0.05,1.05], title="Mole Fraction")
        fig.update_xaxes(title="Simulation Step")

        analyzer = StepAnalyzer(self.phase_map, self.rxn_set)
        traces = []
        for phase in self.all_phases:
            if phase != self.phase_map.FREE_SPACE:
                xs = np.arange(len(self.steps))
                ys = [analyzer.mole_fraction(step, phase) for step in self.steps]
                traces.append((xs, ys, phase))

        filtered_traces = [t for t in traces if max(t[1]) > min_prevalence]

        for t in filtered_traces:
            fig.add_trace(go.Scatter(name=t[2], x=t[0], y=t[1], mode='lines'))

        fig.show()


    def plot_elemental_amounts(self) -> None:
        fig = go.Figure()
        fig.update_layout(width=800, height=800)
        fig.update_yaxes(title="Relative Prevalence")
        fig.update_xaxes(title="Simulation Step")

        analyzer = StepAnalyzer(self.phase_map, self.rxn_set)
        elements = list(analyzer.elemental_composition(self.steps[0]).keys())
        traces = []
        amounts = [analyzer.elemental_composition(s, self.mole_steps[idx]) for idx, s in enumerate(self.steps)]
        for el in elements:
            xs = np.arange(len(self.steps))
            ys = [a[el] for a in amounts]
            traces.append((xs, ys, el))

        # filtered_traces = [t for t in traces if max(t[1]) > min_prevalence]

        for t in traces:
            fig.add_trace(go.Scatter(name=t[2], x=t[0], y=t[1], mode='lines'))

        fig.show()

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
            "choices": self.choices,
            "rxn_set": self.rxn_set.to_dict(),
            "phase_map": self.phase_map.to_dict()
        }

    def to_file(self, fpath):
        with open(fpath, 'w+') as f:
            f.write(json.dumps(self.to_dict()))
