import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ..discrete import DiscreteStateResult
from ..discrete import PhaseMap
from .reaction_step_analyzer import ReactionStepAnalyzer

from .scored_reaction_set import ScoredReactionSet
from .reaction_step import ReactionStep

import numpy as np
import time
import typing

class ReactionResult(DiscreteStateResult):
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

    def __init__(self, rxn_set: ScoredReactionSet, phase_map: PhaseMap):
        """Initializes a ReactionResult with the reaction set used in the simulation

        Args:
            rxn_set (ScoredReactionSet):
        """
        self.steps: list[ReactionStep] = []
        self.mole_steps: list[np.array] = []
        self.rxn_set: ScoredReactionSet = rxn_set
        self.phase_map: PhaseMap = phase_map
        self.analyzer = ReactionStepAnalyzer(self.phase_map, self.rxn_set)

    def get_choices_at(self, step_no: int, top: int = None, exclude_ids = True) -> None:

        data = super().get_metadata_at(step_no)
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


    def jupyter_play(self, color_map: typing.Dict[str, typing.Tuple[int, int ,int]] = None, cell_size: int = 20, wait: int = 1):
        """In a jupyter notebook environment, plays the simulation visualization back by showing a
        series of images with {wait} seconds between each one.

        Args:
            color_map (typing.Dict[str, typing.Tuple[int, int ,int]], optional): Defaults to None.
            cell_size (int, optional): The sidelength of a grid cell in pixels. Defaults to 20.
            wait (int, optional): The time duration between frames in the animation. Defaults to 1.
        """
        from IPython.display import clear_output

        imgs = self._get_images(color_map, cell_size)
        for img in imgs:
            clear_output()
            display(img)
            time.sleep(wait)

    def plot_volume_fractions(self, min_prevalence=0.01) -> None:
        """In a Jupyter Notebook environment, plots the phase prevalence traces for the simulation.

        Returns:
            None:
        """

        fig = go.Figure()
        fig.update_layout(width=800, height=800)
        fig.update_yaxes(range=[-0.05,1.05], title="Volume Fraction")
        fig.update_xaxes(range=[0, len(self.steps) - 1], title="Simulation Step")

        traces = []
        for phase in self.all_phases:
            if phase != self.phase_map.FREE_SPACE:
                xs = np.arange(len(self.steps))
                ys = [self.analyzer.volume_fraction(step, phase) for step in self.steps]
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

        traces = []
        for phase in self.all_phases:
            if phase != self.phase_map.FREE_SPACE:
                xs = np.arange(len(self.steps))
                ys = [self.analyzer.mole_fraction(step, phase) for step in self.steps]
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


    def to_dict(self):
        return {
            "steps": [s.to_dict() for s in self.steps],
            "choices": self.choices,
            "rxn_set": self.rxn_set.to_dict(),
            "phase_map": self.phase_map.to_dict()
        }