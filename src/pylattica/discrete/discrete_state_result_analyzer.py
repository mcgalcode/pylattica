import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ..core import SimulationResult
from .discrete_step_analyzer import DiscreteStepAnalyzer

import numpy as np

class DiscreteResultAnalyzer():
    """A class that stores the result of running a simulation. Keeps track of all
    the steps that the simulation proceeded through, and the set of reactions that
    was used in the simulation.
    """

    def __init__(self, result: SimulationResult):
        self._result = result

    def all_phases(self):
        analyzer = DiscreteStepAnalyzer()
        phases = []
        for step in self._result.steps:
            curr_phases = analyzer.phases_present(step)
            phases = phases + curr_phases

        return list(set(phases))

    def _get_steps_to_plot(self):
        num_points = 10
        if len(self._result) > num_points:
            step_size = round(len(self._result) / num_points)
            step_idxs = list(range(1, len(self._result), step_size))
        else:
            step_idxs = list(range(1, len(self._result)))
        return step_idxs, [self._result.get_step(step_idx) for step_idx in step_idxs]

    def plot_phase_fractions(self, min_prevalence=0.01) -> None:
        """In a Jupyter Notebook environment, plots the phase prevalence traces for the simulation.

        Returns:
            None:
        """

        fig = go.Figure()
        fig.update_layout(width=800, height=800)
        fig.update_yaxes(range=[-0.05,1.05], title="Volume Fraction")

        analyzer = DiscreteStepAnalyzer()
        traces = []

        step_idxs, steps = self._get_steps_to_plot()
        fig.update_xaxes(range=[0, step_idxs[-1]], title="Simulation Step")

        for phase in self.all_phases():
            ys = [analyzer.cell_fraction(step, phase) for step in steps]
            traces.append((step_idxs, ys, phase))

        filtered_traces = [t for t in traces if max(t[1]) > min_prevalence]

        for t in filtered_traces:
            fig.add_trace(go.Scatter(name=t[2], x=t[0], y=t[1], mode='lines'))

        fig.show()

    def final_phase_fractions(self):
        analyzer = DiscreteStepAnalyzer()
        fracs = {}
        for phase in self.all_phases():
            fracs[phase] = analyzer.cell_fraction(self._result.last_step, phase)

        return fracs

    def print_final_phase_fracs(self):
        for phase, frac in self.final_phase_fractions().items():
            print(f'{phase}: {frac}')

    def phase_fraction_at(self, step, phase):
        analyzer = DiscreteStepAnalyzer()
        return analyzer.cell_fraction(self._result.get_step(step), phase)

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
            "phase_map": self.phase_set.as_dict()
        }