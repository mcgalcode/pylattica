from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from functools import lru_cache

from ..core import SimulationResult, SimulationState
from .discrete_step_analyzer import DiscreteStepAnalyzer


class DiscreteResultAnalyzer:
    """A class that stores the result of running a simulation. Keeps track of all
    the steps that the simulation proceeded through, and the set of reactions that
    was used in the simulation.
    """

    def __init__(self, result: SimulationResult):
        """Instantiates the DiscreteResultAnalyzer

        Parameters
        ----------
        result : SimulationResult
            The result to analyze.
        """
        self._result = result

    @lru_cache
    def all_phases(self) -> List[str]:
        """Returns all of the phases present across the analyzed result.

        Returns
        -------
        List[str]
            A list of every phase present in any step of the result.
        """
        analyzer = DiscreteStepAnalyzer()
        phases = []
        for step in self._result.steps():
            curr_phases = analyzer.phases_present(step)
            phases = phases + curr_phases

        return frozenset(phases)

    def plot_phase_fractions(self, min_prevalence=0.01) -> None:
        """In a Jupyter Notebook environment, plots the phase prevalence traces for the simulation.

        Returns:
            None:
        """

        fig = go.Figure()
        fig.update_layout(width=800, height=800)
        fig.update_yaxes(range=[-0.05, 1.05], title="Volume Fraction")

        analyzer = DiscreteStepAnalyzer()
        traces = []

        step_idxs, steps = self._get_steps_to_plot()
        fig.update_xaxes(range=[0, step_idxs[-1]], title="Simulation Step")

        for phase in self.all_phases():
            ys = [analyzer.cell_fraction(step, phase) for step in steps]
            traces.append((step_idxs, ys, phase))

        filtered_traces = [t for t in traces if max(t[1]) > min_prevalence]

        for t in filtered_traces:
            fig.add_trace(go.Scatter(name=t[2], x=t[0], y=t[1], mode="lines"))

        return fig

    def final_phase_fractions(self) -> Dict[str, float]:
        """Returns the fractions of each phase in the last frame of the simulation.

        Returns
        -------
        Dict[str, float]
        """
        analyzer = DiscreteStepAnalyzer()
        fracs = {}
        for phase in self.all_phases():
            fracs[phase] = analyzer.cell_fraction(self._result.last_step, phase)

        return fracs

    def phase_fraction_at(self, step: int, phase: str) -> float:
        """Returns the fraction of the specified phase at the specified simulation frame.

        Parameters
        ----------
        step : int
            The simulation frame for which this analysis should be done.
        phase : str
            The phase for which the phase fraction should be calculated

        Returns
        -------
        float
            The phase fraction of the specified phase at the specified frame.
        """
        analyzer = DiscreteStepAnalyzer()
        return analyzer.cell_fraction(self._result.get_step(step), phase)

    def plot_phase_counts(self) -> None:  # pragma: no cover
        """In a jupyter notebook environment, plots the number of phases at each
        time step.
        """
        xs = np.arange(len(self.steps))
        ys = [step.phase_count for step in self.steps]
        plt.plot(xs, ys)

    def _get_steps_to_plot(self) -> Tuple[List[int], List[SimulationState]]:
        num_points = min(100, len(self._result))
        step_size = max(1, round(len(self._result) / num_points))
        self._result.load_steps(step_size)
        step_idxs = list(range(0, len(self._result), step_size))
        return step_idxs, [self._result.get_step(step_idx) for step_idx in step_idxs]
