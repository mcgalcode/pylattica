import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ..discrete import DiscreteStateResult
from ..discrete import PhaseMap
from .reaction_step_analyzer import ReactionStepAnalyzer

from .scored_reaction_set import ScoredReactionSet
from .reaction_step import ReactionStep

import numpy as np

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
        super().__init__(phase_map)
        self.steps: list[ReactionStep] = []
        self.mole_steps: list[np.array] = []
        self.rxn_set: ScoredReactionSet = rxn_set
        self.analyzer = ReactionStepAnalyzer(self.state_map, self.rxn_set)

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

    def plot_elemental_amounts(self) -> None:
        fig = go.Figure()
        fig.update_layout(width=800, height=800)
        fig.update_yaxes(title="Relative Prevalence")
        fig.update_xaxes(title="Simulation Step")

        analyzer = ReactionStepAnalyzer(self.state_map, self.rxn_set)
        elements = list(analyzer.elemental_composition(self.steps[0]).keys())
        traces = []
        amounts = [analyzer.elemental_composition(s) for idx, s in enumerate(self.steps)]
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
            "phase_map": self.state_map.to_dict()
        }