from ..discrete import DiscreteStepAnalyzer
from ..discrete import PhaseMap

from .scored_reaction_set import ScoredReactionSet

from pymatgen.core.composition import Composition

class ReactionStepAnalyzer(DiscreteStepAnalyzer):

    def __init__(self, phase_map: PhaseMap, reaction_set: ScoredReactionSet) -> None:
        self.rxn_set: ScoredReactionSet = reaction_set
        super().__init__(phase_map)

    def summary(self, step, phases = None):
        if phases is None:
            phases = self.phases_present(step)

        for p in phases:
            print(f'{p} moles: ', self.cell_count(step, p))

        denom = min([self.cell_count(step, p) for p in phases])
        for p in phases:
            print(f'mole ratio of {p}: ', self.cell_count(step, p) / denom)

        for el, amt in self.elemental_composition(step).items():
            print(f'{el} moles: ', amt)

    def elemental_composition(self, step, mole_amts = None):
        phases = self.phases_present(step)
        elemental_amounts = {}
        total = 0
        for p in phases:
            comp = Composition(p)
            moles = self.cell_count(step, p, mole_amts)
            for el, am in comp.as_dict().items():
                num_moles = moles * am
                if el in elemental_amounts:
                    elemental_amounts[el] += num_moles
                else:
                    elemental_amounts[el] = num_moles
                total += num_moles

        for el, am in elemental_amounts.items():
            elemental_amounts[el] = am / total


        return elemental_amounts