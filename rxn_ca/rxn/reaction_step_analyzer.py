from rxn_ca.core.basic_simulation_step import BasicSimulationStep
from rxn_ca.rxn.solid_phase_map import SolidPhaseMap
from ..discrete import DiscreteStepAnalyzer
from ..discrete import PhaseMap

from .scored_reaction_set import ScoredReactionSet

from pymatgen.core.composition import Composition

class ReactionStepAnalyzer(DiscreteStepAnalyzer):

    def __init__(self, phase_map: SolidPhaseMap, reaction_set: ScoredReactionSet) -> None:
        super().__init__(phase_map)
        self.rxn_set: ScoredReactionSet = reaction_set

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

    def get_reaction_choices(self, step: BasicSimulationStep):
        metadata = step.metadata
        rxns = {}
        for i in range(len(metadata)):
            for j in range(len(metadata[0])):
                rxn = metadata[i][j]
                rxn_str = str(rxn)
                if rxn_str in rxns:
                    rxns[rxn_str] += 1
                else:
                    rxns[rxn_str] = 0
        return rxns

    def elemental_composition(self, step):
        phases = self.phases_present(step)
        elemental_amounts = {}
        total = 0
        for p in phases:
            if p is not self.phase_map.FREE_SPACE:
                comp = Composition(p)
                moles = float(self.cell_count(step, p) * self.rxn_set.volumes[p])
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