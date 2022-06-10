import math
import random
import numpy as np
import typing

from rxn_ca.distance_map import DistanceMap
from rxn_ca.reaction_step import ReactionStep, get_filter_size_from_side_length
from .normalizers import normalize

from rxn_ca.scored_reaction_set import ScoredReactionSet
from rxn_ca.scored_reaction import ScoredReaction
from rxn_ca.phase_map import PhaseMap

from pymatgen.core.composition import Composition

class StepAnalyzer():

    def __init__(self, phase_map: PhaseMap, reaction_set: ScoredReactionSet) -> None:
        self.phase_map: PhaseMap = phase_map
        self.rxn_set: ScoredReactionSet = reaction_set


    def summary(self, step, phases = None):
        if phases is None:
            phases = self.phases_present(step)

        for p in phases:
            print(f'{p} moles: ', self.moles_of(step, p))

        denom = min([self.moles_of(step, p) for p in phases])
        for p in phases:
            print(f'mole ratio of {p}: ', self.moles_of(step, p) / denom)

        for el, amt in self.elemental_composition(step).items():
            print(f'{el} moles: ', amt)

    def volume_fraction(self, step, phase_name: str):
        phase_count = self.cell_count(step, phase_name)
        total_occupied_cells = step.size ** 2
        return phase_count / total_occupied_cells

    def cell_count(self, step, phase_name: str):
        phase_int = self.phase_map.phase_to_int[phase_name]
        filtered = np.count_nonzero(step.state == phase_int)
        return filtered

    def moles_of(self, step, phase_name: str, mole_amts = None):
        if mole_amts is None:
            mole_amts = self.to_mole_array(step)
        phase_int = self.phase_map.phase_to_int[phase_name]
        return np.sum(mole_amts[step.state == phase_int])

    def mole_fraction(self, step, phase_name: str, mole_amts = None):
        if mole_amts is None:
            mole_amts = self.to_mole_array(step)

        phases = self.phases_present(step)
        moles = 0
        for p in phases:
            moles += self.moles_of(step, p, mole_amts)

        return self.moles_of(step, phase_name, mole_amts) / moles

    def mole_ratio(self, step, p1, p2, mole_amts = None):
        if mole_amts is None:
            mole_amts = self.to_mole_array(step)

        return self.moles_of(step, p1, mole_amts) / self.moles_of(step, p2, mole_amts)

    def to_mole_array(self, step: ReactionStep):
        mole_amounts = np.zeros(step.shape)
        pn_array = self.phase_map.as_phase_name_array(step.state)
        for i in range(step.size):
            for j in range(step.size):
                # assume each grid cell is one cc
                # volume / volume / moles
                mole_amounts[i][j] = 1 # / self.rxn_set.volumes[pn_array[i][j]]
        return mole_amounts

    def neighbors_in_padded_state(self, padded_state, i, j, filter_size):
        radius = math.floor(filter_size / 2)
        neighbor_phases = []
        for n_i in range(-1, 2):
            for n_j in range(-1, 2):
                if np.abs(n_i) + np.abs(n_j) == 1:
                    phase_int = int(padded_state[(i + n_i + radius, j + radius + n_j)])
                    neighbor_phases.append(self.phase_map.int_to_phase[phase_int])

        return neighbor_phases

    def species_at(self, padded_state, i, j, filter_size):
        radius = math.floor(filter_size / 2)
        return self.phase_map.int_to_phase[padded_state[i + radius, j + radius]]

    def pad_state(self, step, filter_size: int = None):
        if filter_size is None:
            filter_size: int = get_filter_size_from_side_length(step.size)
        padding: int = int((filter_size - 1) / 2)
        return np.pad(step.state, padding, 'constant', constant_values=self.phase_map.free_space_id)

    def subcell_in_step(self, step, i, j, filter_size):
        padded_state = self.pad_state(step, filter_size)
        return self.subcell_in_padded_state(padded_state, i, j, filter_size)

    def subcell_in_padded_state(self, padded_state, i, j, filter_size):
        # We are accepting coordinates that don't consider padding,
        # So modify them to make 0,0 the first non-padding entry
        radius = math.floor(filter_size / 2)
        i = i + radius
        j = j + radius

        curr_up = i - radius
        curr_down = i + radius + 1
        curr_left = j - radius
        curr_right = j + radius + 1

        return padded_state[curr_up:curr_down, curr_left:curr_right]

    def get_rxns_from_step(self, step: ReactionStep, i, j, filter_size = None, distance_map = None):
        if filter_size is None:
            filter_size = get_filter_size_from_side_length(step.size)

        padded_state = self.pad_state(step, filter_size)
        return self.get_possible_reactions_at(padded_state, i, j, filter_size, distance_map)

    def get_rxns_from_padded_state(self, padded_state: np.array, i, j, filter_size, distance_map = None):
        if distance_map is None:
            distance_map = DistanceMap(filter_size)

        subcell = self.subcell_in_padded_state(padded_state, i, j, filter_size)
        cell_center = np.array([int(subcell.shape[0] / 2), int(subcell.shape[1] / 2)])
        center_phase = self.phase_map.int_to_phase[subcell[tuple(cell_center)]]

        neighbor_phases = self.neighbors_in_padded_state(padded_state, i, j, filter_size)

        # Accumulate possible reactions here - this is a list of tuples: (reaction, likelihood)
        possible_reactions = []
        for i in range(subcell.shape[0]):
            for j in range(subcell.shape[1]):
                if not (i == cell_center[0] and j == cell_center[1]):
                    distance = distance_map.distances[(i, j)]

                    r1 = self.phase_map.int_to_phase[subcell[(i, j)]]
                    possible_rxn = self.get_rxn_and_score([r1, center_phase], distance, neighbor_phases, center_phase)
                    possible_reactions.append(possible_rxn)

        for specie in self.rxn_set.open_species:
            possible_rxn = self.get_rxn_and_score([center_phase, specie], 5, neighbor_phases, center_phase)
            if possible_rxn is not None:
                        possible_reactions.append(possible_rxn)

        return possible_reactions

    def get_product_from_scores(self, reactions_and_likelihoods: list[typing.Tuple[ScoredReaction, float]], reacting_species) -> typing.Tuple[int, ScoredReaction]:
        """Given a list of tuples containing a reaction and it's determined likelihood
        from the current system state and cell, construct an appropriately weighted distribution
        of reactions according to the likelihoods, and draw a sample representing the reaction
        that will proceed. Then, construct another distribution over the products of that reaction, and
        and weight it according to the product stoichiometry. Finally, draw from this distriution
        to determined the final product for the current cell under consideration.

        Args:
            reactions_and_likelihoods (typing.Tuple): A list of tuples in which the first element is
                a reaction and the second is the likelihood of it proceeding.

        Returns:
            int: The integer corresponding in the current simulation to the product phase formed.
        """
        rxns: list[ScoredReaction] = []
        scores: list[float] = []
        for rxn_likelihood in reactions_and_likelihoods:
            rxns.append(rxn_likelihood[0])
            scores.append(rxn_likelihood[1])

        scores: np.array = np.array(scores)
        normalized: np.array = normalize(scores)
        choices: np.array = np.array(range(0,len(rxns)))
        chosen_rxn: ScoredReaction = rxns[np.random.choice(choices, p=normalized)]

        volumes = self.rxn_set.volumes

        if chosen_rxn is not None:
            # Choose whether reaction will proceed according to volumetric reactant stoichiometry
            # likelihood = (chosen_rxn.reactant_stoich(reacting_species) *volumes[reacting_species]) / sum([chosen_rxn.reactant_stoich(r) * volumes[r] for r in chosen_rxn.reactants])
            likelihood = (chosen_rxn.reactant_stoich(reacting_species)) / sum([chosen_rxn.reactant_stoich(r) for r in chosen_rxn.reactants])
            draw = random.random()
            if draw < likelihood:
                possible_products: list[str] = chosen_rxn.products
                # likelihoods: np.array = np.array([chosen_rxn.product_stoich(p) * volumes[p] for p in possible_products])
                likelihoods: np.array = np.array([chosen_rxn.product_stoich(p) for p in possible_products])
                likelihoods: np.array = likelihoods / likelihoods.sum()
                new_phase_name: str = np.random.choice(possible_products, p=np.array(likelihoods))

                if new_phase_name in self.rxn_set.free_species:
                    new_phase_name = self.phase_map.FREE_SPACE
                # for el in set(self.rxn_set.free_species).intersection(chosen_rxn.products):
                #     if el in self.free_element_amounts:
                #         self.free_element_amounts[el] += 1
                #     else:
                #         self.free_element_amounts[el] = 1
                return self.phase_map.phase_to_int[new_phase_name],chosen_rxn
            else:
                return self.phase_map.phase_to_int[reacting_species],chosen_rxn
        else:
            return self.phase_map.free_space_id, None

    def get_rxn_and_score(self, reactants, distance, neighbor_phases, replaced_phase):
        possible_reaction = self.rxn_set.get_reaction(reactants)

        if possible_reaction is not None:
            score = self.get_score_contribution(possible_reaction.competitiveness, distance)
            if len(set(neighbor_phases) & set(possible_reaction.products)) == 0:
                score = score * 0.15

            return (possible_reaction, score)
        elif replaced_phase != self.phase_map.FREE_SPACE:
            rxn = self.rxn_set.get_reaction([replaced_phase])
            score = self.get_score_contribution(rxn.competitiveness, distance)
            return (rxn, score)
        else:
            return (None, 1)

    def get_score_contribution(self, weight, distance):
        return weight * 1 / (distance ** 2)

    def phase_count(self, step):
        return len(np.unique(step.state)) - 1

    def phases_present(self, step):
        return [self.phase_map.int_to_phase[p] for p in np.unique(step.state) if p != self.phase_map.free_space_id]

    def elemental_composition(self, step, mole_amts = None):
        phases = self.phases_present(step)
        elemental_amounts = {}
        total = 0
        for p in phases:
            comp = Composition(p)
            moles = self.moles_of(step, p, mole_amts)
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