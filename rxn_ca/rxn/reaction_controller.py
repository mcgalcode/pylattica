import math
import random
import numpy as np
import typing

from rxn_ca.core.neighborhoods import MooreNeighborhood
from ..core.basic_controller import BasicController
from ..discrete import PhaseMap

from .reaction_result import ReactionResult
from .reaction_step import ReactionStep, get_filter_size_from_side_length
from .normalizers import normalize

from .scored_reaction_set import ScoredReactionSet
from .scored_reaction import ScoredReaction

class ReactionController(BasicController):

    def __init__(self, phase_map: PhaseMap, reaction_set: ScoredReactionSet, step_size: int) -> None:
        self.phase_map: PhaseMap = phase_map
        self.rxn_set: ScoredReactionSet = reaction_set
        self.filter_size = get_filter_size_from_side_length(step_size)
        self.step_size = step_size
        self.neighborhood_radius = math.floor(self.filter_size / 2)
        self.neighborhood = MooreNeighborhood(self.neighborhood_radius)

    def instantiate_result(self):
        return ReactionResult(self.rxn_set, self.phase_map)

    def get_new_state(self, padded_state, row_num, j):
      possible_reactions = self.get_rxns_from_padded_state(padded_state, row_num, j)
      curr_species = self.species_at(padded_state, row_num, j)
      new_phase, chosen_rxn = self.get_product_from_scores(possible_reactions, curr_species)
      return new_phase, chosen_rxn

    def neighbors_in_padded_state(self, padded_state, i, j):
        neighbor_phases = []
        for n_i in range(-1, 2):
            for n_j in range(-1, 2):
                if np.abs(n_i) + np.abs(n_j) == 1:
                    phase_int = int(padded_state[(i + n_i + self.neighborhood_radius, j + self.neighborhood_radius + n_j)])
                    neighbor_phases.append(self.phase_map.int_to_phase[phase_int])

        return neighbor_phases

    def species_at(self, padded_state, i, j):
        return self.phase_map.int_to_phase[padded_state[i + self.neighborhood_radius, j + self.neighborhood_radius]]

    def pad_state(self, step):
        return self.neighborhood.pad_state(step, self.phase_map.free_space_id)

    def get_rxns_from_step(self, step: ReactionStep, i, j):
        padded_state = self.pad_state(step, self.filter_size)
        return self.get_possible_reactions_at(padded_state, i, j)

    def get_rxns_from_padded_state(self, padded_state: np.array, i, j):
        center_phase = self.species_at(padded_state, i, j)
        neighbor_phases = self.neighbors_in_padded_state(padded_state, i, j)

        # Accumulate possible reactions here - this is a list of tuples: (reaction, likelihood)
        possible_reactions = []
        for cell, distance in self.neighborhood.iterate(padded_state, i, j, exclude_center=True):
            r1 = self.phase_map.int_to_phase[cell]
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
