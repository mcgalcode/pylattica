import math
import random
import numpy as np
import typing

from rxn_ca.core.neighborhoods import MooreNeighborhood, Neighborhood, VonNeumannNeighborhood
from rxn_ca.rxn.solid_phase_map import SolidPhaseMap
from ..core.basic_controller import BasicController

from .reaction_result import ReactionResult
from .reaction_step import ReactionStep
from .normalizers import normalize

from .scored_reaction_set import ScoredReactionSet
from .scored_reaction import ScoredReaction

class ReactionController(BasicController):

    @classmethod
    def get_neighborhood_from_size(cls, size, nb_type = MooreNeighborhood):
        neighborhood_radius = cls.nb_radius_from_size(size)
        return nb_type(neighborhood_radius)

    @classmethod
    def get_neighborhood_from_step(cls, step, nb_type = MooreNeighborhood):
        return cls.get_neighborhood_from_size(step.size, nb_type=nb_type)

    @classmethod
    def nb_radius_from_size(cls, size: int) -> int:
        """Provided the side length of the simulation stage, generate the
        appropriate filter size for the simulation. This routine chooses the smaller of:
            1. The filter size that will cover the entire reaction stage
            2. 25, which is a performance compromise
        If the filter size is 25, then we are looking at reactant pairs up to 12 squares
        away from eachother. Given that probability of proceeding scales as 1/d^2, this is equivalent
        to cutting off possibility after the scaling drops below 1/144.

        Args:
            size (int): The side length of the simulation stage.

        Returns:
            int: The side length of the filter used in the convolution step.
        """
        return math.floor(min((size - 1) * 2 + 1, 21) / 2)

    def __init__(self, phase_map: SolidPhaseMap, reaction_set: ScoredReactionSet, neighborhood: Neighborhood) -> None:
        self.phase_map: SolidPhaseMap = phase_map
        self.rxn_set: ScoredReactionSet = reaction_set
        self.neighborhood = neighborhood
        self.vn_neighborhood = VonNeumannNeighborhood(1)

        # proxy for partial pressures
        self.effective_open_distances = {}
        for specie in self.rxn_set.open_species:
            self.effective_open_distances[specie] = 5

    def instantiate_result(self):
        return ReactionResult(self.rxn_set, self.phase_map)

    def get_new_state(self, padded_state, row_num, j):
        possible_reactions = self.get_rxns_from_padded_state(padded_state, row_num, j)
        curr_species = self.species_at(padded_state, row_num, j)
        new_phase, chosen_rxn = self.get_product_from_scores(possible_reactions, curr_species)
        return new_phase, chosen_rxn

    def neighbors_in_padded_state(self, padded_state, i, j):
        neighbor_phases = []
        for cell, _ in self.vn_neighborhood.iterate(padded_state, i, j, overload_radius=1):
            neighbor_phases.append(self.phase_map.int_to_phase[cell])

        return neighbor_phases

    def species_at(self, padded_state, i, j):
        return self.phase_map.int_to_phase[self.neighborhood.state_at(padded_state, i, j)]

    def get_rxns_from_step(self, step: ReactionStep, i, j):
        padded_state = self.pad_state(step)
        return self.get_possible_reactions_at(padded_state, i, j)

    def get_rxns_from_padded_state(self, padded_state: np.array, i, j):
        this_phase = self.species_at(padded_state, i, j)
        neighbor_phases = self.neighbors_in_padded_state(padded_state, i, j)

        # Look through neighborhood, enumerate possible reactions
        possible_reactions = []
        for other_cell, distance in self.neighborhood.iterate(padded_state, i, j, exclude_center=True):
            other_phase = self.phase_map.int_to_phase[other_cell]
            possible_rxn = self.get_rxn_and_score([other_phase, this_phase], distance, neighbor_phases, this_phase)
            possible_reactions.append(possible_rxn)

        possible_reactions = possible_reactions + self.rxns_with_open_species(this_phase, neighbor_phases)

        return possible_reactions

    def rxns_with_open_species(self, this_phase, neighbor_phases):
        rxns = []
        for specie in self.rxn_set.open_species:
            dist_eff = self.effective_open_distances[specie]
            rxn = self.get_rxn_and_score([this_phase, specie], dist_eff, neighbor_phases, this_phase)
            rxns.append(rxn)
        return rxns

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
            likelihood = (chosen_rxn.reactant_stoich(reacting_species)) / sum([chosen_rxn.reactant_stoich(r) for r in chosen_rxn.reactants])
            draw = random.random()
            if draw < likelihood:
                possible_products: list[str] = chosen_rxn.products
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
            score = self.adjust_score_for_nucleation(score, neighbor_phases, possible_reaction.products)
            return (possible_reaction, score)
        elif replaced_phase == self.phase_map.FREE_SPACE:
            return (None, 1)
        else:
            rxn = self.rxn_set.get_reaction([replaced_phase])
            score = self.get_score_contribution(rxn.competitiveness, distance)
            return (rxn, score)


    def adjust_score_for_nucleation(self, score, neighbors, reaction_products):
        # if len(set(neighbors) & set(reaction_products)) == 0:
        #         score = score * 0.15

        for neighbor in neighbors:
            if neighbor not in reaction_products:
                score = score * 0.5
        return score


    def get_score_contribution(self, weight, distance):
        return weight * 1 / (distance ** 2)
