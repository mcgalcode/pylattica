import math
import numpy as np
import typing

from rxn_ca.core.neighborhoods import MooreNeighborhood, Neighborhood, NeighborhoodView, VonNeumannNeighborhood
from rxn_ca.rxn.scorers import ArrheniusScore, score_rxns
from rxn_ca.rxn.solid_phase_map import SolidPhaseMap
from ..core.basic_controller import BasicController

from .reaction_result import ReactionResult
from .reaction_step import ReactionStep
from .normalizers import normalize

from rxn_network.reactions.reaction_set import ReactionSet

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

    def __init__(self,
        phase_map: SolidPhaseMap,
        neighborhood: Neighborhood,
        reaction_set: ReactionSet = None,
        scored_rxns: ScoredReactionSet = None,
        inertia = 1,
        open_species = {},
        free_species = [],
        temperature = None,
    ) -> None:

        if scored_rxns is not None:
            self.rxn_set = scored_rxns
        elif reaction_set is not None:
            if temperature is None:
                raise ValueError("If no scored reaction set is provided, temperature is required")
            scorer = ArrheniusScore(temperature)
            scored_rxns = score_rxns(reaction_set, scorer)
            self.rxn_set = ScoredReactionSet(scored_rxns)

        self.temperature = temperature
        self.phase_map: SolidPhaseMap = phase_map
        self.neighborhood = neighborhood
        self.nucleation_neighborhood = MooreNeighborhood(1)
        self.inertia = inertia
        self.free_species = free_species
        # proxy for partial pressures
        self.effective_open_distances = {}
        for specie, strength in open_species.items():
            self.effective_open_distances[specie] = strength


    def instantiate_result(self):
        return ReactionResult(self.rxn_set, self.phase_map)

    def get_new_state(self, nb_view: NeighborhoodView):
        np.random.seed(None)
        curr_species = self.phase_map.get_state_name(nb_view.center_value)
        if curr_species == self.phase_map.FREE_SPACE:
            return nb_view.center_value, None
        else:
            possible_reactions = self.rxns_from_view(nb_view)
            chosen_rxn = self.choose_reaction(possible_reactions)
            new_phase, chosen_rxn = self.get_product_from_rxn(chosen_rxn, curr_species)
            return new_phase, chosen_rxn

    def immediate_neighbors(self, state, coords):
        neighbor_phases = []
        view = self.nucleation_neighborhood.get(state, coords)
        for cell, _ in view.iterate(exclude_center=True):
            neighbor_phases.append(self.phase_map.get_state_name(cell))

        return neighbor_phases

    def get_rxns_from_step(self, step: ReactionStep, coords):
        view = self.neighborhood.get_in_step(step, coords)
        return self.rxns_from_view(view)

    def rxns_from_view(self, nb_view: NeighborhoodView):
        this_phase = self.phase_map.get_state_name(nb_view.center_value)
        neighbor_phases = self.immediate_neighbors(nb_view.full_state, nb_view.coords)

        # Look through neighborhood, enumerate possible reactions
        rxns = {}
        for other_cell, distance in nb_view.iterate(exclude_center=True):
            other_phase = self.phase_map.get_state_name(other_cell)
            rxn, score = self.get_rxn_and_score([other_phase, this_phase], distance, neighbor_phases, this_phase)
            if str(rxn) in rxns:
                rxns[str(rxn)]['score'] += score
            else:
                rxns[str(rxn)] = {
                    'reaction': rxn,
                    'score': score
                }

        possible_reactions = {**rxns, **self.rxns_with_open_species(this_phase, neighbor_phases)}

        return possible_reactions

    def rxns_with_open_species(self, this_phase, neighbor_phases):
        rxns = {}
        for specie, dist in self.effective_open_distances.items():
            rxn, score = self.get_rxn_and_score([this_phase, specie], dist, neighbor_phases, this_phase)
            if str(rxn) in rxns:
                rxns[str(rxn)]['score'] += score
            else:
                rxns[str(rxn)] = {
                    'reaction': rxn,
                    'score': score
                }

        return rxns

    def choose_reaction(self, rxns_and_scores):
        rxns: list[ScoredReaction] = []
        scores: list[float] = []

        for rxn in rxns_and_scores.values():
            rxns.append(rxn['reaction'])
            scores.append(rxn['score'])

        scores: np.array = np.array(scores)
        normalized: np.array = normalize(scores)
        choices: np.array = np.array(range(0,len(rxns)))
        chosen_rxn: ScoredReaction = rxns[np.random.choice(choices, p=normalized)]

        # chosen_rxn: ScoredReaction = rxns[np.argmax(scores)]

        return chosen_rxn

    def get_product_from_rxn(self, rxn: ScoredReaction, reacting_species) -> typing.Tuple[int, ScoredReaction]:
        """Given a list of tuples containing a reaction and it's determined likelihood
        from the current system state and cell, construct an appropriately weighted distribution
        of reactions according to the likelihoods, and draw a sample representing the reaction
        that will proceed. Then, construct another distribution over the products of that reaction, and
        and weight it according to the product stoichiometry. Finally, draw from this distriution
        to determined the final product for the current cell under consideration.

        Args:
            rxns_and_scores (typing.Tuple): A list of tuples in which the first element is
                a reaction and the second is the likelihood of it proceeding.

        Returns:
            int: The integer corresponding in the current simulation to the product phase formed.
        """

        new_phase_name = reacting_species

        chosen_reactant: str = self.choose_reactant_by_volume(rxn)

        if chosen_reactant == reacting_species:
            new_phase_name: str = self.choose_product_by_volume(rxn)

            if new_phase_name in self.free_species:
                new_phase_name = self.phase_map.FREE_SPACE

        return self.phase_map.get_state_value(new_phase_name), rxn

    def choose_product_by_volume(self, rxn):
        product_stoichs = [rxn.product_stoich(p) for p in rxn.products]
        return self.choose_phase_from_volume_weighted_stoich(rxn.products, product_stoichs)

    def choose_reactant_by_volume(self, rxn):
        reactant_stoichs = [rxn.reactant_stoich(p) for p in rxn.reactants]
        return self.choose_phase_from_volume_weighted_stoich(rxn.reactants, reactant_stoichs)

    def choose_phase_from_volume_weighted_stoich(self, phases, stoichs):
        volumes = self.rxn_set.volumes
        likelihoods: np.array = np.array([stoichs[idx] * volumes.get(phase, 1) for idx, phase in enumerate(phases)])
        likelihoods: np.array = likelihoods / likelihoods.sum()
        return np.random.choice(phases, p=np.array(likelihoods))


    def get_rxn_and_score(self, reactants, distance, neighbor_phases, replaced_phase):
        possible_reaction = self.rxn_set.get_reaction(reactants)

        if possible_reaction is not None:
            score = self.get_score_contribution(possible_reaction.competitiveness, distance)
            score = self.adjust_score_for_nucleation(score, neighbor_phases, possible_reaction.products, possible_reaction.reactants)
            return (possible_reaction, score)
        else:
            # Utilize the self reaction
            rxn = self.rxn_set.get_reaction([replaced_phase])
            score = self.get_score_contribution(self.inertia, distance)
            return (rxn, score)


    # def adjust_score_for_nucleation(self, score, neighbors, products, reactants):
    #     total_friendly_neighbors = len(neighbors)

    #     for neighbor in neighbors:
    #         if neighbor not in products:
    #             score = score * 0.8
    #         if neighbor not in reactants and neighbor not in products:
    #             total_friendly_neighbors -= 1

    #     if total_friendly_neighbors == 0:
    #         return 0
    #     else:
    #         return score

    def adjust_score_for_nucleation(self, score, neighbors, products, reactants):
        # penalize a new phase not neighboring any products
        if not any([n in products for n in neighbors]):
            return score * 0.15
        else:
            return score

    def get_score_contribution(self, weight, distance):
        return weight * 5 / (distance)
