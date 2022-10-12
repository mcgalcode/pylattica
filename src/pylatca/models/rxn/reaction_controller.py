import math
import numpy as np
import typing

from rxn_ca.core.neighborhoods import StructureNeighborhoodSpec
from rxn_ca.core.periodic_structure import PeriodicStructure
from rxn_ca.core.simulation_step import SimulationState
from rxn_ca.grid2d.neighborhoods import MooreNbHoodSpec
from rxn_ca.rxn.scorers import ArrheniusScore, score_rxns
from rxn_ca.rxn.solid_phase_set import SolidPhaseSet
from ..core.basic_controller import BasicController

from .reaction_result import ReactionResult
from .normalizers import normalize

from rxn_network.reactions.reaction_set import ReactionSet

from .scored_reaction_set import ScoredReactionSet
from .scored_reaction import ScoredReaction

class ReactionController(BasicController):

    @classmethod
    def get_neighborhood_from_size(cls, size, nb_spec = MooreNbHoodSpec):
        neighborhood_radius = cls.nb_radius_from_size(size)
        return nb_spec(neighborhood_radius)

    @classmethod
    def get_neighborhood_from_structure(cls, structure: PeriodicStructure, nb_spec = MooreNbHoodSpec):
        return cls.get_neighborhood_from_size(structure.size, nb_spec=nb_spec)

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
        phase_set: SolidPhaseSet,
        structure: PeriodicStructure,
        nb_spec: StructureNeighborhoodSpec,
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

        self.structure = structure
        self.temperature = temperature
        self.phase_set: SolidPhaseSet = phase_set
        self.nb_graph = nb_spec.get(structure)
        self.nucleation_nb_graph = MooreNbHoodSpec(1).get(structure)
        self.inertia = inertia
        self.free_species = free_species
        # proxy for partial pressures
        self.effective_open_distances = {}
        for specie, strength in open_species.items():
            self.effective_open_distances[specie] = strength


    def instantiate_result(self, starting_state: SimulationState):
        return ReactionResult(starting_state, self.rxn_set, self.phase_set)

    def get_state_update(self, site_id: int, prev_state: SimulationState):
        np.random.seed(None)

        curr_state = prev_state.get_site_state(site_id)

        curr_species = curr_state['_disc_occupancy']
        if curr_species == self.phase_set.FREE_SPACE:
            return {}
        else:
            possible_reactions = self.rxns_at_site(site_id, prev_state)
            chosen_rxn = self.choose_reaction(possible_reactions)
            new_phase, chosen_rxn = self.get_product_from_rxn(chosen_rxn, curr_species)
            return { '_disc_occupancy': new_phase, 'rxn': chosen_rxn }

    def immediate_neighbors(self, site_id: int, state: SimulationState):
        neighbor_phases = []
        for nb_id in self.nucleation_nb_graph.neighbors_of(site_id):
            nb_state = state.get_site_state(nb_id)
            neighbor_phases.append(nb_state['_disc_occupancy'])

        return neighbor_phases

    def get_rxns_from_step(self, simulation_state, coords):
        site_id = self.structure.site_at(coords)['id']
        return self.rxns_at_site(site_id, simulation_state)

    def rxns_at_site(self, site_id: int, state: SimulationState):
        curr_state = state.get_site_state(site_id)
        this_phase = curr_state['_disc_occupancy']

        neighbor_phases = self.immediate_neighbors(site_id, state)

        # Look through neighborhood, enumerate possible reactions
        rxns = {}

        for nb_id, distance in self.nb_graph.neighbors_of(site_id, include_weights=True):
            assert distance != 0, "DISTANCE IS 0"

            neighbor_phase = state.get_site_state(nb_id)['_disc_occupancy']
            rxn, score = self.get_rxn_and_score([neighbor_phase, this_phase], distance, neighbor_phases, this_phase)
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
                new_phase_name = self.phase_set.FREE_SPACE

        return new_phase_name, rxn

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
        return weight * 5 / distance
