from .reaction_result import ReactionResult
from .reaction_step import ReactionStep
from .reaction import Reaction
import numpy as np
from tqdm import tqdm

class Runner():
    
    def __init__(self, initial_step, lens, lab):
        self.initial_step = initial_step
        self.lens = lens
        self.lab = lab
    
    def run(self, steps, distance_metric, normalizer):
        result = ReactionResult(self.lab)
        step = self.initial_step.state
        result.add_step(self.initial_step)
        for i in tqdm(range(steps)):
            step = self._take_step(step, distance_metric, normalizer)
            result.add_step(ReactionStep(step, self.lens, self.lab))
        
        return result
    
    def get_score_contribution(self, weight, distance):
        return weight * 1 / (distance ** 2)    

    def _take_step(self, step, calculate_distance, normalizer):
        width = int(self.lens[0])
        height = int(self.lens[1])
        lookahead = int((width - 1) / 2)
        lookbelow = int((height - 1) / 2)

        up_bound = lookbelow
        low_bound = step.shape[0] - lookbelow
        left_bound = lookahead
        right_bound = step.shape[1] - lookahead

        new_state = np.zeros(step.shape)
        for i in range(up_bound, low_bound):
            for j in range(left_bound, right_bound):
                curr_up = i - lookbelow
                curr_down = i + lookbelow + 1
                curr_left = j - lookahead
                curr_right = j + lookahead + 1

                subcell = step[curr_up:curr_down, curr_left:curr_right]
                possible_reactions = self.get_subcell_update(subcell, calculate_distance)
                new_phase = self.get_product_from_scores(possible_reactions, normalizer)
                new_state[(i,j)] = new_phase

        return new_state
        
    
    def get_subcell_update(self, subcell, calculate_distance):
        cell_center = np.array([int(subcell.shape[0] / 2), int(subcell.shape[1] / 2)])
        
        # Accumulate possible reactions here - this is a list of tuples: (reaction, likelihood)
        possible_reactions = []
        for i in range(subcell.shape[0]):
            for j in range(subcell.shape[1]):
                if not (i == cell_center[0] and j == cell_center[1]):
                    curr_loc = np.array([i, j])
                    distance = calculate_distance(cell_center, curr_loc)
                    r1 = self.lab.int_to_phase[subcell[tuple(curr_loc)]]
                    r2 = self.lab.int_to_phase[subcell[tuple(cell_center)]]
                    possible_reaction = self.lab.get_reaction(r1, r2)
                    if possible_reaction is not None:
                        score = self.get_score_contribution(possible_reaction.competitiveness, distance)
                        possible_reactions = possible_reactions + [(possible_reaction, score)]

        if len(possible_reactions) == 0:
            center_phase = self.lab.int_to_phase[subcell[tuple(cell_center)]]
            possible_reactions = [(Reaction.self_reaction(center_phase), 1)]
        
        return possible_reactions

    def get_product_from_scores(self, reactions_and_likelihoods, normalizer):
        rxns = []
        scores = []
        for rxn_likelihood in reactions_and_likelihoods:
            rxns.append(rxn_likelihood[0])
            scores.append(rxn_likelihood[1])

        scores = np.array(scores)
        normalized = normalizer(scores)
        choices = np.array(range(0,len(rxns)))
        chosen_rxn = rxns[np.random.choice(choices, p=normalized)]
        
        possible_products = chosen_rxn.products
        likelihoods = np.array([chosen_rxn.product_stoich(prod) for prod in possible_products])
        likelihoods = likelihoods / likelihoods.sum()
        new_phase_name = np.random.choice(possible_products, p=np.array(likelihoods))
        
        return self.lab.phase_to_int[new_phase_name]
    