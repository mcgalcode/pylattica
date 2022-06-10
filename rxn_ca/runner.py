from rxn_ca.distance_map import DistanceMap
from rxn_ca.phase_map import PhaseMap
from rxn_ca.scored_reaction_set import ScoredReactionSet
from rxn_ca.step_analyzer import StepAnalyzer
from .reaction_result import ReactionResult
from .reaction_step import ReactionStep, get_filter_size_from_side_length

import numpy as np
from tqdm import tqdm

import multiprocessing as mp


mp_globals = {}


class Runner():
    """Class for orchestrating the running of the simulation. Provide this class a
    set of possible reactions and a ReactionStep that represents the initial system state,
    and it will run a simulation for the prescribed number of steps.
    """

    def __init__(self, parallel = False, workers = None):
        """Initializes a simulation Runner.

        Args:
            initial_step (ReactionStep): The initial state of the system for the simulation
            reaction_set (ScoredReactionSet): The set of reactions possible in the simulation
        """
        self.parallel = parallel
        self.workers = workers

    def run(self, initial_step: ReactionStep, reaction_set: ScoredReactionSet, phase_map: PhaseMap, num_steps: int) -> ReactionResult:
        """Run the simulation for the prescribed number of steps.

        Args:
            num_steps (int): The number of steps for which the simulation should run.

        Returns:
            ReactionResult:
        """
        print("Initializing run")
        step_analyzer = StepAnalyzer(phase_map, reaction_set)
        print("Initialized analyzer")
        self.free_element_amounts = {}
        result = ReactionResult(reaction_set, phase_map)
        print(f'Running w/ sim. size {initial_step.size}')
        if self.parallel:
            print('Running in parallel!')
        step = initial_step
        mols = step_analyzer.to_mole_array(step)

        result.add_step(step)
        result.add_mol_step(mols)

        filter_size = get_filter_size_from_side_length(initial_step.size)
        distance_map = DistanceMap(filter_size)
        global mp_globals

        if self.parallel:
            mp_globals['analyzer'] = step_analyzer
            mp_globals['distance_map'] = distance_map
            mp_globals['filter_size'] = filter_size
            mp_globals['step_size'] = initial_step.size

            if self.workers is None:
                PROCESSES = mp.cpu_count()
            else:
                PROCESSES = self.workers

            with mp.get_context('fork').Pool(PROCESSES) as pool:
                for _ in tqdm(range(num_steps)):
                    padded_state = step_analyzer.pad_state(step, filter_size)
                    new_state, rxn_choices = self._take_step_parallel(padded_state, mols, initial_step.size, pool)
                    step = ReactionStep(new_state)
                    result.add_step(step)
                    result.add_choices(rxn_choices)
        else:
            for _ in tqdm(range(num_steps)):
                padded_state = step_analyzer.pad_state(step, filter_size)
                step, rxn_choices, mols = self._take_step(padded_state, mols, filter_size, initial_step.size, step_analyzer, distance_map)
                result.add_mol_step(mols)
                result.add_step(step)
                result.add_choices(rxn_choices)

        return result

    def _take_step_parallel(self, padded_state, mols, state_size, pool) -> ReactionStep:
        """Given a ReactionStep, advances the system state by one time increment
        and returns a new reaction step.

        Args:
            step (ReactionStep):

        Returns:
            ReactionStep:
        """
        reaction_choices = {}
        params = []
        for i in range(0, state_size):
            params.append([padded_state, mols, i])
        # print(params)
        results = pool.starmap(step_row_parallel, params)

        new_state = np.array(list(map(lambda x: x[0], results)))
        for outcome in results:
            choices = outcome[1]
            for count, rxn in enumerate(choices):
                if rxn in reaction_choices:
                    reaction_choices[rxn] += count
                else:
                    reaction_choices[rxn] = count
        return new_state, reaction_choices


    def _take_step(self, padded_state: np.array, mole_amts: np.array, filter_size: int, state_size: int, step_analyzer: StepAnalyzer, distances: DistanceMap) -> ReactionStep:
        results = []
        reaction_choices = {}

        for i in range(0, state_size):
            results.append(step_row(padded_state, mole_amts, filter_size, state_size, i, step_analyzer, distances))

        new_state = np.array(list(map(lambda x: x[0], results)))
        new_mols = np.array(list(map(lambda x: x[2], results)))

        for outcome in results:
            choices = outcome[1]
            for count, rxn in enumerate(choices):
                if rxn in reaction_choices:
                    reaction_choices[rxn] += count
                else:
                    reaction_choices[rxn] = count
        return ReactionStep(new_state), reaction_choices, new_mols

def step_row_parallel(padded_state, moles, row_num):
    return step_row(
        padded_state,
        moles,
        mp_globals['filter_size'],
        mp_globals['step_size'],
        row_num,
        mp_globals['analyzer'],
        mp_globals['distance_map'],
    )

def step_row(padded_state: np.array, mole_amts: np.array, filter_size: int, state_size: int, row_num: int, step_analyzer: StepAnalyzer, distances: DistanceMap):
    # print(f'starting row {row_num}')
    reaction_choices = {}
    new_state = np.zeros(state_size)
    new_mole_amts = np.zeros(state_size)
    for j in range(0, state_size):
        possible_reactions = step_analyzer.get_rxns_from_padded_state(padded_state, row_num, j, filter_size, distances)
        curr_species = step_analyzer.species_at(padded_state, row_num, j, filter_size)
        new_phase, chosen_rxn = step_analyzer.get_product_from_scores(possible_reactions, curr_species)

        new_phase_name = step_analyzer.phase_map.int_to_phase[new_phase]
        if new_phase_name != step_analyzer.phase_map.FREE_SPACE:
            curr_mole_amt = mole_amts[row_num][j]
            new_mole_amt = curr_mole_amt # * chosen_rxn.stoich_ratio(new_phase_name, curr_species)
            new_mole_amts[j] = new_mole_amt
        else:
            new_mole_amts[j] = 0

        rxn_str = str(chosen_rxn)
        if rxn_str in reaction_choices:
            reaction_choices[rxn_str] += 1
        else:
            reaction_choices[rxn_str] = 1
        new_state[j] = new_phase
    return new_state, reaction_choices, new_mole_amts
