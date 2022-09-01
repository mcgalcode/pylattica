
from rxn_ca.rxn.scored_reaction_set import ScoredReactionSet
from ..utils.automaton_store import AutomatonStore

from .run_automaton import RunRxnAutomatonMaker
from .enumerate_and_score_flow import enumerate_and_score_flow

from jobflow import Flow

RUN_AUTOMATON_FLOW = "RUN_AUTOMATON_FLOW"

def run_automaton_flow(
                    chem_sys: str,
                    temp: int,
                    setup_style: str,
                    setup_args: dict,
                    num_steps: int,
                    scored_rxn_set: ScoredReactionSet = None,
                    task_id: str = None,
                    db_connection_params: dict = {},
                    dimensionality: int = 2,
                    inertia: float = 1,
                    open_species: dict = {},
                    free_species: list = [],
                    parallel: bool = True
    ):

    # store = AutomatonStore(**db_connection_params)
    # store.connect()

    jobs = []

    scored_rxn_model = None

    # if scored_rxn_set is None and task_id is not None:
    #     scored_rxn_set = store.get_scored_rxns_by_task_id(task_id)

    # if scored_rxn_set is None:
    #     scored_rxn_set = store.get_scored_rxns(chem_sys, temperature=temp)

    # if scored_rxn_set is None:
    #     print(f'No rxns for chemical system {chem_sys} and temp {temp} found, adding enumeration')
    #     get_rxns_flow = enumerate_and_score_flow(
    #         chem_sys,
    #         temp
    #     )

    #     jobs.append(get_rxns_flow)
    #     scored_rxn_model = get_rxns_flow.output

    maker = RunRxnAutomatonMaker()
    job = maker.make(
        chem_sys = chem_sys,
        temp = temp,
        setup_style = setup_style,
        setup_args = setup_args,
        num_steps = num_steps,
        scored_rxns_task_id = task_id,
        scored_rxns = scored_rxn_model,
        db_connection_params = db_connection_params,
        dimensionality = dimensionality,
        inertia = inertia,
        open_species = open_species,
        free_species = free_species,
        parallel = parallel,
    )

    jobs.append(job)

    return Flow(jobs, name=RUN_AUTOMATON_FLOW)
