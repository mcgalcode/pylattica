from rxn_ca.rxn.computing.schemas.scored_rxns_schema import ScoredRxnsModel
from ..jobs.run_automaton import RunRxnAutomatonMaker

from jobflow import Flow

def run_automaton_flow(
                    chem_sys: str,
                    temp: int,
                    setup_style: str,
                    setup_args: dict,
                    num_steps: int,
                    scored_rxns: ScoredRxnsModel = None,
                    task_id: str = None,
                    db_connection_params: dict = {},
                    dimensionality: int = 2,
                    inertia: float = 1,
                    open_species: dict = {},
                    free_species: list = [],
                    parallel: bool = True
    ):
    maker = RunRxnAutomatonMaker()
    job = maker.make(
        chem_sys = chem_sys,
        temp = temp,
        setup_style = setup_style,
        setup_args = setup_args,
        num_steps = num_steps,
        scored_rxns = scored_rxns,
        task_id = task_id,
        db_connection_params = db_connection_params,
        dimensionality = dimensionality,
        inertia = inertia,
        open_species = open_species,
        free_species = free_species,
        parallel = parallel,
    )
    flow = Flow([job])
    return flow