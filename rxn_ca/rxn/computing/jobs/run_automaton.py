from jobflow.core.maker import Maker
from jobflow.core.job import job
from jobflow import Response, Flow

from dataclasses import dataclass

from rxn_ca.rxn.computing.schemas.ca_result_schema import RxnCAResultModel
from rxn_ca.rxn.computing.schemas.scored_rxns_schema import ScoredRxnsModel
from rxn_ca.rxn.computing.utils.automaton_store import AutomatonStore
from rxn_ca.rxn.reaction_result import ReactionResult

from ....core import VonNeumannNeighborhood, Runner
from ... import SolidPhaseMap, ReactionSetup, ReactionSetup3D, ScoredReactionSet, ReactionController

from uuid import uuid4

from ..utils.log import get_logger


@dataclass
class RunRxnAutomatonMaker(Maker):
    name: str = "Run Rxn Automaton"

    @job
    def make(self,
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
            parallel: bool = True,
        ):

        log = get_logger()
        scored_rxn_set = None

        if scored_rxns is not None:
            scored_rxn_set: ScoredReactionSet = ScoredReactionSet.from_dict(scored_rxns.scored_rxn_set)

        if scored_rxn_set is None and task_id is not None:
            scored_rxn_set = store.get_scored_rxns_by_task_id(task_id)

        if scored_rxn_set is None:
            store = AutomatonStore(**db_connection_params)
            store.connect()
            scored_rxn_set = store.get_scored_rxns(chem_sys, temperature=temp)

        assert scored_rxn_set is not None, "ScoredRxnSet not found!"
        phase_map: SolidPhaseMap = SolidPhaseMap(scored_rxn_set.phases)

        if dimensionality == 2:
            setup = ReactionSetup(phase_map, volumes = scored_rxn_set.volumes)
        else:
            setup = ReactionSetup3D(phase_map, volumes = scored_rxn_set.volumes)


        if setup_style == "random_mixture":
            initial_state = setup.setup_random_mixture(**setup_args)
        elif setup_style == "interface":
            initial_state = setup.setup_interface(**setup_args)

        nb = ReactionController.get_neighborhood_from_step(initial_state, nb_type = VonNeumannNeighborhood)
        controller = ReactionController(
            phase_map=phase_map,
            neighborhood=nb,
            scored_rxns=scored_rxn_set,
            inertia=inertia,
            open_species = open_species,
            free_species = free_species,
            temperature = temp
        )

        runner = Runner(parallel=parallel)
        result: ReactionResult = runner.run(initial_state, controller, num_steps)

        return RxnCAResultModel(
            task_id=str(uuid4()),
            chem_sys=chem_sys,
            temperature=temp,
            result=result.as_dict()
        )