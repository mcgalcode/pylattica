from jobflow.core.maker import Maker
from jobflow.core.job import job
from jobflow import Response, Flow

from dataclasses import dataclass
from maggma.stores.mongolike import MongoStore

from rxn_ca.rxn.computing.schemas.ca_result_schema import RxnCAResultModel
from rxn_ca.rxn.computing.schemas.scored_rxns_schema import ScoredRxnsModel
from rxn_ca.rxn.reaction_result import ReactionResult

from .enumerate_rxns_maker import EnumerateRxnsMaker
from .score_rxns_in_store import ScoreRxnsMaker


from ..schemas.job_types import JobTypes

from ....core import VonNeumannNeighborhood, Runner
from ... import SolidPhaseMap, ReactionSetup, ReactionSetup3D, ScoredReactionSet, ReactionController

from uuid import uuid4


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

        if scored_rxns is not None:
            scored_rxn_set: ScoredReactionSet = ScoredReactionSet.from_dict(scored_rxns.scored_rxn_set)
        else:
            store = MongoStore(**db_connection_params)
            store.connect()
            if task_id is not None:
                result = store.query_one({
                    "output.task_id": task_id,
                    "output.job_type": JobTypes.SCORE_RXNS.value
                })
                scored_rxn_set = ScoredReactionSet.from_dict(result["output"]["scored_rxn_set"])
            else:
                result = store.query_one({
                    "output.job_type": JobTypes.SCORE_RXNS.value,
                    "output.chem_sys": chem_sys,
                    "output.temp": temp
                })
                if result is None:
                    enumerate_maker = EnumerateRxnsMaker()
                    score_maker = ScoreRxnsMaker()
                    run_maker = RunRxnAutomatonMaker()
                    enumerate_job = enumerate_maker.make(
                        chem_sys=chem_sys,
                        temp=temp,
                        stability_cutoff=0.1
                    )
                    score_job = score_maker.make(
                        chem_sys=chem_sys,
                        temp=temp,
                        rxns=enumerate_job.output
                    )
                    ca_job = run_maker.make(
                        chem_sys = chem_sys,
                        temp = temp,
                        setup_style = setup_style,
                        setup_args = setup_args,
                        num_steps = num_steps,
                        scored_rxns = score_job.output,
                        db_connection_params = db_connection_params,
                        dimensionality = dimensionality,
                        inertia = inertia,
                        open_species = open_species,
                        free_species = free_species,
                        parallel = parallel,
                    )
                    flow = Flow([enumerate_job, score_job, ca_job])
                    return Response(
                        replace=flow
                    )
                scored_rxn_set = ScoredReactionSet.from_dict(result["output"]["scored_rxn_set"])

        assert scored_rxn_set is not None, "ScoredRxnSet not found!"
        phase_map: SolidPhaseMap = SolidPhaseMap(scored_rxn_set.phases)
        print(scored_rxn_set)
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