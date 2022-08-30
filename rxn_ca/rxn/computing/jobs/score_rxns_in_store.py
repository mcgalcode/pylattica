from jobflow.core.maker import Maker
from jobflow.core.job import job

from dataclasses import dataclass

from rxn_ca.rxn.computing.schemas.enumerated_rxns_schema import EnumeratedRxnsModel
from rxn_ca.rxn.computing.schemas.scored_rxns_schema import ScoredRxnsModel
from ..schemas.job_types import JobTypes

from ...scored_reaction_set import ScoredReactionSet

from ...scorers import ArrheniusScore, score_rxns

from rxn_network.reactions.reaction_set import ReactionSet

from maggma.stores.mongolike import MongoStore

@dataclass
class ScoreRxnsMaker(Maker):

    name: str = "Score Rxns In Store"

    @job
    def make(self,
             chem_sys: str,
             temp: int,
             rxns: EnumeratedRxnsModel = None,
             task_id: int = None,
             db_connection_params: dict = {}
        ):

        if rxns is None:
            store = MongoStore(**db_connection_params)
            store.connect()
            result = store.query_one({
                "output.task_id": task_id,
                "output.job_type": JobTypes.ENUMERATE_RXNS.value
            })
            rxn_set = ReactionSet.from_dict(result["output"]["rxn_set"])
        else:
            rxn_set = ReactionSet.from_dict(rxns.rxn_set)

        scorer = ArrheniusScore(temp)
        scored_rxns = score_rxns(rxn_set, scorer)
        scored_rxn_set = ScoredReactionSet(scored_rxns)

        result_model = ScoredRxnsModel.from_obj(
            scored_rxn_set,
            chem_sys = chem_sys,
            temp=temp
        )

        return result_model

