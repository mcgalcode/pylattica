from jobflow.core.maker import Maker
from jobflow.core.job import job

from dataclasses import dataclass

from rxn_ca.rxn.computing.schemas.enumerated_rxns_schema import EnumeratedRxnsModel
from rxn_ca.rxn.computing.schemas.scored_rxns_schema import ScoredRxnsModel
from ..schemas.job_types import JobTypes

from ...scored_reaction_set import ScoredReactionSet

from ...scorers import ArrheniusScore, score_rxns

from rxn_network.reactions.reaction_set import ReactionSet

from ..utils.automaton_store import AutomatonStore

@dataclass
class ScoreRxnsMaker(Maker):

    name: str = "Score Rxns In Store"

    @job
    def make(self,
             chem_sys: str,
             temp: int,
             rxns: EnumeratedRxnsModel = None,
             task_id: int = None,
             db_connection_params: dict = {},
             launchpad_file = None
        ):

        if rxns is None:
            if launchpad_file is not None:
                store = AutomatonStore.from_launchpad_file(launchpad_file)
            else:
                store = AutomatonStore(**db_connection_params)
            store.connect()
            rxn_set = store.get_rxn_enumeration_by_task_id(task_id)
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

