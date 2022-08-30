from maggma.stores.mongolike import MongoStore
from rxn_ca.rxn.computing.utils.functions import format_chem_sys

from rxn_ca.rxn.scored_reaction_set import ScoredReactionSet

from ..schemas.job_types import JobTypes

from rxn_network.reactions.reaction_set import ReactionSet


class AutomatonStore(MongoStore):


    def get_rxn_enumeration_by_task_id(self, task_id: str):
        result = self.query_one({
                "output.task_id": task_id,
                "output.job_type": JobTypes.ENUMERATE_RXNS.value
            })
        if result is not None:
            return ReactionSet.from_dict(result["output"]["rxn_set"])
        else:
            return None

    def get_scored_rxns_by_task_id(self, task_id: str):
        result = self.query_one({
            "output.task_id": task_id,
            "output.job_type": JobTypes.SCORE_RXNS.value
        })
        if result is not None:
            return ScoredReactionSet.from_dict(result["output"]["scored_rxn_set"])
        else:
            return None

    def get_scored_rxns(self, chem_sys, temperature):
        chem_sys = format_chem_sys(chem_sys)
        result = self.query_one({
            "output.job_type": JobTypes.SCORE_RXNS.value,
            "output.chem_sys": chem_sys,
            "output.temp": temperature
        })
        if result is not None:
            ScoredReactionSet.from_dict(result["output"]["scored_rxn_set"])
        else:
            return None

