from pydantic import BaseModel, Field

from rxn_ca.rxn.scored_reaction_set import ScoredReactionSet
from uuid import uuid4

from .job_types import JobTypes

class ScoredRxnsModel(BaseModel):

    task_id: str = Field(description="The ID of this result")
    scored_rxn_set: dict = Field(description="The scored reactions")
    chem_sys: str = Field(description="The chemical system containing these reactions")
    job_type: str = Field(default=JobTypes.SCORE_RXNS.value)
    temp: int = Field(description="The temperature used to score these reactions")


    @classmethod
    def from_obj(cls, scored_rxn_set: ScoredReactionSet, chem_sys: str, temp: int):
        return cls(
            task_id = str(uuid4()),
            scored_rxn_set = scored_rxn_set.as_dict(),
            chem_sys = chem_sys,
            temp = temp
        )