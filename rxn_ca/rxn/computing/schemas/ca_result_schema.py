from pydantic import BaseModel, Field
from .job_types import JobTypes

class RxnCAResultModel(BaseModel):

    task_id: str = Field(description="The ID of this result")
    chem_sys: str = Field(description="The chemical system used")
    temperature: int = Field(description="The temperature of the simulation")
    result: dict = Field(description="The serialized result object")
    job_type: str = Field(default=JobTypes.RUN_RXN_AUTOMATON.value)