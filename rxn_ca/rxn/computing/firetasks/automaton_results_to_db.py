import datetime
from typing import Any, Dict, List
from fireworks import FiretaskBase, explicit_serialize, FWAction

from monty.serialization import loadfn
from monty.serialization import dumpfn

from rxn_ca.core.neighborhoods import VonNeumannNeighborhood
from rxn_ca.core import Runner

from rxn_ca.rxn import SolidPhaseMap
from rxn_ca.rxn.reaction_result import ReactionResult
from ..reaction_store import ReactionStore

from rxn_network.firetasks.utils import load_json, env_chk

@explicit_serialize
class AutomatonResultsToDb(FiretaskBase):
    """
    Given a filepath to a serialized CA result, store that result in a MongoDb.

    """

    required_params = []

    optional_params = [
        "rxn_ca_result_path",
        "db_file"
    ]

    def run_task(self, fw_spec,):
        ca_result: ReactionResult = load_json(self, "rxn_ca_result_path", fw_spec)
        db_file = env_chk(self.get("db_file"), fw_spec)

        db = CalcDb(db_file)

        return FWAction(update_spec={"result_filepath": result_fname })