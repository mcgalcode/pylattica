from fireworks import FiretaskBase, explicit_serialize, FWAction

from rxn_ca.rxn.scorers import ArrheniusScore

from ..reaction_store import ReactionStore


@explicit_serialize
class ScoreRxns(FiretaskBase):
    """
    Given a path to a reaction repository directory, scores those reactions
    with the Arrhenius score and saves the scored reactions.

    Required params:
        chemsys (str): A chemical system like "Li-Mn-O-H-C" the describes which elements
            should be used to filter down materials project entries.
        temperature (float): A temperature at which the reaction energies should be calculated

    Optional params:
        unscored_rxn_set (ReactionSet): If specified, this reaction set will be scored
            as opposed to using the one stored in the fw_spec or on the filesystem
        unscored_reaction_path (str): If not using the reaction repository construct,
            a path to a downloaded ReactionSet to score.
        rxn_repo_dir (str): A path to the a directory which should be used as the reaction
            repository (i.e. the directory which was used int eh DownloadRxns job)
    """

    required_params = [
        "chemsys",
        "temperature",
    ]

    optional_params = [
        "unscored_reaction_set",
        "unscored_reaction_path"
        "rxn_repo_dir"
    ]

    def run_task(self, fw_spec,):
        chem_sys = self["chemsys"]
        temp = self["temperature"]
        rxn_path = self.get("unscored_reaction_path", None)
        rxn_path = rxn_path if rxn_path is not None else fw_spec.get("downloaded_rxns_path")
        rxn_repo_dir = self.get("rxn_repo_dir", ".")
        rxns = self.get("unscored_rxn_set")

        store = ReactionStore(repo_root=rxn_repo_dir)
        scorer = ArrheniusScore(temp)

        if rxns is not None:
            scored_rxns_path = store.score_and_write_rxns_from_obj(
                rxns,
                chem_sys,
                temp,
                scorer
            )
        elif rxn_path is not None:
            scored_rxns_path = store.score_and_write_rxns_from_path(
                rxn_path,
                chem_sys=chem_sys,
                temp=temp,
                scorer=scorer,
            )
        else:
            scored_rxns_path = store.score_and_write_rxns(
                chem_sys,
                temp,
                scorer
            )

        return FWAction(update_spec={"scored_rxns_path": scored_rxns_path })