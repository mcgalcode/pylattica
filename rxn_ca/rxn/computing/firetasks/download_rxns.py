from fireworks import FiretaskBase, explicit_serialize, FWAction

from ..reaction_store import ReactionStore


@explicit_serialize
class DownloadRxns(FiretaskBase):
    """Downloads entries and enumerates reactions for use in Automata and KMC runs

    Required params:
        chemsys (str): A chemical system like "Li-Mn-O-H-C" the describes which elements
            should be used to filter down materials project entries.
        temperature (float): A temperature at which the reaction energies should be calculated
        stability_cutoff (float): The threshhold (in eV above the hull) at which a compound
            is no longer considered stable

    Optional params:
        open_el (str): An open element, used in conjunction with chempot to specify
            an altered chemical environment for which reaction energies should be calculated
        chempot (float): The chemical potential of the open element
        mp_api_key (str): The API key used to download entries from the materials project. If
            not specified, this will be drawn from the environment
        rxn_repo_dir (str): A path to a directory inside which the reactions should be saved
            after they are downloaded
    """

    required_params = [
        "chemsys",
        "temperature",
        "stability_cutoff"
    ]

    optional_params = [
        "open_el",
        "chempot",
        "mp_api_key",
        "rxn_repo_dir"
    ]

    def run_task(self, fw_spec):
        mp_api_key = self.get("mp_api_key", None)
        chem_sys = self["chemsys"]
        temp = self["temperature"]
        stability_cutoff = self["stability_cutoff"]
        open_el = self.get("open_el", None)
        rxn_repo_dir = self.get("rxn_repo_dir", ".")
        chempot = self.get("chempot", 0)

        store = ReactionStore(repo_root=rxn_repo_dir)

        rxns_path = store.download_rxns(
            chem_sys=chem_sys,
            temp=temp,
            stability_cutoff=stability_cutoff,
            open_el=open_el,
            chempot=chempot,
            mp_api_key=mp_api_key
        )

        return FWAction(update_spec={"downloaded_rxns_path": rxns_path })