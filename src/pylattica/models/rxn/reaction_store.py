from copy import copy
from pymatgen.ext.matproj import MPRester

from rxn_network.enumerators.minimize import MinimizeGibbsEnumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.reactions.reaction_set import ReactionSet

from .scorers import score_rxns
from .scored_reaction_set import ScoredReactionSet

import os
from pathlib import Path

from monty.serialization import loadfn, dumpfn

class ReactionStore():
    """
    Stores downloaded unscored and scored reactions sorted by chemical system
    in the filesystem.

    /{repo_root}
    -/ChemSys1
    --/unscored_rxns
    --/scored_rxns
    """

    def __init__(self, repo_root='.'):
        self.root = repo_root

    def get_fname_for_rxn_group(self, chem_sys, temp):
        return f'{chem_sys}-{temp}K.json'

    def get_sorted_chemsys_string(self, chem_sys):
        if type(chem_sys) is list:
            cs_copy = copy(chem_sys)
            cs_copy.sort()
            return "-".join(cs_copy)
        elif type(chem_sys) is str:
            cs_list = chem_sys.split("-")
            cs_list.sort()
            return "-".join(cs_list)

    def unscored_dir(self, chem_sys):
        dir_path = f'{self.root}/{self.get_sorted_chemsys_string(chem_sys)}/unscored_rxns'
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return dir_path

    def scored_dir(self, chem_sys):
        dir_path = f'{self.root}/{self.get_sorted_chemsys_string(chem_sys)}/scored_rxns'
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return dir_path

    def load_rxn_set(self, chem_sys, temp):
        return loadfn(self.get_scored_path(chem_sys, temp))

    def get_unscored_path(self, chem_sys, temp):
        return os.path.join(self.unscored_dir(chem_sys), self.get_fname_for_rxn_group(chem_sys, temp))

    def get_scored_path(self, chem_sys, temp):
        return os.path.join(self.scored_dir(chem_sys), self.get_fname_for_rxn_group(chem_sys, temp))

    def download_rxns(self, chem_sys, temp, stability_cutoff, open_el = None, chempot=0.0, mp_api_key = None):

        if mp_api_key is None:
            with MPRester() as mpr:  # insert your Materials Project API key here if it's not stored in .pmgrc.yaml
                entries = mpr.get_entries_in_chemsys(chem_sys, inc_structure="final")
        else:
            with MPRester(api_key=mp_api_key) as mpr:  # insert your Materials Project API key here if it's not stored in .pmgrc.yaml
                entries = mpr.get_entries_in_chemsys(chem_sys, inc_structure="final")

        gibbs_entries = GibbsEntrySet.from_entries(entries, temp)
        entry_set = gibbs_entries.filter_by_stability(stability_cutoff)

        gibbs_enumerator = MinimizeGibbsEnumerator()
        gibbs_rxns = gibbs_enumerator.enumerate(entry_set)
        rxn_set = ReactionSet.from_rxns(
                    gibbs_rxns, entry_set, open_elem=open_el, chempot=chempot
                )

        filepath = self.get_unscored_path(chem_sys, temp)
        dumpfn(rxn_set, filepath)
        return filepath

    def score_and_write_rxns(self, chem_sys, temp, scorer):
        unscored_path = self.get_unscored_path(chem_sys, temp)
        rxn_set = loadfn(unscored_path)
        return self.score_and_write_rxns_from_obj(rxn_set, chem_sys, temp, scorer)

    def score_and_write_rxns_from_path(self, rxn_fpath, chem_sys, temp, scorer):
        rxn_set = loadfn(rxn_fpath)
        return self.score_and_write_rxns_from_obj(rxn_set, chem_sys, temp, scorer)

    def score_and_write_rxns_from_obj(self, rxn_set: ReactionSet, chem_sys, temp, scorer):
        scored_rxns = score_rxns(rxn_set, scorer)
        scored_rxn_set = ScoredReactionSet(scored_rxns)
        scored_path = self.get_scored_path(chem_sys, temp)
        dumpfn(scored_rxn_set, scored_path)
        return scored_path
