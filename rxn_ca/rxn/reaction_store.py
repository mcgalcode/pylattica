from pymatgen.ext.matproj import MPRester

from rxn_network.enumerators.minimize import MinimizeGibbsEnumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.reactions.reaction_set import ReactionSet

from .scorers import score_rxns
from .scored_reaction_set import ScoredReactionSet

import os
from pathlib import Path

import json

class ReactionStore():

    def __init__(self, repo_root='.'):
        self.root = repo_root
        self.unscored_root = f'{self.root}/unscored_rxns'
        self.scored_root = f'{self.root}/scored_rxns'
        Path(self.unscored_root).mkdir(parents=True, exist_ok=True)
        Path(self.scored_root).mkdir(parents=True, exist_ok=True)

    def get_fname_for_rxn_group(self, chem_sys, temp):
        return f'{chem_sys}-{temp}K.json'

    def load_rxn_set(self, chem_sys, temp):
        return ScoredReactionSet.from_file(self.get_scored_path(chem_sys, temp))

    def get_unscored_path(self, chem_sys, temp):
        return os.path.join(self.unscored_root, self.get_fname_for_rxn_group(chem_sys, temp))

    def get_scored_path(self, chem_sys, temp):
        return os.path.join(self.scored_root, self.get_fname_for_rxn_group(chem_sys, temp))

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
        with open(filepath, 'w+') as f:
            f.write(json.dumps(rxn_set.as_dict()))

    def score_and_write_rxns(self, chem_sys, temp, scorer, free_species = []):
        unscored_path = self.get_unscored_path(chem_sys, temp)
        f = open(unscored_path, 'r+')
        rxn_set = ReactionSet.from_dict(json.loads(f.read()))
        f.close()
        scored_rxns = score_rxns(rxn_set, scorer)
        scored_rxn_set = ScoredReactionSet(scored_rxns, free_species=free_species)

        scored_path = self.get_scored_path(chem_sys, temp)
        with open(scored_path, 'w+') as f:
            f.write(json.dumps(scored_rxn_set.to_dict()))
