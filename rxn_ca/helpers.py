from pymatgen.ext.matproj import MPRester

from rxn_network.enumerators.minimize import MinimizeGibbsEnumerator
from rxn_network.entries.entry_set import GibbsEntrySet
from rxn_network.reactions.reaction_set import ReactionSet

from rxn_ca.scorers import ArrheniusScore, score_reactions
from rxn_ca.scored_reaction_set import ScoredReactionSet

import os

import json

def download_rxns(chem_sys, temp, stability_cutoff, parent_dir='.', open_el = None, chempot=0.0):
    with MPRester() as mpr:  # insert your Materials Project API key here if it's not stored in .pmgrc.yaml
        entries = mpr.get_entries_in_chemsys(chem_sys, inc_structure="final")

    gibbs_entries = GibbsEntrySet.from_entries(entries, temp)
    entry_set = gibbs_entries.filter_by_stability(stability_cutoff)

    gibbs_enumerator = MinimizeGibbsEnumerator()
    gibbs_rxns = gibbs_enumerator.enumerate(entry_set)
    rxn_set = ReactionSet.from_rxns(
                gibbs_rxns, entry_set, open_elem=open_el, chempot=chempot
            )

    name = os.path.join(parent_dir, f'{chem_sys}-{temp}K.json')
    with open(name, 'w+') as f:
        f.write(json.dumps(rxn_set.as_dict()))

def score_and_write_rxns(unscored_path, output_path, temp, free_species = [], open_species = []):
    f = open(unscored_path, 'r+')
    rxn_set = ReactionSet.from_dict(json.loads(f.read()))
    f.close()
    scorer = ArrheniusScore(temp)
    scored_rxns = score_reactions(rxn_set, scorer)
    scored_rxn_set = ScoredReactionSet(scored_rxns, open_species = open_species, free_species=free_species)

    with open(output_path, 'w+') as f:
        f.write(json.dumps(scored_rxn_set.to_dict()))
