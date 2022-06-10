from .scored_reaction import ScoredReaction, phases_to_str
from pymatgen.core.composition import Composition
from pymatgen.ext.matproj import MPRester
from pymatgen.core.units import Mass

import typing

class ScoredReactionSet():
    """A set of ScoredReactions that capture the events that can occur during a simulation. Typically
    includes every reaction possible in the chemical system defined by the precursors and open
    elements
    """

    @classmethod
    def from_dict(cls, rxn_set_dict):
        return cls(
            [ScoredReaction.from_dict(r) for r in rxn_set_dict["reactions"]],
            rxn_set_dict["open_species"],
            rxn_set_dict["free_species"]
        )

    def __init__(self, reactions: list[ScoredReaction], open_species: list[str] = [], free_species: list[str] = [], skip_vols: bool = False):
        """Initializes a SolidReactionSet object. Requires a list of possible reactions
        and the elements which should be considered available in the atmosphere of the
        simulation.

        Args:
            reactions (list[Reaction]):
            open_species (list[str], optional): A list of open species, e.g. CO2. Defaults to [].
            free_species (list[str]), optional): A list of gaseous or liquid species which should not be represented in the grid
        """
        phases: list[str] = []

        for r in reactions:
            if len(set(r.products) - set(open_species)) != 0:
                phases = phases + r.products + r.reactants

        phases = list(set(phases))
        self_reactions: list[ScoredReaction] = [ScoredReaction.self_reaction(phase) for phase in phases if phase not in open_species]
        self.phases: list[str] = phases
        if not skip_vols:
            self._get_phase_vols()
        self.reactions: list[ScoredReaction] = reactions + self_reactions
        self.open_species: list[str] = open_species
        self.free_species: list[str] = list(set(open_species + free_species))

        # A hashmap for quick lookup of reactions in this reaction set.
        self.reactant_map: typing.Dict[str, ScoredReaction] = { rxn.reactant_str(): rxn for rxn in self.reactions}
        self.rxn_map: typing.Dict[str, ScoredReaction] = { str(rxn): rxn for rxn in self.reactions}

    def _get_phase_vols(self):

        self.volumes = {}

        with MPRester() as mpr:
            res = mpr.query(criteria={
                "pretty_formula": {
                    "$in": self.phases,
                },
                "e_above_hull": 0
            }, properties=["structure", "full_formula", "task_id"])
            for item in res:
                struct = item["structure"]
                comp = Composition(item["full_formula"])
                density = struct.density
                scaling = comp.get_reduced_composition_and_factor()[1]
                molar_mass = Mass(comp.weight, "amu").to("g") * 6.022*10**23
                self.volumes[comp.reduced_formula] = molar_mass / (scaling * density)

    def get_reaction(self, reactants: list[str]) -> ScoredReaction:
        """Given a list of string reaction names, returns a reaction that uses exactly those
        reactants as precursors.

        Args:
            reactants (list[str]): The list of reactants to match with

        Returns:
            Reaction: The matching reaction, if it exists, otherwise None.
        """
        return self.reactant_map.get(phases_to_str(reactants), None)

    def get_rxn_by_str(self, rxn_str: str) -> ScoredReaction:
        """Retrieves a reaction from this set by it's serialized string form

        Args:
            rxn_str (str):

        Returns:
            Reaction:
        """
        return self.rxn_map.get(rxn_str)

    def search_products(self, products: list[str]) -> list[ScoredReaction]:
        """Returns all the reactions in this SolidReactionSet that produce all of the
        product phases specified.

        Args:
            products (list[str]): The products which matching reactions will produce.

        Returns:
            list[Reaction]: The matching reactions.
        """
        return [rxn for rxn in self.reactions if set(rxn.products).issuperset(products)]

    def to_dict(self):
        return {
            "reactions": [r.to_dict() for r in self.reactions],
            "open_species": self.open_species,
            "free_species": self.free_species
        }