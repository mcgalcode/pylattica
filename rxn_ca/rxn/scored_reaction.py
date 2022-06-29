from __future__ import annotations
import typing
from numbers import Number

from rxn_network.reactions.basic import BasicReaction

def stoich_map_to_str(stoich_map: typing.Dict[str, Number]) -> str:
    """Generates a string that encapsulates a stoichiometry map. For example
    the map { "Na": 1, "Cl": 1 } will become 1Na + 1Cl. Useful for serializing
    reactions as strings.

    Args:
        stoich_map (dict): The stoichiometry map as exemplified in the description.

    Returns:
        string:
    """
    result = ""

    for phase, stoich in stoich_map.items():
        result = result + f"{stoich}{phase}+"

    result = result[:-1]
    return result


def phases_to_str(phases: list[str]) -> str:
    """Generates a string of phases concatenated with a plus sign.

    Args:
        phases (list[str]): _description_

    Returns:
        _type_: _description_
    """
    phases = sorted(list(set(phases)))
    return "+".join(phases)


class ScoredReaction:

    @classmethod
    def from_dict(cls, rxn_dict):
        return cls(
            rxn_dict["reactants"],
            rxn_dict["products"],
            rxn_dict["competitiveness"],
        )

    @classmethod
    def self_reaction(cls, phase: str) -> ScoredReaction:
        """Instantiates a reaction object representing the identity reaction, i.e. this phase
        reacting to form itself.

        Args:
            phase (str): The string name for this phase e.g. LiO2 or Na

        Returns:
            Reaction: The Reaction object representing the identity reaction
        """

        reactants = {phase: 1}

        products = {phase: 1}

        return cls(reactants, products, 1)

    @classmethod
    def from_rxn_network(cls, score, original_rxn: BasicReaction) -> ScoredReaction:
        react_dict = { comp.reduced_formula: round(-coeff, 3) for comp, coeff in original_rxn.reactant_coeffs.items() }
        product_dict = { comp.reduced_formula: round(coeff, 3) for comp, coeff in original_rxn.product_coeffs.items() }
        return ScoredReaction(react_dict, product_dict, score, original_rxn)

    def __init__(self, reactants, products, competitiveness, original_rxn = None):
        """Instantiate a reaction object by providing stoichiometry maps describing the
        reactant and product stoichiometry, and the relative competitiveness of this
        reaction.

        Args:
            reactants (typing.Dict[str, Number]): A map representing the stoichiometry of the
            reactants, e.g. { "Na": 1, "Cl": 1 }
            products (typing.Dict[str, Number]): A map representing the stoichiometry of the products.
            competitiveness (Number): A competitiveness score for the reaction.
        """
        self._reactants: typing.Dict[str, Number] = reactants
        self._products: typing.Dict[str, Number] = products
        self._total_reactant_stoich = sum(reactants.values())
        self._total_product_stoich = sum(reactants.values())
        self.competitiveness: Number = competitiveness
        self.original_rxn = original_rxn

    def can_proceed_with(self, reactants: list[str]) -> bool:
        """Helper method that, given a list of reactants, returns true if it is the same
        as the list of reactants for this reaction. Note that this is an exact match.

        Args:
            reactants (list[str]): A list of reactant phase names

        Returns:
            bool: True if the reactants match, otherwise False
        """
        return set(reactants) == set(self.reactants)

    def reactant_str(self) -> str:
        """Returns a string representing the reactant side of this reaction. For instance,
        1Na+1Cl

        Returns:
            str:
        """
        return phases_to_str(self.reactants)

    def is_identity(self) -> bool:
        """Indicates whether or not this reaction is an identity reaction

        Returns:
            bool:
        """
        return set(self.reactants) == set(self.products)

    @property
    def reactants(self):
        return list(self._reactants.keys())

    @property
    def products(self):
        return list(self._products.keys())

    @property
    def all_phases(self):
        return list(set(self.reactants + self.products))

    def stoich_ratio(self, phase1, phase2) -> Number:
        all_phases = {**self._reactants, **self._products}
        return all_phases[phase1] / all_phases[phase2]

    def product_stoich(self, phase: str) -> Number:
        """Returns the stoichiometry in this reaction for the desired product phase.

        Args:
            phase (str): The phase whose stoichiometry is desired

        Returns:
            Number:
        """
        return self._products[phase]

    def reactant_stoich(self, phase: str) -> Number:
        """Returns the stoichiometry in this reaction for the desired product phase.

        Args:
            phase (str): The phase whose stoichiometry is desired

        Returns:
            Number:
        """
        return self._reactants[phase]

    def reactant_stoich_fraction(self, phase: str) -> Number:
        """Returns the stoichiometry in this reaction for the desired product phase.

        Args:
            phase (str): The phase whose stoichiometry is desired

        Returns:
            Number:
        """
        return self._reactants[phase] / self._total_reactant_stoich

    @property
    def is_identity(self) -> bool:
        """Indicates whether or not this reaction is an identity reaction

        Returns:
            bool:
        """
        return set(self.reactants) == set(self.products)

    def __str__(self):
        return (
            f"{stoich_map_to_str(self._reactants)}->{stoich_map_to_str(self._products)}"
        )

    def as_dict(self):
        return {
            "reactants": self._reactants,
            "products": self._products,
            "competitiveness": self.competitiveness,
        }
