

class Reaction():
    
    @classmethod
    def self_reaction(cls, phase):
        reactants = {
            phase: 1
        }
        
        products = {
            phase: 1
        }
        
        return cls(reactants, products, 1)
    
    def __init__(self, reactants, products, competitiveness):
        self._reactants = reactants
        self._products = products
        self.competitiveness = competitiveness
    
    def can_proceed_with(self, reactants):
        return set(reactants) == set(self.reactants)
            
    @property
    def reactants(self):
        return list(self._reactants.keys())
    
    @property
    def products(self):
        return list(self._products.keys())
    
    def product_stoich(self, phase):
        return self._products[phase]
    
    def reactant_stoich(self, phase):
        return self._reactants[phase]