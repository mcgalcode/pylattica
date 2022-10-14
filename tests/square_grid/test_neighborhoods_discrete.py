from pylattica.core.periodic_structure import PeriodicStructure
from pylattica.square_grid.neighborhoods import MooreNbHoodSpec

def test_moore_neighborhood(square_grid_2D_2x2: PeriodicStructure):
    spec = MooreNbHoodSpec()
    nb_hood = spec.get(square_grid_2D_2x2)
