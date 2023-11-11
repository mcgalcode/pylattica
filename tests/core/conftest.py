import pytest
from typing import Dict
from pylattica.core import Lattice
from pylattica.core import PeriodicStructure

@pytest.fixture(scope="module")
def square_2D_basis_vecs():
    return [
        (1, 0),
        (0, 1)
    ]

@pytest.fixture(scope="module")
def simple_motif(): 
    return {
        "A": [
            (0.5, 0.5)
        ]
    }

@pytest.fixture(scope="module")
def square_2D_lattice(square_2D_basis_vecs):
    return Lattice(square_2D_basis_vecs)    

@pytest.fixture(scope="module")
def square_2x2_2D_grid_in_test(square_2D_lattice: Lattice, simple_motif: Dict):
    return PeriodicStructure.build_from(square_2D_lattice, [2, 2], simple_motif)