import pytest

from pathlib import Path
from pymatgen.core.structure import Structure

from pylattica.core import Lattice, StructureBuilder

THIS_DIR = Path(__file__).parent

@pytest.fixture()
def zr_pmg_struct():
    return Structure.from_file(THIS_DIR / "./Zr.cif")

@pytest.fixture()
def pyl_struct():

    lat = Lattice([[1,0,0], [0,1,0], [0,0,1]])

    motif = {
        "Zr": [[
            0.5, 0.5, 0.5
        ]]
    }

    builder = StructureBuilder(lat, motif)
    return builder.build((1, 2, 3))