import pytest

from pymatgen.core.structure import Structure

from pylattica.core import Lattice, StructureBuilder


@pytest.fixture()
def zr_pmg_struct():
    Zr = {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "charge": 0.0,
        "lattice": {
            "matrix": [
                [3.23923141, 0.0, 1.9834571889770337e-16],
                [-1.6196157049999995, 2.8052566897964866, 1.9834571889770337e-16],
                [0.0, 0.0, 5.17222],
            ],
            "pbc": [True, True, True],
            "a": 3.23923141,
            "b": 3.2392314099999995,
            "c": 5.17222,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 119.99999999999999,
            "volume": 46.99931962635987,
        },
        "properties": {},
        "sites": [
            {
                "species": [{"element": "Zr", "occu": 1.0}],
                "abc": [0.3333333333333333, 0.6666666666666666, 0.25],
                "xyz": [3.5599729479122526e-16, 1.870171126530991, 1.2930550000000003],
                "properties": {},
                "label": "Zr0",
            },
            {
                "species": [{"element": "Zr", "occu": 1.0}],
                "abc": [0.6666666666666666, 0.3333333333333333, 0.75],
                "xyz": [1.6196157050000002, 0.9350855632654955, 3.8791650000000004],
                "properties": {},
                "label": "Zr1",
            },
        ],
    }
    return Structure.from_dict(Zr)


@pytest.fixture()
def pyl_struct():
    lat = Lattice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    motif = {"Zr": [[0.5, 0.5, 0.5]]}

    builder = StructureBuilder(lat, motif)
    return builder.build((1, 2, 3))
