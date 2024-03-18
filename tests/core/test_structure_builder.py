import pytest
import numpy as np

from pylattica.core import Lattice, PeriodicStructure
from pylattica.core.constants import LOCATION, SITE_CLASS
from pylattica.core.structure_builder import StructureBuilder


def test_basic_structure_building():
    lattice = Lattice([[1, 0], [0, 1]])

    motif = {"A": [(0.5, 0.5)], "B": [(0.25, 0.5)]}

    builder = StructureBuilder(lattice, motif)
    structure = builder.build((1, 1))

    assert len(structure.site_ids) == 2
    assert structure.site_at((0, 0)) is None
    assert structure.site_at((0.5, 0.5)) is not None
    assert structure.site_at((0.5, 0.5))[SITE_CLASS] is "A"

    assert structure.site_at((0.25, 0.5)) is not None
    assert structure.site_at((0.25, 0.5))[SITE_CLASS] is "B"


def test_basic_structure_building_size_as_int():
    lattice = Lattice([[1, 0], [0, 1]])

    motif = {"A": [(0.5, 0.5)], "B": [(0.25, 0.5)]}

    builder = StructureBuilder(lattice, motif)
    structure = builder.build(2)

    assert len(structure.site_ids) == 8
    assert structure.site_at((0, 0)) is None
    assert structure.site_at((0.5, 0.5)) is not None
    assert structure.site_at((0.5, 0.5))[SITE_CLASS] is "A"

    assert structure.site_at((0.25, 0.5)) is not None
    assert structure.site_at((0.25, 0.5))[SITE_CLASS] is "B"


def test_basic_structure_building_bad_size_value():
    lattice = Lattice([[1, 0], [0, 1]])

    motif = {"A": [(0.5, 0.5)], "B": [(0.25, 0.5)]}

    builder = StructureBuilder(lattice, motif)
    with pytest.raises(ValueError, match="Desired structure dimensions"):
        builder.build((2, 2, 2, 2))


def test_structure_building_unequal_dirs():
    lattice = Lattice([[1, 0], [0, 1]])

    motif = {"A": [(0.5, 0.5)], "B": [(0.25, 0.5)]}

    builder = StructureBuilder(lattice, motif)
    structure = builder.build((2, 1))

    assert len(structure.site_ids) == 4
    assert structure.site_at((0, 0)) is None
    assert structure.site_at((1.5, 0.5)) is not None
    assert structure.site_at((1.5, 0.5))[SITE_CLASS] is "A"

    assert structure.site_at((0.25, 0.5)) is not None
    assert structure.site_at((1.25, 0.5))[SITE_CLASS] is "B"


def test_structure_building_frac_coords():
    lattice = Lattice([[2, 0], [0, 2]])

    motif = {"A": [(0.5, 0.5)], "B": [(0.25, 0.5)]}

    builder = StructureBuilder(lattice, motif)
    builder.frac_coords = True
    structure = builder.build((2, 1))

    assert len(structure.site_ids) == 4
    assert structure.site_at((0, 0)) is None
    assert structure.site_at((3, 1)) is not None
    assert structure.site_at((3, 1))[SITE_CLASS] is "A"

    assert structure.site_at((0.5, 1)) is not None
    assert structure.site_at((0.5, 1))[SITE_CLASS] is "B"
