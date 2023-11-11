from pylattica.core.neighborhood_builders import MotifNeighborhoodBuilder, SiteClassNeighborhoodBuilder, StochasticNeighborhoodBuilder, DistanceNeighborhoodBuilder
from pylattica.core import Lattice, PeriodicStructure

import numpy as np

def test_site_class_neighborhood():
    lattice = Lattice([
        [1, 0],
        [0, 1]
    ])

    motif = {
        "A": [[0.25, 0.25]],
        "B": [[0.5, 0.5]],
        "C": [[0.75, 0.75]],
    }

    struct = PeriodicStructure.build_from(
        lattice,
        (3,3),
        motif
    )

    A_builder = MotifNeighborhoodBuilder([(0.25, 0.25), (0.5, 0.5)])
    C_builder = MotifNeighborhoodBuilder([(0.5, 0.5)])

    overall_nbhood_builder = SiteClassNeighborhoodBuilder({
        "A": A_builder,
        "C": C_builder
    })
    overall_nbhood = overall_nbhood_builder.get(struct)

    A_site = struct.id_at((1.25, 1.25))
    B_site = struct.id_at((1.5, 1.5))
    C_site = struct.id_at((1.75, 1.75))

    A_nbs = overall_nbhood.neighbors_of(A_site)
    assert len(list(A_nbs)) == 2
    assert len([nbid for nbid in A_nbs if struct.site_class(nbid) == "B"]) == 1
    assert len([nbid for nbid in A_nbs if struct.site_class(nbid) == "C"]) == 1

    B_nbs = overall_nbhood.neighbors_of(B_site)
    assert len(list(B_nbs)) == 0
    
    C_nbs = overall_nbhood.neighbors_of(C_site)
    assert len(list(C_nbs)) == 1
    assert len([nbid for nbid in C_nbs if struct.site_class(nbid) == "A"]) == 1
    assert len([nbid for nbid in C_nbs if struct.site_class(nbid) == "B"]) == 0

def test_motif_nbhood():
    lattice = Lattice([
        [1, 0],
        [0, 1]
    ])
    
    motif = [[0.5, 0.5]]

    structure = PeriodicStructure.build_from(lattice, (3,3), motif)

    motif_builder = MotifNeighborhoodBuilder([(0,1)])

    nb = motif_builder.get(structure)

    assert len(nb.neighbors_of(4)) == 1
    assert np.allclose(structure.site_location(nb.neighbors_of(4)[0]), [1.5, 2.5])

def test_stochastic_nbhood():
    lattice = Lattice([
        [1, 0],
        [0, 1]
    ])

    motif = [[0.5, 0.5]]
    structure = PeriodicStructure.build_from(lattice, (5,5), motif)

    motif1 = MotifNeighborhoodBuilder([(0,1)])
    motif2 = MotifNeighborhoodBuilder([(0,-1)])
    motif3 = MotifNeighborhoodBuilder([(1,0)])
    motif4 = MotifNeighborhoodBuilder([(-1, 0)])

    stoch_nbbuilder = StochasticNeighborhoodBuilder([motif1, motif2, motif3, motif4])
    stoch_nb = stoch_nbbuilder.get(structure)

    assert len(stoch_nb.neighbors_of(0)) == 1

def test_fully_periodic_neighborhoods():
    lattice_vecs = [
        [1, 0],
        [0, 1]
    ]

    motif = [[0.5, 0.5]]

    von_neumann_nb_builder = DistanceNeighborhoodBuilder(1.01)

    full_periodic_lattice = Lattice(lattice_vecs, True)
    full_periodic_struct = PeriodicStructure.build_from(full_periodic_lattice, (3,3), motif)
    full_periodic_nbhood = von_neumann_nb_builder.get(full_periodic_struct)

    edge_coords = (0.5, 1.5)
    corner_coords = (0.5, 2.5)

    edge_id = full_periodic_struct.id_at(edge_coords)
    edge_nbs = full_periodic_nbhood.neighbors_of(edge_id)

    assert len(edge_nbs) == 4

    corner_id = full_periodic_struct.id_at(corner_coords)
    corner_nbs = full_periodic_nbhood.neighbors_of(corner_id)

    assert len(corner_nbs) == 4

def test_partially_periodic_neighborhoods():
    lattice_vecs = [
        [1, 0],
        [0, 1]
    ]

    motif = [[0.5, 0.5]]

    von_neumann_nb_builder = DistanceNeighborhoodBuilder(1.01)

    partial_periodic_lattice = Lattice(lattice_vecs, (False, True))
    partial_periodic_struct = PeriodicStructure.build_from(partial_periodic_lattice, (3,3), motif)
    partial_periodic_nbhood = von_neumann_nb_builder.get(partial_periodic_struct)

    edge_coords = (0.5, 1.5)
    corner_coords = (0.5, 2.5)

    edge_id = partial_periodic_struct.id_at(edge_coords)
    edge_nbs = partial_periodic_nbhood.neighbors_of(edge_id)

    assert len(edge_nbs) == 3

    corner_id = partial_periodic_struct.id_at(corner_coords)
    corner_nbs = partial_periodic_nbhood.neighbors_of(corner_id)

    assert len(corner_nbs) == 3


def test_non_periodic_neighborhoods():
    lattice_vecs = [
        [1, 0],
        [0, 1]
    ]

    motif = [[0.5, 0.5]]

    von_neumann_nb_builder = DistanceNeighborhoodBuilder(1.01)

    non_periodic_lattice = Lattice(lattice_vecs, False)
    non_periodic_struct = PeriodicStructure.build_from(non_periodic_lattice, (3,3), motif)
    non_periodic_nbhood = von_neumann_nb_builder.get(non_periodic_struct)

    edge_coords = (0.5, 1.5)
    corner_coords = (0.5, 2.5)

    edge_id = non_periodic_struct.id_at(edge_coords)
    edge_nbs = non_periodic_nbhood.neighbors_of(edge_id)

    assert len(edge_nbs) == 3

    corner_id = non_periodic_struct.id_at(corner_coords)
    corner_nbs = non_periodic_nbhood.neighbors_of(corner_id)

    assert len(corner_nbs) == 2