from pymatgen.core.structure import Structure as PmgStructure
from pymatgen.core.lattice import Lattice as PmgLattice

from ..core import (
    Lattice as PylLattice,
    StructureBuilder,
    PeriodicStructure,
    SimulationState,
)
from typing import Tuple
from ..discrete.state_constants import DISCRETE_OCCUPANCY
from ..core.constants import SITE_ID


class PymatgenStructureConverter:
    """A PymatgenStructureConverter provides utilities for converting pylattica
    PeriodicStructure and Lattice objects to pymatgen's Structure and Lattice objects.
    """

    def to_pylattica_lattice(self, pmg_lat: PmgLattice) -> PylLattice:
        """Converts a pymatgen Lattice to a pylattica Lattice

        Parameters
        ----------
        pmg_lat : PmgLattice
            The pymatgen lattice to convert

        Returns
        -------
        Lattice
            A pylattica lattice
        """
        pyl_lat = PylLattice(pmg_lat.matrix)
        return pyl_lat

    def to_pylattica_structure_builder(
        self, pmg_struct: PmgStructure
    ) -> StructureBuilder:
        """Converts a pymatgen Structure into a pylattica StructureBuilder which
        can be used to build pylattica Structures with the same symmetry as the
        input pymatgen Structure.

        Parameters
        ----------
        pmg_struct : PmgStructure
            The pymatgen structure to convert

        Returns
        -------
        StructureBuilder
            The resulting StructureBuilder
        """
        lat = self.to_pylattica_lattice(pmg_struct.lattice)

        struct_motif = {}

        for site in pmg_struct.sites:
            site_cls = site.species_string
            frac_coords = site.frac_coords

            if site_cls in struct_motif:
                struct_motif.get(site_cls).append(frac_coords)
            else:
                struct_motif[site_cls] = [frac_coords]

        struct_builder = StructureBuilder(lat, struct_motif)
        struct_builder.frac_coords = True
        return struct_builder

    def to_pylattica_structure_and_state(
        self, pmg_struct: PmgStructure
    ) -> Tuple[PeriodicStructure, SimulationState]:
        """Converts a pymatgen Structure into a pylattica StructureBuilder which
        can be used to build pylattica Structures with the same symmetry as the
        input pymatgen Structure.

        Parameters
        ----------
        pmg_struct : PmgStructure
            The pymatgen structure to convert

        Returns
        -------
        StructureBuilder
            The resulting StructureBuilder
        """
        struct_builder = self.to_pylattica_structure_builder(pmg_struct)

        struct = struct_builder.build(1)
        state = SimulationState.from_struct(struct)
        for site in pmg_struct.sites:
            site_id = struct.site_at(site.coords)[SITE_ID]
            state.set_site_state(site_id, {DISCRETE_OCCUPANCY: site.species_string})

        return struct, state

    def to_pymatgen_lattice(self, pyl_lat: PylLattice) -> PmgLattice:
        """Converts a pylattica Lattice object into a pymatgen Lattice object.

        Parameters
        ----------
        pyl_lat : PylLattice
            The pylattica Lattice to convert

        Returns
        -------
        PmgLattice
            The resulting pymatgen Lattice
        """
        return PmgLattice(pyl_lat.vecs)

    def to_pymatgen_structure(self, pyl_struct: PeriodicStructure) -> PmgStructure:
        """Converts a pylattica PeriodicStructure into a pymatgen

        Parameters
        ----------
        pyl_struct : PeriodicStructure
            The pylattica PeriodicStructure to convert

        Returns
        -------
        PmgStructure
            The resulting pymatgen Structure
        """
        pmg_lat = self.to_pymatgen_lattice(pyl_struct.lattice)

        species = []
        coords = []

        for sid in pyl_struct.site_ids:
            species.append(pyl_struct.site_class(sid))
            coords.append(
                pyl_struct.lattice.get_fractional_coords(pyl_struct.site_location(sid))
            )

        return PmgStructure(pmg_lat, species, coords)
