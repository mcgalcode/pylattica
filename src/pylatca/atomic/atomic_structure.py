from pylatca.discrete.state_constants import DISCRETE_OCCUPANCY, VACANT
from ..core.simulation_state import SimulationState
from ..core.periodic_structure import PeriodicStructure

from pylatca.core.simulation_state import SimulationState

import numpy as np

from pymatgen.core import Structure

class AtomicStructure(PeriodicStructure):
    """Represents a periodic structure where sites are occupied by atomic species.
    Supports an interface to pymatgen for extended materials science workflows.
    """    

    def to_pymatgen(self, state: SimulationState) -> Structure:
        """Generates a pymatgen atomic structure with atomic identities
        specified by a given simulation state.

        Parameters
        ----------
        state : SimulationState
            The state which contains occupancy information (i.e. each site has 
            an "occupancy" attribute with the atomic species at that site.)

        Returns
        -------
        Structure
            The pymatgen structure object representing this periodic structure with
            occupancies taken from the provided SimulationState.
        """        

        lattice_vecs = []
        for idx, b in enumerate(self.bounds):
            vec = []
            for i in range(idx):
                vec.append(0)
            vec.append(b)
            for i in range(idx, len(self.bounds) - 1):
                vec.append(0)
            lattice_vecs.append(vec)

        species = []
        sites = []

        for site in self.sites():
            site_state = state.get_site_state(site["id"])
            occ = site_state[DISCRETE_OCCUPANCY]
            if occ is not VACANT:
                species.append(occ)
                frac_loc = [loc / b for loc, b in zip(site['location'], self.bounds)]
                sites.append(np.array(frac_loc))

        sites = np.array(sites)

        return Structure(np.array(lattice_vecs), species, sites)