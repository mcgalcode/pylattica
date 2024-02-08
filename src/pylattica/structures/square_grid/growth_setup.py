from ...core import SynchronousRunner, Simulation
from ...core.neighborhood_builders import NeighborhoodBuilder
from ...discrete import PhaseSet
from ...models.growth import GrowthController
from .grid_setup import DiscreteGridSetup

from typing import Dict


class GrowthSetup:
    """A class for generating simulation starting states by a growth + impingement algorithm.
    This algorithm can be summarized as follows:

    1. Random sites are selected in the strucure and assigned phases in the ratios
    specified.
    2. A simulation is run in which unassigned cells adopt the phase of their occupied
    neighbors, allowing particles to "grow" outward into unoccupied space until they
    impinge on one another.
    """

    def __init__(self, phase_set: PhaseSet, dim=2):
        """Instantiates the GrowthSetup object.

        Parameters
        ----------
        phase_set : PhaseSet
            The phases to be used in the growth process.
        dim : int, optional
            The dimension of the desired simulation., by default 2
        """
        self._phases = phase_set
        self.dim = dim

    def grow(
        self,
        size: int,
        num_sites_desired: int,
        background_spec: str,
        nuc_amts: Dict[str, float],
        nb_builder: NeighborhoodBuilder,
        buffer: int = 2,
    ) -> Simulation:
        """Runs the growth simulation and produces a starting state.

        Parameters
        ----------
        size : int
            The size of the desired simulation.
        num_sites_desired : int
            The number of nucleation sites for growing particles.
        background_spec : str
            The background species.
        nuc_amts : Dict[str, float]
            A mapping of phase names to their relative desired amounts.
        nb_builder : NeighborhoodBuilder
            The NeighborhoodBuilder to use to identify the growth environment for
            each particle.
        buffer : int, optional
            The minimum distance between seed sites, by default 2

        Returns
        -------
        Simulation
            The resulting Simulation.
        """
        setup = DiscreteGridSetup(self._phases, dim=self.dim)

        simulation = setup.setup_random_sites(
            size,
            num_sites_desired=num_sites_desired,
            background_spec=background_spec,
            nuc_amts=nuc_amts,
            buffer=buffer,
        )

        controller = GrowthController(
            self._phases,
            simulation.structure,
            nb_builder=nb_builder,
            background_phase=background_spec,
        )

        runner = SynchronousRunner(parallel=True)
        res = runner.run(simulation.state, controller, num_steps=size)
        return Simulation(res.last_step, simulation.structure)
