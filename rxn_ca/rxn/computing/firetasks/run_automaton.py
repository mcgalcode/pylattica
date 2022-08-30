import datetime
from typing import Any, Dict, List
from fireworks import FiretaskBase, explicit_serialize, FWAction

from monty.serialization import loadfn
from monty.serialization import dumpfn

from rxn_ca.core.neighborhoods import VonNeumannNeighborhood
from rxn_ca.core import Runner

from rxn_ca.rxn import SolidPhaseMap
from rxn_ca.rxn.reaction_controller import ReactionController
from rxn_ca.rxn.reaction_setup import ReactionSetup3D, ReactionSetup
from rxn_ca.rxn.reaction_store import ReactionStore
from rxn_ca.rxn.scored_reaction_set import ScoredReactionSet
from ..reaction_store import ReactionStore


@explicit_serialize
class RunRxnAutomaton(FiretaskBase):
    """
    Runs a cellular automaton reaction simulation.

    Required params:
        sidelength (int): The simulation state size in number of grid cells
        setup_style (str): The type of initial state desired. Options include:
            interface
            random_mixture
        setup_args (dict): The arguments required for the specified setup_style.
            For details, see the ReactionSetup classes
        dimensionality (int): The dimensionality of the simulation state (2 or 3)
        chem_sys (str): The chemical system being simulated (e.g. Li-C-O)
        temperature (int): The temperature of the simulation
        num_step (int): The number of simulation steps to take

    Optional params:
        scored_rxns_path: A filepath for a ScoredReactionSet. Use instead of repo_root
            This will be drawn from the fw_spec otherwise, with the assumption that this
            job was invoked downstream of ScoreRxns
        repo_root: If providing the ScoredReactionSet via a ReactionStore, the location of
            the store
        inertia: An integer value that tunes the resistance to reactions happening in the
            simulation
        open_species: Species flown as gas or liquid across the reacting solid
        free_species: Species which evacuate the simulation state upon production (gasses or liquids)
        parallel: Whether or not this simulation should be parallelized
    """

    required_params = [
        "sidelength",
        "setup_style",
        "setup_args",
        "dimensionality",
        "chem_sys",
        "temperature",
        "num_steps"
    ]

    optional_params = [
        "scored_rxns_path",
        "repo_root",
        "inertia",
        "open_species",
        "free_species",
        "parallel"
    ]

    def run_task(self, fw_spec,):
        chem_sys: str = self["chem_sys"]
        temp: float = self["temperature"]
        rxn_path: str = self.get("scored_rxns_path", None)
        rxn_path = rxn_path if rxn_path is not None else fw_spec.get("scored_rxns_path")
        rxn_repo_dir: str = self.get("repo_root", ".")
        scored_rxn_set: ScoredReactionSet = self.get("scored_rxn_set")
        dimensionality: int = self.get("dimensionality", 2)
        setup_style: str = self.get("setup_style")
        sidelength: int = self.get("sidelength")
        setup_args: Any = self.get("setup_args", {})
        inertia: float = self.get("inertia", 1)
        open_species: Dict = self.get("open_species", {})
        free_species: List[str] = self.get("free_species", [])
        temperature: int = self.get("temperature")
        parallel: bool = self.get("parallel", True)
        num_steps: int = self.get("num_steps")

        repo = ReactionStore(rxn_repo_dir)

        if scored_rxn_set is None:
            if rxn_path is not None:
                scored_rxn_set = loadfn(rxn_path)
            else:
                scored_rxn_set = repo.load_rxn_set(chem_sys, temp)

        phase_map = SolidPhaseMap(scored_rxn_set.phases)
        if dimensionality == 2:
            setup = ReactionSetup(phase_map, volumes = scored_rxn_set.volumes)
        else:
            setup = ReactionSetup3D(phase_map, volumes = scored_rxn_set.volumes)


        if setup_style is "random_mixture":
            initial_state = setup.setup_random_mixture(**setup_args)
        elif setup_style is "interface":
            initial_state = setup.setup_interface(size=sidelength, **setup_args)

        nb = ReactionController.get_neighborhood_from_step(initial_state, nb_type = VonNeumannNeighborhood)
        controller = ReactionController(
            phase_map=phase_map,
            neighborhood=nb,
            scored_rxns=scored_rxn_set,
            inertia=inertia,
            open_species = open_species,
            free_species = free_species,
            temperature = temperature
        )

        runner = Runner(parallel=parallel)
        result = runner.run(initial_state, controller, num_steps)

        chem_sys_list = chem_sys.split("-")
        chem_sys_list.sort()
        sorted_chem_sys = "-".join(chem_sys_list)
        timestamp = datetime.utcnow().strftime('%Y-%m-%d-%H:%M:%S')
        result_fname = f'{sorted_chem_sys}_{temp}K_{setup_style}_{num_steps}steps_{timestamp}.json'

        dumpfn(result, result_fname)

        return FWAction(update_spec={"rxn_ca_result_path": result_fname })