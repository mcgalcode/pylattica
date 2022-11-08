import random

from ...core import BasicController, Neighborhood, SimulationState
from ...core.periodic_structure import PeriodicStructure
from ...core.simulation_state import GENERAL
from ...core.utils import printif

from .consts import (
    TET_SITE,
    InitialAtomCount,
    InitialLiCount,
    Mn2Count,
    Occupancy,
    RemovedLiCount,
    Vacant,
    Li,
    Mn,
    Ti,
    OCT_SITE,
)


class ChargeController(BasicController):
    def __init__(
        self, nb_graph: Neighborhood, structure: PeriodicStructure, verbose=False
    ):
        self.nbs = nb_graph
        self.struct = structure
        self.verbose = verbose

    def get_random_site(self):
        oct_sites = [site["id"] for site in self.struct.sites(OCT_SITE)]
        return random.choice(oct_sites)

    def get_state_update(self, site_id, curr_simulation_state: SimulationState):
        site_state = curr_simulation_state.get_site_state(site_id)
        site_class = self.struct.get_site(site_id)["site_class"]

        if site_class == OCT_SITE and site_state[Occupancy] == Li:
            return self.vacate_oct_site(site_id, curr_simulation_state)
        elif site_class == OCT_SITE and site_state[Occupancy] == Vacant:
            printif(self.verbose, f"Checking tet neighbors of newly vacant {site_id}")
            return {}, self.nbs.neighbors_of(site_id)
        elif site_class == TET_SITE:
            return self.handle_tet_site(site_id, curr_simulation_state)
        else:
            return {}, []

    def vacate_oct_site(self, site_id, state: SimulationState):
        printif(self.verbose, f"Vacating {site_id}")

        curr_occupancy = state.get_site_state(site_id).get(Occupancy)

        updates = {}
        updates[site_id] = {Occupancy: Vacant}

        if curr_occupancy == Li:
            curr_li_lost_count = state.get_general_state().get(RemovedLiCount)
            updates[GENERAL] = {RemovedLiCount: curr_li_lost_count + 1}

        return updates, self.nbs.neighbors_of(site_id)

    def handle_tet_site(self, tet_site_id: int, state: SimulationState):
        printif(self.verbose, f"Checking for trivacancy at {tet_site_id}")
        tet_site_state = state.get_site_state(tet_site_id)

        if tet_site_state.get(Occupancy) is not Vacant:
            # print("THIS SHOULD NEVER BE PRINTED (not vacant)")
            return {}, []

        occupied_oct_neighb_count = 0
        occupied_oct_site = None
        occupied_oct_site_id = None

        occupied_edge_sharing_tet_count = 0

        for nb_id in self.nbs.neighbors_of(tet_site_id):
            nb_site_state = state.get_site_state(nb_id)
            nb_class = self.struct.get_site(nb_id)["site_class"]
            if nb_class == TET_SITE and nb_site_state.get(Occupancy) is not Vacant:
                occupied_edge_sharing_tet_count += 1
            else:
                oct_species = nb_site_state.get(Occupancy)
                if oct_species is not Vacant:
                    occupied_oct_neighb_count += 1
                    occupied_oct_site = nb_site_state
                    occupied_oct_site_id = nb_id

        if (
            occupied_oct_neighb_count == 1
            and occupied_oct_site[Occupancy] is not Ti
            and occupied_edge_sharing_tet_count == 0
        ):
            updates = {}
            if occupied_oct_site[Occupancy] is Mn:
                n_atoms = state.get_general_state().get(InitialAtomCount)
                n_li_initial = state.get_general_state().get(InitialLiCount)
                n_li_removed = state.get_general_state().get(RemovedLiCount)
                n_mn_2 = state.get_general_state().get(Mn2Count)
                score = (
                    n_atoms / 4
                    - 3 * (n_li_initial / 2 - n_atoms / 4)
                    - n_li_removed / 2
                )
                if n_mn_2 > score:
                    printif(
                        self.verbose, f"Maximum number of Mn disproportionated already"
                    )
                    return {}, []
                else:
                    updates[GENERAL] = {Mn2Count: n_mn_2 + 1}
            printif(
                self.verbose,
                f"Trivacancy identified at {tet_site_id}, collapsing  from {occupied_oct_site_id}",
            )

            updates[tet_site_id] = {Occupancy: occupied_oct_site[Occupancy]}
            updates[occupied_oct_site_id] = {Occupancy: Vacant}

            return updates, [occupied_oct_site_id]
        else:
            return {}, []


class DischargeController(BasicController):
    def __init__(
        self, nb_graph: Neighborhood, structure: PeriodicStructure, verbose=False
    ):
        self.nbs = nb_graph
        self.struct = structure
        self.verbose = verbose

    def get_random_site(self):
        oct_sites = [site["id"] for site in self.struct.sites(OCT_SITE)]
        return random.choice(oct_sites)

    def get_state_update(self, site_id, curr_simulation_state: SimulationState):
        printif(self.verbose, f"Updating {site_id}")
        site_state = curr_simulation_state.get_site_state(site_id)
        site_class = self.struct.get_site(site_id)["site_class"]
        if site_class == OCT_SITE and site_state[Occupancy] == Vacant:
            return self.fill_oct_site(
                site_id,
                Li,
                curr_simulation_state,
            )
        elif site_class == TET_SITE:
            return self.handle_tet_site(site_id, curr_simulation_state)
        elif site_class == OCT_SITE and site_state.get(Occupancy) is not Vacant:
            return {}, self.nbs.neighbors_of(site_id)

    def fill_oct_site(self, site_id, spec, state):
        printif(self.verbose, f"Filling {site_id}")

        updates = {}

        updates[site_id] = {Occupancy: spec}

        return updates, self.nbs.neighbors_of(site_id)

    def handle_tet_site(self, site_id, state):
        tet_site_state = state.get_site_state(site_id)
        tet_occupancy = tet_site_state.get(Occupancy)

        if tet_occupancy is Vacant:
            return {}, []

        printif(self.verbose, f"Neighboring filled tet site identified at {site_id}")

        possible_filling_oct_targets = []
        for oct_nb_id in self.nbs.neighbors_of(site_id):
            oct_site_state = state.get_site_state(oct_nb_id)
            oct_species = oct_site_state.get(Occupancy)
            if oct_species is Vacant:
                possible_filling_oct_targets.append(oct_nb_id)

        assert (
            len(possible_filling_oct_targets) > 0
        ), f"No space for {site_id} to collapse into"
        chosen_site = random.choice(possible_filling_oct_targets)
        printif(self.verbose, f"Collapsing into {chosen_site}")

        updates = {}
        updates[site_id] = {Occupancy: Vacant}
        updates[chosen_site] = {Occupancy: tet_occupancy}

        return updates, [chosen_site]

    # def fill_oct_site_old(self, site_id, spec, state, updates = None, already_visited = None, newly_filled_sites = None, newly_vacated_sites = None):
    #     printif(self.verbose, f'Filling {site_id}')

    #     already_visited.append(site_id)
    #     newly_filled_sites.add(site_id)

    #     if updates is None:
    #         updates = {}

    #     updates[site_id] = {
    #         Occupancy: spec
    #     }

    #     tet_nb_ids = self.nbs.neighbors_of(site_id)
    #     next_oct_sites = []

    #     for tet_id in tet_nb_ids:
    #         tet_site_state = state.get_site_state(tet_id)
    #         tet_occupancy = tet_site_state.get(Occupancy)

    #         if tet_occupancy is Vacant or tet_id in newly_vacated_sites:
    #             continue

    #         printif(self.verbose, f'Neighboring filled tet site identified at {tet_id}')
    #         possible_filling_oct_targets =[]
    #         for oct_nb_id in self.nbs.neighbors_of(tet_id):
    #             oct_site_state = state.get_site_state(oct_nb_id)
    #             oct_species = oct_site_state.get(Occupancy)
    #             if oct_species is Vacant and oct_nb_id not in newly_filled_sites:
    #                 possible_filling_oct_targets.append(oct_nb_id)

    #         assert len(possible_filling_oct_targets) > 0, f'No space for {tet_id} to collapse into'
    #         chosen_site = random.choice(possible_filling_oct_targets)
    #         printif(self.verbose, f'Collapsing into {chosen_site}')
    #         newly_filled_sites.add(chosen_site)
    #         newly_vacated_sites.add(tet_id)
    #         updates[tet_id] = {
    #             Occupancy: Vacant
    #         }
    #         next_oct_sites.append((chosen_site, tet_occupancy))

    #     # print(next_oct_sites)

    #     final_oct_site_list = []
    #     for oct_site, fill in next_oct_sites:
    #         # print(already_visited)
    #         if oct_site not in already_visited:
    #             final_oct_site_list.append((oct_site, fill))
    #             # self.fill_oct_site(oct_site, fill, state, updates, already_visited, newly_filled_sites, newly_vacated_sites)

    #     return updates, final_oct_site_list
