from pylattica.discrete import PhaseSet
from pylattica.structures.square_grid import DiscreteGridSetup
from pylattica.structures.square_grid.neighborhoods import PseudoHexagonalNeighborhoodBuilder2D
from pylattica.core.simulation_state import SimulationState
from pylattica.core.constants import SITES, GENERAL

def test_can_run_growth_sim_series():
    phases = PhaseSet(["A", "B", "C", "D"])
    nb_spec = PseudoHexagonalNeighborhoodBuilder2D()
    setup = DiscreteGridSetup(phases)
    periodic_initial_state = setup.setup_random_sites(20, 20, "A", ["B", "C", "D"], [1, 1, 1])
    assert 'SITES' not in periodic_initial_state.state.site_ids()

def test_can_serialize():
    state = SimulationState()
    state.set_site_state(0, { "a": 1 })

    d = state.as_dict()
    assert "state" in d
    assert "@module" in d
    assert "@class" in d
    
    assert SITES in d["state"] and GENERAL in d["state"]

    rehydrated = SimulationState.from_dict(d)
    assert rehydrated.get_site_state(0)["a"] == 1

def test_batch_update():
    state = SimulationState()

    updates = {
        1: { "a": 3 },
        2: { "b": 4 }
    }
    
    state.batch_update(updates)

    assert state.get_site_state(1)["a"] == 3
    assert state.get_site_state(2)["b"] == 4