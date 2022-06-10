import typing
import numpy as np
from reaction_model.reaction_step import ReactionStep

from reaction_model.scored_reaction_set import ScoredReactionSet

class PhaseMap():

    FREE_SPACE = "_free_space"

    @classmethod
    def from_dict(cls, pm_dict):
        pm = cls([])
        pm.phases = pm_dict["phases"]
        pm.int_to_phase = { int(k): v for (k, v) in pm_dict["int_to_phase"].items() }
        pm.phase_to_int = pm_dict["phase_to_int"]
        pm.free_space_id = pm_dict["free_space_id"]
        return pm


    def __init__(self, phases: list[str]):
        """Initializes a SolidReactionSet object. Requires a list of possible reactions
        and the elements which should be considered available in the atmosphere of the
        simulation.

        Args:
            reactions (list[Reaction]):
            open_species (list[str], optional): A list of open species, e.g. CO2. Defaults to [].
            free_species (list[str]), optional): A list of gaseous or liquid species which should not be represented in the grid
        """
        self.phases: list[str] = [self.FREE_SPACE] + phases

        self.int_to_phase: typing.Dict[int, str] = {}
        self.phase_to_int: typing.Dict[str, int] = {}
        for idx, phase_name in enumerate(self.phases):
            self.int_to_phase[idx] = phase_name
            self.phase_to_int[phase_name] = idx

        self.free_space_id: int = self.phase_to_int[self.FREE_SPACE]

    def step_as_phase_name_array_no_padding(self, step: ReactionStep):
        return self.as_phase_name_array(step.state)

    def step_as_phase_name_array(self, step):
        state = step.state
        return self.as_phase_name_array(state)

    def as_phase_name_array(self, state = None):
        phase_name_map = np.empty(state.shape, dtype=np.dtype('U100'))
        for phase_idx in self.int_to_phase.keys():
            phase_name = self.int_to_phase[phase_idx]
            phase_name_map[state == phase_idx] = phase_name
        return phase_name_map

    def to_dict(self):
        return {
            "phases": self.phases,
            "phase_to_int": self.phase_to_int,
            "int_to_phase": self.int_to_phase,
            "free_space_id": self.free_space_id
        }