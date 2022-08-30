from rxn_ca.discrete import PhaseMap

class SolidPhaseMap(PhaseMap):

    FREE_SPACE = "Free Space"

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
        """
        phases = [self.FREE_SPACE] + phases
        super().__init__(phases)
        self.free_space_id: int = self.phase_to_int[self.FREE_SPACE]

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "phases": self.phases,
            "phase_to_int": self.phase_to_int,
            "int_to_phase": self.int_to_phase,
            "free_space_id": self.free_space_id
        }