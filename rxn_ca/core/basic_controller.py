import numpy as np
from rxn_ca.core.basic_simulation_result import BasicSimulationResult

from rxn_ca.core.basic_simulation_step import BasicSimulationStep

class BasicController():

    def instantiate_result(self):
        pass

    def get_new_state(self, padded_state: np.array, row_num: int, j: int):
        pass

    def pad_step(self, step: BasicSimulationStep):
        return self.neighborhood.pad_step(step)

    def instantiate_result(self):
        return BasicSimulationResult()
