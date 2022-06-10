import matplotlib.pyplot as plt
from .artist import StepArtist
import time
from IPython.display import clear_output
import numpy as np


class ReactionResult():
    
    def __init__(self, lab):
        self.steps = []
        self.lab = lab
    
    def add_step(self, step):
        self.steps.append(step)
    
    def play(self, display_phases = None):
        artist = StepArtist(display_phases)
        for idx, step in enumerate(self.steps):
            time.sleep(0.5)
            clear_output(wait=True)
            artist.draw(step)
            print("Step: ", idx + 1)
            print("")
    
    def plot(self):
        for phase in self.lab.phase_to_int.keys():
            if phase != "F":
                xs = np.arange(len(self.steps))
                ys = [step.phase_fraction(phase) for step in self.steps]
                plt.plot(xs,ys, label=phase)
        plt.ylim([0,1])
        plt.legend()
    
    def plot_phase_counts(self):
        xs = np.arange(len(self.steps))
        ys = [step.phase_count for step in self.steps]
        plt.plot(xs, ys)
