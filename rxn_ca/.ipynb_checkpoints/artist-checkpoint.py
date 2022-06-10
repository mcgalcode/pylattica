import numpy as np

class StepArtist():
    
    def __init__(self, display_phases = None):
        self.display_phases = display_phases

    def draw(self, reaction_step, show_padding=False):
        state = reaction_step.state
        colors = ["🟥", "🟦", "🟩", "🟧"]
        
        if self.display_phases is None:
            display_phases = {}
        else:
            display_phases = self.display_phases
        
        color_counter = 0
        
        if show_padding:
            padding = 0
        else:
            padding = int(reaction_step.filter[0] / 2)
            
        phase_map = reaction_step.as_phase_name_array()
        
        for row in range(padding, phase_map.shape[0] - padding):
            for col in range(padding, phase_map.shape[1] - padding):
                phase_name = phase_map[row, col]
                if phase_name not in display_phases:
                    display_phases[phase_name] = colors[color_counter]
                    color_counter += 1
                print(display_phases[phase_name], end='')
            print('')
        
        print(display_phases)
