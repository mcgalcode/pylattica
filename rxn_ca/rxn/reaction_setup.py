
from ..grid2d import DiscreteGridSetup

class ReactionSetup(DiscreteGridSetup):

    def __init__(self, phase_map, volumes = None):
        super().__init__(phase_map)
        self.volumes = volumes

    def setup_random_sites(self, size, num_sites_desired, background_spec, nuc_species, nuc_ratios = None, buffer = 2):
        volume_scaled_ratios = [self.volumes[phase] * nuc_ratios[idx] for idx, phase in enumerate(nuc_species)]
        return super().setup_random_sites(
            size,
            num_sites_desired=num_sites_desired,
            background_spec=background_spec,
            nuc_species=nuc_species,
            nuc_ratios=volume_scaled_ratios,
            buffer=buffer
        )

    def setup_random_mixture(self, side_length, grain_size, phases, weights=None):
        volume_scaled_ratios = [self.volumes[phase] * weights[idx] for idx, phase in enumerate(phases)]
        return super().setup_random_mixture(side_length, grain_size, phases, volume_scaled_ratios)