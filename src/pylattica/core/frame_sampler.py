import math


def derive_compress_freq(num_steps: int, num_frames: int) -> int:
    """Derive a sampling interval that yields roughly num_frames samples.

    The first and final samples are always included by the caller, so the
    stored count is approximately num_frames.

    Parameters
    ----------
    num_steps : int
        The total number of steps the simulation will run.
    num_frames : int
        The target number of samples to store over the run.

    Returns
    -------
    int
        The sampling interval (>= 1).
    """
    return max(1, math.ceil(num_steps / num_frames))


class FrameSampler:
    """Encapsulates a frame-sampling cadence.

    This is the single source of truth for how often a sampled value (a stored
    frame in SimulationResult, or an observation in ObservedResult) is taken.

    Either compress_freq or num_frames may be specified (not both). When
    num_frames is given, the effective interval cannot be known until the total
    step count is available, so it is derived later via prepare_for_steps(). The
    interval defaults to every step until then.

    Attributes
    ----------
    compress_freq : int
        The effective sampling interval. A sample is taken whenever the step
        number is a multiple of this value.
    num_frames : int, optional
        The target total number of samples, if configured by frame count.
    """

    def __init__(self, compress_freq: int = None, num_frames: int = None):
        if compress_freq is not None and num_frames is not None:
            raise ValueError("Specify either compress_freq or num_frames, not both.")
        if compress_freq is not None and compress_freq < 1:
            raise ValueError(f"compress_freq must be >= 1, got {compress_freq}.")
        if num_frames is not None and num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}.")

        self.num_frames = num_frames
        # Until derived (when num_frames is set), default to sampling every step.
        self.compress_freq = compress_freq if compress_freq is not None else 1

    def prepare_for_steps(self, num_steps: int) -> None:
        """Derive the sampling interval from the planned step count.

        A no-op unless the sampler was configured with num_frames.

        Parameters
        ----------
        num_steps : int
            The total number of steps the simulation will run.
        """
        if self.num_frames is None or num_steps is None:
            return
        self.compress_freq = derive_compress_freq(num_steps, self.num_frames)

    def should_sample(self, step_no: int) -> bool:
        """Whether a sample should be taken at the given step number.

        Step 0 always samples (0 is a multiple of any interval), matching the
        convention that the initial frame is always recorded.
        """
        return step_no % self.compress_freq == 0
