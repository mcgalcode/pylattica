import pytest

from pylattica.core.frame_sampler import FrameSampler, derive_compress_freq


def test_derive_compress_freq_rounds_up():
    # ceil(100 / 3) == 34
    assert derive_compress_freq(100, 3) == 34


def test_derive_compress_freq_floor_is_one():
    # More frames requested than steps still yields a valid interval.
    assert derive_compress_freq(5, 100) == 1


def test_default_is_every_step():
    sampler = FrameSampler()
    assert sampler.compress_freq == 1
    assert sampler.num_frames is None


def test_compress_freq_is_honored():
    sampler = FrameSampler(compress_freq=7)
    assert sampler.compress_freq == 7
    assert sampler.num_frames is None


def test_num_frames_defers_until_prepared():
    sampler = FrameSampler(num_frames=10)
    # Until prepared, sampling defaults to every step.
    assert sampler.compress_freq == 1
    assert sampler.num_frames == 10

    sampler.prepare_for_steps(100)
    assert sampler.compress_freq == 10


def test_prepare_noop_without_num_frames():
    sampler = FrameSampler(compress_freq=4)
    sampler.prepare_for_steps(100)
    assert sampler.compress_freq == 4


def test_prepare_noop_with_none_steps():
    sampler = FrameSampler(num_frames=10)
    sampler.prepare_for_steps(None)
    assert sampler.compress_freq == 1


def test_both_args_raise():
    with pytest.raises(ValueError, match="either"):
        FrameSampler(compress_freq=5, num_frames=5)


def test_compress_freq_below_one_raises():
    with pytest.raises(ValueError, match="compress_freq"):
        FrameSampler(compress_freq=0)


def test_num_frames_below_one_raises():
    with pytest.raises(ValueError, match="num_frames"):
        FrameSampler(num_frames=0)


def test_should_sample_on_multiples():
    sampler = FrameSampler(compress_freq=5)
    assert sampler.should_sample(0) is True
    assert sampler.should_sample(5) is True
    assert sampler.should_sample(10) is True
    assert sampler.should_sample(3) is False
    assert sampler.should_sample(7) is False


def test_should_sample_step_zero_always():
    # 0 is a multiple of any interval -> the initial frame always samples.
    sampler = FrameSampler(compress_freq=999)
    assert sampler.should_sample(0) is True
