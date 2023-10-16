import numpy as np

import movis as mv


def test_audio_ndarray():
    duration = 1
    audio = np.ones((2, duration * mv.AUDIO_SAMPLING_RATE), dtype=np.float32)
    layer = mv.layer.Audio(audio)
    assert layer.duration == float(duration)

    audio = np.ones((duration * mv.AUDIO_SAMPLING_RATE), dtype=np.float32)
    layer = mv.layer.Audio(audio)
    assert layer.duration == float(duration)

    x = layer.get_audio(0.0, 0.5)
    assert x.shape == (2, mv.AUDIO_SAMPLING_RATE // 2)
    assert np.all(x == 1.0)

    x = layer.get_audio(0.5, 1.0)
    assert x.shape == (2, mv.AUDIO_SAMPLING_RATE // 2)
    assert np.all(x == 1.0)

    x = layer.get_audio(-0.5, 0.0)
    assert x is None

    x = layer.get_audio(1.0, 1.5)
    assert x is None

    x = layer.get_audio(-0.5, 0.5)
    assert x.shape == (2, mv.AUDIO_SAMPLING_RATE)
    assert np.all(x[:, 0] == 0.0)
    assert np.all(x[:, -1] == 1.0)

    x = layer.get_audio(0.5, 1.5)
    assert x.shape == (2, mv.AUDIO_SAMPLING_RATE)
    assert np.all(x[:, 0] == 1.0)
    assert np.all(x[:, -1] == 0.0)

    x = layer.get_audio(-0.5, 1.5)
    assert x.shape == (2, 2 * mv.AUDIO_SAMPLING_RATE)
    assert np.all(x[:, 0] == 0.0)
    assert np.all(x[:, mv.AUDIO_SAMPLING_RATE] == 1.0)
    assert np.all(x[:, -1] == 0.0)


def test_audiosequence_ndarray():
    start_times = np.array([0.0, 1.0, 2.0])
    end_times = np.array([1.0, 2.0, 3.0])
    audios = [
        1.0 * np.ones((2, mv.AUDIO_SAMPLING_RATE), dtype=np.float32),
        2.0 * np.ones((2, mv.AUDIO_SAMPLING_RATE), dtype=np.float32),
        3.0 * np.ones((2, mv.AUDIO_SAMPLING_RATE), dtype=np.float32),
    ]
    layer = mv.layer.AudioSequence(start_times, end_times, audios)
    T = mv.AUDIO_SAMPLING_RATE

    x = layer.get_audio(0.0, 3.0)
    assert x.shape == (2, 3 * T)
    assert np.all(x[:, 0:T] == 1.0)
    assert np.all(x[:, T:2 * T] == 2.0)
    assert np.all(x[:, 2 * T:3 * T] == 3.0)

    x = layer.get_audio(0.5, 2.5)
    assert x.shape == (2, 2 * T)
    assert np.all(x[:, 0:T // 2] == 1.0)
    assert np.all(x[:, T // 2:T * 3 // 2] == 2.0)
    assert np.all(x[:, T * 3 // 2:T * 2] == 3.0)

    x = layer.get_audio(-0.5, 2.5)
    assert x.shape == (2, 3 * T)
    assert np.all(x[:, 0:T // 2] == 0.0)
    assert np.all(x[:, T // 2:T * 3 // 2] == 1.0)
    assert np.all(x[:, T * 3 // 2:T * 5 // 2] == 2.0)
    assert np.all(x[:, T * 5 // 2:T * 7 // 2] == 3.0)

    x = layer.get_audio(2.5, 3.5)
    assert x.shape == (2, T)
    assert np.all(x[:, 0:T // 2] == 3.0)
    assert np.all(x[:, T // 2:T] == 0.0)

    x = layer.get_audio(-0.5, 0.0)
    assert x is None

    x = layer.get_audio(3.0, 3.5)
    assert x is None
