from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

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


def generate_sine_wave(frequency: float = 440.0, duration: float = 1.0):
    T = int(mv.AUDIO_SAMPLING_RATE * duration)
    t = np.linspace(0, duration, T, endpoint=False)
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    sine_wave = np.broadcast_to(sine_wave[:, None], (T, 2))
    return sine_wave


def test_audio_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.wav"
        sine_wave = generate_sine_wave(duration=1.0)
        sf.write(str(temp_file), sine_wave, mv.AUDIO_SAMPLING_RATE, subtype='PCM_24')

        layer = mv.layer.Audio(temp_file)
        assert layer.duration == 1.0
        audio = layer.get_audio(0.0, 1.0)
        assert audio.shape == (2, mv.AUDIO_SAMPLING_RATE)


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


def test_audiosequence_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "test.wav"
        sine_wave = generate_sine_wave(duration=1.0)
        sf.write(str(temp_file), sine_wave, mv.AUDIO_SAMPLING_RATE, subtype='PCM_24')

        start_times = np.array([0.0, 1.0])
        end_times = np.array([1.0, 2.0])
        audios = [temp_file, temp_file]
        layer = mv.layer.AudioSequence(start_times, end_times, audios)
        assert layer.duration == 2.0
        audio = layer.get_audio(0.0, 2.0)
        assert audio.shape == (2, 2 * mv.AUDIO_SAMPLING_RATE)
