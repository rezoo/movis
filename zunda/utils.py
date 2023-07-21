import hashlib
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pydub import AudioSegment


def get_paths(src_dir: Union[str, Path], ext: str) -> list[Path]:
    src_dir = Path(src_dir)
    return sorted(f for f in src_dir.iterdir() if f.suffix == ext)


def get_audio_length(filename: Path) -> float:
    audio = AudioSegment.from_file(str(filename), format="wav")
    return audio.duration_seconds


def make_voicevox_dataframe(audio_dir: Union[str, Path]) -> pd.DataFrame:
    wav_files = sorted(f for f in Path(audio_dir).iterdir() if f.suffix == '.wav')
    frame = []
    start_time = 0.0
    for wav_file in wav_files:
        duration = get_audio_length(wav_file)
        end_time = start_time + duration
        dic = {
            'start_time': start_time,
            'end_time': end_time,
        }
        frame.append(dic)
        start_time = end_time
    return pd.DataFrame(frame)


def rand_from_string(string: str, seed: int = 0) -> float:
    string = f'{seed}:{string}'
    s = hashlib.sha224(f'{seed}:{string}'.encode('utf-8')).digest()
    x = np.frombuffer(s, dtype=np.uint32)[0]
    return np.random.RandomState(x).rand()


def normalize_2dvector(x: Union[float, tuple[float, float], list[float]]) -> tuple[float, float]:
    if isinstance(x, float):
        return (x, x)
    elif isinstance(x, list):
        if len(x) != 2:
            raise ValueError(f'len(x) must be 2: {len(x)}')
        return (x[0], x[1])
    elif isinstance(x, tuple):
        if len(x) != 2:
            raise ValueError(f'len(x) must be 2: {len(x)}')
        return x
    raise TypeError(f'x must be float, tuple or list: {type(x)}')
