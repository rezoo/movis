import hashlib
import os
from typing import List

import numpy as np
import pandas as pd
from pydub import AudioSegment


def get_paths(src_dir: str, ext: str) -> List[str]:
    return sorted([
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir) if f.endswith(ext)])


def get_audio_length(filename: str) -> float:
    audio = AudioSegment.from_file(filename, format="wav")
    return audio.duration_seconds


def get_audio_dataframe(audio_dir: str) -> pd.DataFrame:
    wav_files = get_paths(audio_dir, '.wav')
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
    s = np.frombuffer(s, dtype=np.uint32)[0]
    return np.random.RandomState(s).rand()
