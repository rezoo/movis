import math
from pathlib import Path
from typing import Union

from pydub import AudioSegment


def concat_audio_files(start_times: list[float], audio_files: list[Union[str, Path]]) -> AudioSegment:
    assert len(audio_files) == len(start_times)
    audio = AudioSegment.empty()
    silence = AudioSegment.silent(duration=1000)
    for path, start_time in zip(audio_files, start_times):
        p = Path(path)
        audio_i = AudioSegment.from_file(str(p), format=p.suffix[1:])
        diff_duration = int((start_time * 1000) - len(audio))
        if diff_duration > 0:
            audio += silence[:diff_duration]
        audio += audio_i
    return audio


def make_loop_music(audio_file: Union[str, Path], duration: float) -> AudioSegment:
    path = Path(audio_file)
    bgm: AudioSegment = AudioSegment.from_file(str(path), format=path.suffix[1:])
    bgm_repeat_times = int(math.ceil(duration / bgm.duration_seconds))
    bgm = bgm * bgm_repeat_times
    bgm = bgm[: int(duration * 1000)]
    return bgm
