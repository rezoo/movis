import difflib
import hashlib
from collections.abc import Sequence as SequenceType
from pathlib import Path
from typing import Hashable, Sequence, Union

import pandas as pd
from pydub import AudioSegment


def make_voicevox_dataframe(audio_dir: Union[str, Path]) -> pd.DataFrame:
    def get_audio_length(filename: Union[Path, str]) -> float:
        audio = AudioSegment.from_file(str(filename), format="wav")
        return audio.duration_seconds

    wav_files = sorted(f for f in Path(audio_dir).iterdir() if f.suffix == ".wav")
    rows = []
    start_time = 0.0
    for wav_file in wav_files:
        duration = get_audio_length(wav_file)
        end_time = start_time + duration
        dic = {
            "start_time": start_time,
            "end_time": end_time,
        }
        rows.append(dic)
        start_time = end_time
    frame = pd.DataFrame(rows)
    frame["audio_file"] = [str(p) for p in wav_files]
    return frame


def make_timeline_from_voicevox(
    audio_dir: Union[str, Path],
    max_text_length: int = 25,
    extra_columns: tuple[tuple[str, Hashable], ...] = (
        ("slide", 0), ("status", "n"), ("action", "")),
) -> pd.DataFrame:

    def get_paths(src_dir: Union[str, Path], ext: str) -> list[Path]:
        src_dir = Path(src_dir)
        return sorted(f for f in src_dir.iterdir() if f.suffix == ext)

    def get_hash_prefix(text):
        text_bytes = text.encode("utf-8")
        sha1_hash = hashlib.sha1(text_bytes)
        hashed_text = sha1_hash.hexdigest()
        prefix = hashed_text[:6]
        return prefix

    txt_files = get_paths(audio_dir, ".txt")
    lines = []
    for txt_file in txt_files:
        raw_text = open(txt_file, "r", encoding="utf-8-sig").read()
        if raw_text == "":
            raise RuntimeError(
                f"Empty text file: {txt_file}. Please remove it and try again."
            )
        character_dict = {
            "ずんだもん": "zunda",
            "四国めたん": "metan",
            "春日部つむぎ": "tsumugi",
        }
        character = txt_file.stem.split("_")[1].split("（")[0]
        text = "\\n".join([
            raw_text[i: i + max_text_length]
            for i in range(0, len(raw_text), max_text_length)]
        )
        dic = {
            "character": character_dict[character],
            "hash": get_hash_prefix(raw_text),
            "text": text,
        }
        for column_name, default_value in extra_columns:
            dic[column_name] = default_value
        lines.append(dic)
    return pd.DataFrame(lines)


def merge_timeline(
    old_timeline: pd.DataFrame,
    new_timeline: pd.DataFrame,
    key="hash",
    description="text",
) -> pd.DataFrame:
    differ = difflib.Differ()
    diff = differ.compare(old_timeline[key].to_list(), new_timeline[key].tolist())
    result = []
    old_indices = old_timeline.index.tolist()
    new_indices = new_timeline.index.tolist()
    old_idx, new_idx = 0, 0
    for d in diff:
        if d.startswith("-"):
            row = old_timeline.iloc[old_indices[old_idx]].copy()
            row[description] = f"<<<<< {row[description]}"
            result.append(row)
            old_idx += 1
        elif d.startswith("+"):
            row = new_timeline.iloc[new_indices[new_idx]].copy()
            row[description] = f">>>>> {row[description]}"
            result.append(row)
            new_idx += 1
        else:
            result.append(old_timeline.iloc[old_indices[old_idx]])
            old_idx += 1
            new_idx += 1
    return pd.DataFrame(result)


def add_materials_to_video(
    video_file: Union[str, Path],
    audio_file: Union[str, Path],
    dst_file: Union[str, Path],
    subtitle_file: Union[str, Path, None] = None,
) -> None:
    import ffmpeg
    kwargs = {"vf": f"ass={str(subtitle_file)}"} if subtitle_file is not None else {'vcodec': 'copy'}
    video_input = ffmpeg.input(video_file)
    audio_input = ffmpeg.input(audio_file)
    output = ffmpeg.output(
        video_input.video,
        audio_input.audio,
        dst_file,
        **kwargs,
        acodec="aac",
        ab="128k",
    )
    output.run(overwrite_output=True)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def to_rgb(color: Union[str, tuple[int, int, int], Sequence[int]]) -> tuple[int, int, int]:
    if isinstance(color, SequenceType) and all(isinstance(x, int) for x in color):
        return (int(color[0]), int(color[1]), int(color[2]))
    elif isinstance(color, str):
        return hex_to_rgb(color)
    else:
        raise ValueError
