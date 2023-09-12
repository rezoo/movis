from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Hashable

import pandas as pd
import librosa


def make_voicevox_dataframe(audio_dir: str | Path) -> pd.DataFrame:
    def get_audio_length(filename: str | Path) -> float:
        return librosa.get_duration(path=str(filename))

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
    audio_dir: str | Path,
    max_text_length: int = 25,
    extra_columns: tuple[tuple[str, Hashable], ...] = (
        ("slide", 0), ("status", "n"), ("action", "")),
) -> pd.DataFrame:

    def get_paths(src_dir: str | Path, ext: str) -> list[Path]:
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
