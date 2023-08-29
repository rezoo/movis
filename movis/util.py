from collections.abc import Sequence as SequenceType
from pathlib import Path
from typing import Sequence, Union


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
