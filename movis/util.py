from __future__ import annotations

from collections.abc import Sequence as SequenceType
from os import PathLike
from typing import Sequence


def add_materials_to_video(
    video_file: str | PathLike,
    audio_file: str | PathLike,
    dst_file: str | PathLike,
    subtitle_file: str | PathLike | None = None,
) -> None:
    """Merges a video file, an audio file, and optionally a subtitle file into a new video file.

    It uses the ffmpeg library for media processing. The resulting video will have the audio
    and optionally the subtitles embedded.

    Args:
        video_file:
            A ``str`` or ``PathLike`` object representing the path to the source video file.
        audio_file:
            A ``str`` or ``PathLike`` object representing the path to the audio file to be added.
        dst_file:
            A ``str`` or ``PathLike`` object representing the path to the destination video file.
        subtitle_file:
            A ``str``, ``PathLike``, or ``None`` representing the path to
            the subtitle file to be added. Default is ``None``.
    """
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


def _csscolor_to_rgb(css_name: str) -> tuple[int, int, int]:
    color_dict = {
        'aliceblue': '#F0F8FF',
        'antiquewhite': '#FAEBD7',
        'aqua': '#00FFFF',
        'aquamarine': '#7FFFD4',
        'azure': '#F0FFFF',
        'beige': '#F5F5DC',
        'bisque': '#FFE4C4x',
        'black': '#000000',
        'blanchedalmond': '#FFEBCD',
        'blue': '#0000FF',
        'blueviolet': '#8A2BE2',
        'brown': '#A52A2A',
        'burlywood': '#DEB887',
        'cadetblue': '#5F9EA0',
        'chartreuse': '#7FFF00',
        'chocolate': '#D2691E',
        'coral': '#FF7F50',
        'cornflowerblue': '#6495ED',
        'cornsilk': '#FFF8DC',
        'crimson': '#DC143C',
        'cyan': '#00FFFF',
        'darkblue': '#00008B',
        'darkcyan': '#008B8B',
        'darkgoldenrod': '#B8860B',
        'darkgray': '#A9A9A9',
        'darkgrey': '#A9A9A9',
        'darkgreen': '#006400',
        'darkkhaki': '#BDB76B',
        'darkmagenta': '#8B008B',
        'darkolivegreen': '#556B2F',
        'darkorange': '#FF8C00',
        'darkorchid': '#9932CC',
        'darkred': '#8B0000',
        'darksalmon': '#E9967A',
        'darkseagreen': '#8FBC8F',
        'darkslateblue': '#483D8B',
        'darkslategray': '#2F4F4F',
        'darkslategrey': '#2F4F4F',
        'darkturquoise': '#00CED1',
        'darkviolet': '#9400D3',
        'deeppink': '#FF1493',
        'deepskyblue': '#00BFFF',
        'dimgray': '#696969',
        'dimgrey': '#696969',
        'dodgerblue': '#1E90FF',
        'firebrick': '#B22222',
        'floralwhite': '#FFFAF0',
        'forestgreen': '#228B22',
        'fuchsia': '#FF00FF',
        'gainsboro': '#DCDCDC',
        'ghostwhite': '#F8F8FF',
        'gold': '#FFD700',
        'goldenrod': '#DAA520',
        'gray': '#808080',
        'grey': '#808080',
        'green': '#008000',
        'greenyellow': '#ADFF2F',
        'honeydew': '#F0FFF0',
        'hotpink': '#FF69B4',
        'indianred': '#CD5C5C',
        'indigo': '#4B0082',
        'ivory': '#FFFFF0',
        'khaki': '#F0E68C',
        'lavender': '#E6E6FA',
        'lavenderblush': '#FFF0F5',
        'lawngreen': '#7CFC00',
        'lemonchiffon': '#FFFACD',
        'lightblue': '#ADD8E6',
        'lightcoral': '#F08080',
        'lightcyan': '#E0FFFF',
        'lightgoldenrodyellow': '#FAFAD2',
        'lightgray': '#D3D3D3',
        'lightgrey': '#D3D3D3',
        'lightgreen': '#90EE90',
        'lightpink': '#FFB6C1',
        'lightsalmon': '#FFA07A',
        'lightseagreen': '#20B2AA',
        'lightskyblue': '#87CEFA',
        'lightslategray': '#778899',
        'lightslategrey': '#778899',
        'lightsteelblue': '#B0C4DE',
        'lightyellow': '#FFFFE0',
        'lime': '#00FF00',
        'limegreen': '#32CD32',
        'linen': '#FAF0E6',
        'magenta': '#FF00FF',
        'maroon': '#800000',
        'mediumaquamarine': '#66CDAA',
        'mediumblue': '#0000CD',
        'mediumorchid': '#BA55D3',
        'mediumpurple': '#9370DB',
        'mediumseagreen': '#3CB371',
        'mediumslateblue': '#7B68EE',
        'mediumspringgreen': '#00FA9A',
        'mediumturquoise': '#48D1CC',
        'mediumvioletred': '#C71585',
        'midnightblue': '#191970',
        'mintcream': '#F5FFFA',
        'mistyrose': '#FFE4E1',
        'moccasin': '#FFE4B5',
        'navajowhite': '#FFDEAD',
        'navy': '#000080',
        'oldlace': '#FDF5E6',
        'olive': '#808000',
        'olivedrab': '#6B8E23',
        'orange': '#FFA500',
        'orangered': '#FF4500',
        'orchid': '#DA70D6',
        'palegoldenrod': '#EEE8AA',
        'palegreen': '#98FB98',
        'paleturquoise': '#AFEEEE',
        'palevioletred': '#DB7093',
        'papayawhip': '#FFEFD5',
        'peachpuff': '#FFDAB9',
        'peru': '#CD853F',
        'pink': '#FFC0CB',
        'plum': '#DDA0DD',
        'powderblue': '#B0E0E6',
        'purple': '#800080',
        'rebeccapurple': '#663399',
        'red': '#FF0000',
        'rosybrown': '#BC8F8F',
        'royalblue': '#4169E1',
        'saddlebrown': '#8B4513',
        'salmon': '#FA8072',
        'sandybrown': '#F4A460',
        'seagreen': '#2E8B57',
        'seashell': '#FFF5EE',
        'sienna': '#A0522D',
        'silver': '#C0C0C0',
        'skyblue': '#87CEEB',
        'slateblue': '#6A5ACD',
        'slategray': '#708090',
        'slategrey': '#708090',
        'snow': '#FFFAFA',
        'springgreen': '#00FF7F',
        'steelblue': '#4682B4',
        'tan': '#D2B48C',
        'teal': '#008080',
        'thistle': '#D8BFD8',
        'tomato': '#FF6347',
        'turquoise': '#40E0D0',
        'violet': '#EE82EE',
        'wheat': '#F5DEB3',
        'white': '#FFFFFF',
        'whitesmoke': '#F5F5F5',
        'yellow': '#FFFF00',
        'yellowgreen': '#9ACD32',
    }
    if css_name in color_dict:
        hex_str = color_dict[css_name]
        return _hex_to_rgb(hex_str)
    else:
        raise ValueError(f'Invalid color name: {css_name}')


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


def to_rgb(color: str | tuple[int, int, int] | Sequence[int]) -> tuple[int, int, int]:
    """Converts a color input to its RGB tuple representation.

    The input can be either a string representing a color name or a hex code,
    or a sequence of integers representing the RGB values.

    Args:
        color:
            A union type that can be either a string or a tuple containing RGB integers or a sequence of integers.
            If ``str``: The string can be either a CSS color name (e.g., 'red') or a hexadecimal RGB string
            (e.g., ``'#FF0000'``). If ``tuple[int, int, int]`` or ``Sequence[int]``, it represents RGB values
            as integers between 0 and 255 (`e.g.`, ``(255, 0, 0)``).

    Returns:
        A tuple of three integers ``(R, G, B)`` that represent the RGB values.
    """
    if isinstance(color, SequenceType) and all(isinstance(x, int) for x in color):
        return (int(color[0]), int(color[1]), int(color[2]))
    elif isinstance(color, str):
        if not color.startswith('#'):
            return _csscolor_to_rgb(color.lower())
        else:
            return _hex_to_rgb(color)
    else:
        raise ValueError
