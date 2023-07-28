from typing import NamedTuple, Union

import numpy as np
from PIL import Image

from zunda.utils import normalize_2dvector


class Transform(NamedTuple):

    anchor_point: tuple[float, float] = (0.0, 0.0)
    position: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)
    opacity: float = 1.0

    def __post_init__(self):
        if self.opacity < 0.0 or 1.0 < self.opacity:
            raise ValueError("opacity must be in the range [0, 1]")

    @staticmethod
    def create(
        anchor_point: Union[float, tuple[float, float], list[float]] = (0.0, 0.0),
        position: Union[float, tuple[float, float], list[float]] = (0.0, 0.0),
        scale: Union[float, tuple[float, float], list[float]] = (1.0, 1.0),
        opacity: float = 1.0,
    ) -> "Transform":
        return Transform(
            anchor_point=normalize_2dvector(anchor_point),
            position=normalize_2dvector(position),
            scale=normalize_2dvector(scale),
            opacity=float(opacity),
        )


def resize(img: Image.Image, scale: tuple[float, float] = (1.0, 1.0)) -> Image.Image:
    if scale == (1.0, 1.0):
        return img
    w, h = img.size
    return img.resize(
        (round(w * scale[0]), round(h * scale[1])), Image.Resampling.BICUBIC
    )


def alpha_composite(
    base_img: Image.Image,
    component: Image.Image,
    position: tuple[float, float] = (
        0.0,
        0.0,
    ),
    opacity: float = 1.0,
) -> Image.Image:
    assert 0.0 <= opacity <= 1.0, f"opacity must be in [0, 1], but {opacity} is given."
    if opacity < 1.0:
        component = component.copy()
        c_alpha = np.asarray(component)[:, :, 3].astype(np.uint16)
        a = int(np.round(opacity * 256))
        c_alpha = (c_alpha * a // 256).astype(np.uint8)
        component.putalpha(Image.fromarray(c_alpha))
    base_img.alpha_composite(component, (round(position[0]), round(position[1])))
    return base_img
