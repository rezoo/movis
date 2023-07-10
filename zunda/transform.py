from typing import NamedTuple

import numpy as np
from PIL import Image


class TransformProperty(NamedTuple):

    anchor_point: tuple[float, float] = (0., 0.)
    position: tuple[float, float] = (0., 0.)
    scale: tuple[float, float] = (1., 1.)
    opacity: float = 1.


def resize(img: Image.Image, scale: tuple[float, float] = (1., 1.)) -> Image.Image:
    if scale == (1., 1.):
        return img
    w, h = img.size
    return img.resize(
        (round(w * scale[0]), round(h * scale[1])), Image.Resampling.BICUBIC)


def alpha_composite(
        base_img: Image.Image, component: Image.Image,
        position: tuple[float, float] = (0., 0.,), opacity: float = 1.) -> Image.Image:
    assert 0. <= opacity <= 1., f'opacity must be in [0, 1], but {opacity} is given.'
    if opacity < 1.0:
        component = component.copy()
        c_alpha = np.asarray(component)[:, :, 3].astype(np.uint16)
        a = int(np.round(opacity * 256))
        c_alpha = (c_alpha * a // 256).astype(np.uint8)
        component.putalpha(Image.fromarray(c_alpha))
    base_img.alpha_composite(component, (round(position[0]), round(position[1])))
    return base_img
