from typing import NamedTuple, Union

import numpy as np
import jax

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


def resize(img: jax.Array, scale: tuple[float, float] = (1.0, 1.0)) -> jax.Array:
    if scale == (1.0, 1.0):
        return img
    h, w, c = img.shape
    return jax.image.resize(img, (round(h * scale[1]), round(w * scale[0]), c), method='cubic')


@jax.jit
def _overlay(bg_subset: jax.Array, fg_subset: jax.Array, opacity: float) -> jax.Array:
    bg_subset = jax.device_put(bg_subset)
    fg_subset = jax.device_put(fg_subset)

    bg_a = bg_subset[..., 3] / 255
    fg_a = fg_subset[..., 3] * (opacity / 255)

    out_a = fg_a + bg_a * (1 - fg_a)
    out_rgb = \
        (fg_subset[..., :3] * fg_a[..., None] + bg_subset[..., :3] * bg_a[..., None] * (1 - fg_a[..., None])) / out_a[..., None]
    out_subset = jax.lax.concatenate((
        out_rgb.astype(np.uint8), (out_a[:, :, None] * 255).astype(np.uint8)), 2)
    return out_subset


def alpha_composite(base_img, component, position=(0, 0), opacity=1.0):
    h1, w1 = base_img.shape[:2]
    h2, w2 = component.shape[:2]

    x1, y1 = max(0, position[0]), max(0, position[1])
    x2, y2 = - min(0, position[0]), - min(0, position[1])
    w = min(position[0] + w2, w1) - x1
    h = min(position[1] + h2, h1) - y1

    if w <= 0 or h <= 0:
        return base_img

    bg = jax.device_put(base_img)
    fg = jax.device_put(component)

    x_bg, y_bg = x1, y1
    x_fg, y_fg = x2, y2
    bg_subset = jax.lax.dynamic_slice(bg, (y_bg, x_bg, 0), (h, w, 4))
    fg_subset = jax.lax.dynamic_slice(fg, (y_fg, x_fg, 0), (h, w, 4))

    out_subset = _overlay(bg_subset, fg_subset, opacity)

    img = jax.lax.dynamic_update_slice(bg, out_subset, (y_bg, x_bg, 0))
    return img
