from typing import NamedTuple, Union
from PIL import Image

import numpy as np

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


def resize(img: np.ndarray, scale: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    if scale == (1.0, 1.0):
        return img
    h, w = img.shape[:2]
    return np.asarray(Image.fromarray(img).resize(
        (round(w * scale[0]), round(h * scale[1])), Image.BILINEAR))


def _overlay(
        bg: np.ndarray, fg: np.ndarray, p_bg: tuple[int, int], p_fg: tuple[int, int],
        size: tuple[int, int], opacity: float) -> np.ndarray:
    x_bg, y_bg = p_bg
    x_fg, y_fg = p_fg
    w, h = size
    bg_subset = bg[y_bg: (y_bg + h), x_bg: (x_bg + w)].astype(np.uint32)
    fg_subset = fg[y_fg: (y_fg + h), x_fg: (x_fg + w)].astype(np.uint32)

    bg_a = bg_subset[..., 3:]
    fg_a = (fg_subset[..., 3:] * opacity).astype(np.uint32) if opacity < 1.0 else fg_subset[..., 3:]
    out_a = 255 * fg_a + bg_a * (255 - fg_a)
    bg_rgb, fg_rgb = bg_subset[..., :3], fg_subset[..., :3]
    out_rgb = \
        (255 * fg_rgb * fg_a + bg_rgb * bg_a * (255 - fg_a)) // (out_a + (out_a == 0))
    bg[y_bg: (y_bg + h), x_bg: (x_bg + w), :3] = out_rgb.astype(np.uint8)
    bg[y_fg: (y_fg + h), x_fg: (x_fg + w), 3:] = (out_a // 255).astype(np.uint8)
    return bg


def alpha_composite_numpy(
    base_img: np.ndarray,
    component: np.ndarray,
    position: tuple[int, int] = (0, 0),
    opacity: float = 1.0,
) -> np.ndarray:
    h1, w1 = base_img.shape[:2]
    h2, w2 = component.shape[:2]

    x1, y1 = max(0, position[0]), max(0, position[1])
    x2, y2 = - min(0, position[0]), - min(0, position[1])
    w = min(position[0] + w2, w1) - x1
    h = min(position[1] + h2, h1) - y1

    if w <= 0 or h <= 0:
        return base_img

    return _overlay(base_img, component, (x1, y1), (x2, y2), (w, h), opacity)


def alpha_composite(
    base_img: np.ndarray,
    component: np.ndarray,
    position: tuple[float, float] = (0.0, 0.0),
    opacity: float = 1.0,
) -> np.ndarray:
    assert 0.0 <= opacity <= 1.0, f"opacity must be in [0, 1], but {opacity} is given."
    if opacity < 1.0:
        component = component.copy()
        c_alpha = component[:, :, 3].astype(np.uint16)
        a = int(np.round(opacity * 256))
        c_alpha = (c_alpha * a // 256).astype(np.uint8)
        component[:, :, 3] = c_alpha
    base_img_pil = Image.fromarray(base_img)
    base_img_pil.alpha_composite(
        Image.fromarray(component), (round(position[0]), round(position[1])))
    return np.asarray(base_img_pil)
