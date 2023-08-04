from enum import Enum
from typing import Union

import numpy as np
from PIL import Image


def resize(img: np.ndarray, scale: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    if scale == (1.0, 1.0):
        return img
    h, w = img.shape[:2]
    return np.asarray(Image.fromarray(img).resize(
        (round(w * scale[0]), round(h * scale[1])), Image.BILINEAR))


class BlendingMode(Enum):
    NORMAL = 0
    MULTIPLY = 1
    SCREEN = 2
    OVERLAY = 3
    HARD_LIGHT = 4
    SOFT_LIGHT = 5

    @staticmethod
    def from_string(s: str) -> "BlendingMode":
        if s == "normal":
            return BlendingMode.NORMAL
        elif s == "multiply":
            return BlendingMode.MULTIPLY
        elif s == "screen":
            return BlendingMode.SCREEN
        elif s == "overlay":
            return BlendingMode.OVERLAY
        elif s == "hard_light":
            return BlendingMode.HARD_LIGHT
        elif s == "soft_light":
            return BlendingMode.SOFT_LIGHT
        else:
            raise ValueError(f"Unknown blending mode: {s}")


def _blend_overlay(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return np.where(bg < 128, 2 * bg * fg // 255, 255 - 2 * (255 - bg) * (255 - fg) // 255)


def _blend_soft_light(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return (255 - 2 * fg) * bg * bg // (255 * 255) + 2 * fg * bg // 255


BLENDING_MODE_TO_FUNC = {
    BlendingMode.NORMAL: lambda bg, fg: fg,
    BlendingMode.MULTIPLY: lambda bg, fg: bg * fg // 255,
    BlendingMode.SCREEN: lambda bg, fg: 255 - (255 - bg) * (255 - fg) // 255,
    BlendingMode.OVERLAY: lambda bg, fg: _blend_overlay(bg, fg),
    BlendingMode.HARD_LIGHT: lambda bg, fg: _blend_overlay(fg, bg),
    BlendingMode.SOFT_LIGHT: lambda bg, fg: _blend_soft_light(bg, fg),
}


def _overlay(
        bg: np.ndarray, fg: np.ndarray, p_bg: tuple[int, int], p_fg: tuple[int, int],
        size: tuple[int, int], opacity: float, mode: BlendingMode = BlendingMode.NORMAL) -> np.ndarray:
    x_bg, y_bg = p_bg
    x_fg, y_fg = p_fg
    w, h = size
    bg_subset = bg[y_bg: (y_bg + h), x_bg: (x_bg + w)].astype(np.uint32)
    fg_subset = fg[y_fg: (y_fg + h), x_fg: (x_fg + w)].astype(np.uint32)

    bg_a = bg_subset[..., 3:]
    fg_a = (fg_subset[..., 3:] * opacity).astype(np.uint32) if opacity < 1.0 else fg_subset[..., 3:]
    coeff1, coeff2 = 255 * fg_a, bg_a * (255 - fg_a)
    out_a = coeff1 + coeff2
    bg_rgb, fg_rgb = bg_subset[..., :3], fg_subset[..., :3]
    target_rgb = BLENDING_MODE_TO_FUNC[mode](bg_rgb, fg_rgb)
    out_rgb = (coeff1 * target_rgb + coeff2 * bg_rgb) // (out_a + (out_a == 0))
    bg[y_bg: (y_bg + h), x_bg: (x_bg + w), :3] = out_rgb.astype(np.uint8)
    bg[y_fg: (y_fg + h), x_fg: (x_fg + w), 3:] = (out_a // 255).astype(np.uint8)
    return bg


def alpha_composite_numpy(
    base_img: np.ndarray,
    component: np.ndarray,
    position: tuple[int, int] = (0, 0),
    opacity: float = 1.0,
    blending_mode: BlendingMode = BlendingMode.NORMAL,
) -> np.ndarray:
    h1, w1 = base_img.shape[:2]
    h2, w2 = component.shape[:2]

    x1, y1 = max(0, position[0]), max(0, position[1])
    x2, y2 = - min(0, position[0]), - min(0, position[1])
    w = min(position[0] + w2, w1) - x1
    h = min(position[1] + h2, h1) - y1

    if w <= 0 or h <= 0:
        return base_img

    return _overlay(base_img, component, (x1, y1), (x2, y2), (w, h), opacity, blending_mode)


def alpha_composite_pil(
    base_img: np.ndarray,
    component: np.ndarray,
    position: tuple[int, int] = (0, 0),
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
        Image.fromarray(component), (position[0], position[1]))
    return np.asarray(base_img_pil)


def alpha_composite(
    base_img: np.ndarray,
    component: np.ndarray,
    position: tuple[int, int] = (0, 0),
    opacity: float = 1.0,
    blending_mode: Union[str, BlendingMode] = BlendingMode.NORMAL,
) -> np.ndarray:
    if blending_mode == BlendingMode.NORMAL:
        return alpha_composite_pil(base_img, component, position, opacity)
    else:
        mode = BlendingMode.from_string(blending_mode) \
            if isinstance(blending_mode, str) else blending_mode
        return alpha_composite_numpy(base_img, component, position, opacity, mode)
