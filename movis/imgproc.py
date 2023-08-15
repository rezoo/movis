from typing import Union

import numpy as np
from PIL import Image

from .enum import BlendingMode

try:
    from PySide6.QtGui import QImage
    pyside6_available = True
except ImportError:
    pyside6_available = False


def qimage_to_numpy(image: QImage) -> np.ndarray:
    assert pyside6_available, "PySide6 is not available."
    assert image.format() == QImage.Format.Format_ARGB32
    ptr = image.bits()
    array_shape = (image.height(), image.width(), 4)
    return np.array(ptr).reshape(array_shape)


def resize(img: np.ndarray, scale: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    if scale == (1.0, 1.0):
        return img
    h, w = img.shape[:2]
    return np.asarray(Image.fromarray(img).resize(
        (round(w * scale[0]), round(h * scale[1])), Image.BILINEAR))


def _blend_overlay(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return np.where(bg < 128, 2 * bg * fg // 255, 255 - 2 * (255 - bg) * (255 - fg) // 255)


def _blend_soft_light(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:

    def soft_light_dark(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
        return bg - (255 - 2 * fg) * bg * (255 - bg) // (255 ** 2)

    def soft_light_light(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
        def g_w3c(x: np.ndarray) -> np.ndarray:
            return np.where(
                x < 64,
                ((16 * x - (12 * 255 ** 2) * x + 4 * (255 ** 2)) * x) // (255 ** 2),
                np.sqrt(255 * x).astype(np.uint32))

        return bg + (2 * fg - 255) * (g_w3c(bg) - bg) // 255

    return np.where(fg < 128, soft_light_dark(bg, fg), soft_light_light(bg, fg))


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
        size: tuple[int, int], opacity: float,
        mode: BlendingMode = BlendingMode.NORMAL, alpha_matte_mode: bool = False) -> np.ndarray:
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
    if not alpha_matte_mode:
        bg[y_fg: (y_fg + h), x_fg: (x_fg + w), 3:] = (out_a // 255).astype(np.uint8)
    return bg


def alpha_composite_numpy(
    bg_image: np.ndarray,
    fg_image: np.ndarray,
    position: tuple[int, int] = (0, 0),
    opacity: float = 1.0,
    blending_mode: BlendingMode = BlendingMode.NORMAL,
    alpha_matte_mode: bool = False,
) -> np.ndarray:
    h1, w1 = bg_image.shape[:2]
    h2, w2 = fg_image.shape[:2]

    x1, y1 = max(0, position[0]), max(0, position[1])
    x2, y2 = - min(0, position[0]), - min(0, position[1])
    w = min(position[0] + w2, w1) - x1
    h = min(position[1] + h2, h1) - y1

    if w <= 0 or h <= 0:
        return bg_image

    return _overlay(
        bg_image, fg_image, (x1, y1), (x2, y2), (w, h), opacity,
        blending_mode, alpha_matte_mode)


def alpha_composite_pil(
    bg_image: np.ndarray,
    fg_image: np.ndarray,
    position: tuple[int, int] = (0, 0),
    opacity: float = 1.0,
) -> np.ndarray:
    assert 0.0 <= opacity <= 1.0, f"opacity must be in [0, 1], but {opacity} is given."
    if opacity < 1.0:
        fg_image = fg_image.copy()
        c_alpha = fg_image[:, :, 3].astype(np.uint16)
        a = int(np.round(opacity * 256))
        c_alpha = (c_alpha * a // 256).astype(np.uint8)
        fg_image[:, :, 3] = c_alpha
    base_img_pil = Image.fromarray(bg_image)
    base_img_pil.alpha_composite(
        Image.fromarray(fg_image), (position[0], position[1]))
    return np.asarray(base_img_pil)


def alpha_composite(
    bg_image: np.ndarray,
    fg_image: np.ndarray,
    position: tuple[int, int] = (0, 0),
    opacity: float = 1.0,
    blending_mode: Union[str, BlendingMode] = BlendingMode.NORMAL,
    alpha_matte_mode: bool = False,
) -> np.ndarray:
    if not bg_image.flags.writeable:
        bg_image = bg_image.copy()
    if blending_mode == BlendingMode.NORMAL and not alpha_matte_mode:
        # Use PIL for normal blending mode
        # because it is faster than my implementation
        return alpha_composite_pil(bg_image, fg_image, position, opacity)
    else:
        mode = BlendingMode.from_string(blending_mode) \
            if isinstance(blending_mode, str) else blending_mode
        return alpha_composite_numpy(
            bg_image, fg_image, position, opacity, mode, alpha_matte_mode)
