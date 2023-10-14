from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from PySide6.QtGui import QImage

from .enum import BlendingMode, MatteMode


def _normal(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return fg


def _blend_multiply(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return (bg.astype(np.int32) * fg.astype(np.int32) // 255).astype(np.uint8)


def _blend_overlay(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return np.where(bg < 128, 2 * bg * fg // 255, 255 - 2 * (255 - bg) * (255 - fg) // 255)


def _darken(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return np.minimum(bg, fg)


def _lighten(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return np.maximum(bg, fg)


def _blend_screen(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    x = 255 - (255 - bg).astype(np.int32) * (255 - fg).astype(np.int32) // 255
    return x.astype(np.uint8)


def _color_dodge(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    x = 255 * bg.astype(np.int32) / (255 - fg.astype(np.int32) + 1)
    return np.clip(x, 0, 255).astype(np.uint8)


def _color_burn(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    x = 255 * (255 - bg.astype(np.int32)) / (fg.astype(np.int32) + 1)
    return 255 - np.clip(x, 0, 255).astype(np.uint8)


def _linear_dodge(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return np.minimum(255, bg.astype(np.int32) + fg.astype(np.int32)).astype(np.uint8)


def _linear_burn(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return np.maximum(0, bg.astype(np.int32) + fg.astype(np.int32) - 255).astype(np.uint8)


def _hard_light(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return _blend_overlay(fg, bg)


def _blend_soft_light(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:

    def soft_light_dark(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
        return bg - (255 - 2 * fg) * bg * (255 - bg) // (255 ** 2)

    def soft_light_light(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
        def g_w3c(x: np.ndarray) -> np.ndarray:
            return np.where(
                x < 64,
                ((16 * x - (12 * 255 ** 2) * x + 4 * (255 ** 2)) * x) // (255 ** 2),
                np.sqrt(255 * x).astype(np.int32))

        return bg + (2 * fg - 255) * (g_w3c(bg) - bg) // 255

    return np.where(fg < 128, soft_light_dark(bg, fg), soft_light_light(bg, fg)).astype(np.uint8)


def _vivid_light(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    fg2 = 2 * fg.astype(np.int32)
    return np.where(
        fg > 127,
        _color_dodge(bg, 2 * (fg2 - 127)),
        _color_burn(bg, fg2))


def _linear_light(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    fg2 = 2 * fg.astype(np.int32)
    return np.where(
        fg > 127,
        _linear_dodge(bg, fg2 - 255),
        _linear_burn(bg, fg2)
    )


def _pin_light(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    fg2 = 2 * fg.astype(np.int32)
    return np.where(
        fg > 127,
        np.maximum(bg, fg2 - 255),
        np.minimum(bg, fg2)).astype(np.uint8)


def _difference(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return np.abs(bg.astype(np.int32) - fg.astype(np.int32)).astype(np.uint8)


def _exclusion(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    bg = bg.astype(np.int32)
    fg = fg.astype(np.int32)
    return np.clip(bg + fg - 2 * bg * fg // 255, 0, 255).astype(np.uint8)


def _subtract(bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
    return np.clip(bg.astype(np.int32) - fg.astype(np.int32), 0, 255).astype(np.uint8)


BLENDING_MODE_TO_FUNC = {
    BlendingMode.NORMAL: _normal,
    BlendingMode.MULTIPLY: _blend_multiply,
    BlendingMode.SCREEN: _blend_screen,
    BlendingMode.OVERLAY: _blend_overlay,
    BlendingMode.DARKEN: _darken,
    BlendingMode.LIGHTEN: _lighten,
    BlendingMode.COLOR_DODGE: _color_dodge,
    BlendingMode.COLOR_BURN: _color_burn,
    BlendingMode.LINEAR_DODGE: _linear_dodge,
    BlendingMode.LINEAR_BURN: _linear_burn,
    BlendingMode.HARD_LIGHT: _hard_light,
    BlendingMode.SOFT_LIGHT: _blend_soft_light,
    BlendingMode.VIVID_LIGHT: _vivid_light,
    BlendingMode.LINEAR_LIGHT: _linear_light,
    BlendingMode.PIN_LIGHT: _pin_light,
    BlendingMode.DIFFERENCE: _difference,
    BlendingMode.EXCLUSION: _exclusion,
    BlendingMode.SUBTRACT: _subtract,
}


def _overlay(
        bg: np.ndarray, fg: np.ndarray, p_bg: tuple[int, int], p_fg: tuple[int, int],
        size: tuple[int, int], opacity: float,
        mode: BlendingMode = BlendingMode.NORMAL,
        matte_mode: MatteMode = MatteMode.NONE) -> np.ndarray:
    x_bg, y_bg = p_bg
    x_fg, y_fg = p_fg
    w, h = size
    bg_subset = bg[y_bg: (y_bg + h), x_bg: (x_bg + w)]
    fg_subset = fg[y_fg: (y_fg + h), x_fg: (x_fg + w)]

    if matte_mode == MatteMode.LUMINANCE:
        mask = cv2.cvtColor(bg_subset, cv2.COLOR_RGB2GRAY).astype(np.uint32)
        fg_a = (fg_subset[..., 3] * opacity).astype(np.uint32) \
            if opacity < 1.0 else fg_subset[..., 3].astype(np.uint32)
        bg[y_bg: (y_bg + h), x_bg: (x_bg + w), :3] = fg_subset[..., :3]
        bg[:, :, 3] = 0
        bg[y_bg: (y_bg + h), x_bg: (x_bg + w), 3] = (mask * fg_a) // 255
        return bg

    bg_subset = bg_subset.astype(np.uint32)
    fg_subset = fg_subset.astype(np.uint32)
    bg_a = bg_subset[..., 3:]
    fg_a = (fg_subset[..., 3:] * opacity).astype(np.uint32) if opacity < 1.0 else fg_subset[..., 3:]
    coeff1, coeff2 = 255 * fg_a, bg_a * (255 - fg_a)
    out_a = coeff1 + coeff2
    bg_rgb, fg_rgb = bg_subset[..., :3], fg_subset[..., :3]
    target_rgb = BLENDING_MODE_TO_FUNC[mode](bg_rgb, fg_rgb)
    out_rgb = (coeff1 * target_rgb + coeff2 * bg_rgb) // (out_a + (out_a == 0))
    bg[y_bg: (y_bg + h), x_bg: (x_bg + w), :3] = out_rgb.astype(np.uint8)
    if matte_mode == MatteMode.NONE:
        bg[y_bg: (y_bg + h), x_bg: (x_bg + w), 3:] = (out_a // 255).astype(np.uint8)
    elif matte_mode == MatteMode.ALPHA:
        pass  # Do nothing
    return bg


def _alpha_composite_numpy(
    bg_image: np.ndarray,
    fg_image: np.ndarray,
    position: tuple[int, int] = (0, 0),
    opacity: float = 1.0,
    blending_mode: BlendingMode = BlendingMode.NORMAL,
    matte_mode: MatteMode = MatteMode.NONE,
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
        blending_mode, matte_mode)


def _alpha_composite_pil(
    bg_image: np.ndarray,
    fg_image: np.ndarray,
    position: tuple[int, int] = (0, 0),
    opacity: float = 1.0,
) -> np.ndarray:
    assert 0.0 <= opacity <= 1.0, f"opacity must be in [0, 1], but {opacity} is given."
    if opacity < 1.0:
        fg_image = fg_image.copy()
        c_alpha = fg_image[:, :, 3].astype(np.uint16)
        a = int(np.round(opacity * 255))
        c_alpha = (c_alpha * a // 255).astype(np.uint8)
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
    blending_mode: str | BlendingMode = BlendingMode.NORMAL,
    matte_mode: MatteMode = MatteMode.NONE,
) -> np.ndarray:
    """Perform alpha compositing of two images (with alpha channels).

    This function asserts that both the background and foreground images have 4 channels (RGBA) and
    ``dtype=numpy.uint8``. If the background image is not writeable, a copy will be made.

    Args:
        bg_image:
            The background image as a 3D numpy array of shape ``(height, width, 4)``.
            The image should have 4 channels (RGBA), with ``dtype=numpy.uint8``.
        fg_image:
            the foreground image as a 3D numpy array of shape ``(height, width, 4)``.
            The image should have 4 channels (RGBA), with ``dtype=numpy.uint8``.
        position:
            The x, y coordinates indicating where the top-left corner of the
            foreground image should be placed on the background image.
            Default is ``(0, 0)``.
        opacity:
            The opacity level of the foreground image, between 0.0 and 1.0.
            Default is 1.0.
        blending_mode:
            The blending mode used for compositing the two images.
            Available modes are defined in the ``BlendingMode`` enum.
            Default is ``BlendingMode.NORMAL``. Note that the blending mode can also be
            specified as a string.
        matte_mode:
            The mode used for handling the matte channel.
            Available modes are defined in the ``MatteMode`` enum (``NONE``, ``ALPHA``, and ``LUMINANCE``).
            Default is ``MatteMode.NONE``. Note that the matte mode can also be specified as a string.

    Returns:
        The composited image as a 3D numpy array of shape ``(height, width, 4)``
        with ``dtype=numpy.uint8``.
    """
    assert bg_image.ndim == 3
    assert fg_image.ndim == 3
    assert bg_image.shape[2] == 4
    assert fg_image.shape[2] == 4
    assert bg_image.dtype == np.uint8
    assert fg_image.dtype == np.uint8
    if not bg_image.flags.writeable:
        bg_image = bg_image.copy()
    if blending_mode == BlendingMode.NORMAL and matte_mode == MatteMode.NONE:
        # Use PIL for normal blending mode
        # because it is faster than my implementation
        return _alpha_composite_pil(bg_image, fg_image, position, opacity)
    else:
        mode = BlendingMode.from_string(blending_mode) \
            if isinstance(blending_mode, str) else blending_mode
        return _alpha_composite_numpy(
            bg_image, fg_image, position, opacity, mode, matte_mode)


def qimage_to_numpy(image: QImage) -> np.ndarray:
    """Convert a QImage to a numpy ndarray.

    .. note::
        It asserts that the input QImage format is ``QImage.Format.Format_ARGB32``.
        The memory layout of the returned numpy array corresponds to the QImage layout.

    Args:
        image:
            The input QImage object. The function assumes that the image format
            is ``QImage.Format.Format_ARGB32``.

    Returns:
        A 3D numpy array of shape ``(height, width, 4)`` representing
        ``QImage``. The returned array will have 4 channels (RGBA) and dtype will be uint8.
    """
    assert image.format() == QImage.Format.Format_ARGB32
    ptr = image.bits()
    array_shape = (image.height(), image.width(), 4)
    return np.array(ptr).reshape(array_shape)
