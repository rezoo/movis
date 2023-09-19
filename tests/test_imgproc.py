import numpy as np
import pytest

from movis.enum import BlendingMode, MatteMode
from movis.imgproc import alpha_composite

alpha_composite_params = [
    (0.5, BlendingMode.NORMAL),
    (0.2, BlendingMode.MULTIPLY),
    (1.0, BlendingMode.SCREEN),
    (0.9, BlendingMode.OVERLAY),
    (0.8, BlendingMode.DARKEN),
    (0.7, BlendingMode.LIGHTEN),
    (0.6, BlendingMode.COLOR_DODGE),
    (0.5, BlendingMode.LINEAR_DODGE),
    (0.4, BlendingMode.LINEAR_BURN),
    (0.3, BlendingMode.HARD_LIGHT),
    (0.1, BlendingMode.SOFT_LIGHT),
    (0.2, BlendingMode.VIVID_LIGHT),
    (0.3, BlendingMode.LINEAR_LIGHT),
    (0.4, BlendingMode.PIN_LIGHT),
    (0.5, BlendingMode.DIFFERENCE),
    (0.6, BlendingMode.EXCLUSION),
    (0.7, BlendingMode.SUBTRACT),
]


@pytest.mark.parametrize("opacity, blending_mode", alpha_composite_params)
def test_alpha_composite(opacity, blending_mode):
    bg = np.random.randint(0, 255, size=(128, 256, 4)).astype(np.uint8)
    fg = np.random.randint(0, 255, size=(64, 128, 4)).astype(np.uint8)

    for (x, y) in [(0, 0), (-10, -20), (96, 48), (-10, 48), (96, -20)]:
        bg_dst = alpha_composite(
            bg.copy(), fg, position=(x, y),
            opacity=opacity, blending_mode=blending_mode)
        assert bg.shape == bg_dst.shape
        assert bg_dst.dtype == np.uint8


@pytest.mark.parametrize("opacity, blending_mode", alpha_composite_params)
def test_alpha_composite_outside(opacity, blending_mode):
    bg = np.random.randint(0, 255, size=(128, 256, 4)).astype(np.uint8)
    fg = np.random.randint(0, 255, size=(64, 128, 4)).astype(np.uint8)
    for (x, y) in [(-200, -200), (512, 512)]:
        bg_dst = alpha_composite(
            bg.copy(), fg, position=(x, y),
            opacity=opacity, blending_mode=blending_mode)
        np.testing.assert_allclose(bg, bg_dst)
        assert bg_dst.dtype == np.uint8


@pytest.mark.parametrize("opacity, blending_mode", alpha_composite_params)
def test_alpha_composite_alphamatte(opacity, blending_mode):
    bg = np.random.randint(0, 255, size=(128, 256, 4)).astype(np.uint8)
    fg = np.random.randint(0, 255, size=(64, 128, 4)).astype(np.uint8)
    for (x, y) in [(0, 0), (-10, -20), (96, 48), (-10, 48), (96, -20)]:
        bg_dst = alpha_composite(
            bg.copy(), fg, position=(x, y),
            opacity=opacity, blending_mode=blending_mode, matte_mode=MatteMode.ALPHA)
        np.testing.assert_allclose(bg[:, :, 3], bg_dst[:, :, 3])
        assert bg_dst.dtype == np.uint8


@pytest.mark.parametrize("opacity, blending_mode", alpha_composite_params)
def test_alpha_composite_luminancematte(opacity, blending_mode):
    bg = np.random.randint(0, 255, size=(128, 256, 4)).astype(np.uint8)
    fg = np.random.randint(0, 255, size=(64, 128, 4)).astype(np.uint8)
    for (x, y) in [(0, 0), (-10, -20), (96, 48), (-10, 48), (96, -20)]:
        bg2 = bg.copy()
        bg2[:, :, :3] = 0
        bg_dst = alpha_composite(
            bg2, fg, position=(x, y),
            opacity=opacity, blending_mode=blending_mode, matte_mode=MatteMode.LUMINANCE)
        np.testing.assert_allclose(bg_dst[:, :, 3], 0)
        assert bg_dst.dtype == np.uint8
