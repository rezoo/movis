import numpy as np

import movis as mv


def test_alpha_matte():
    mask_frame = np.array(
        [[[0, 0, 0, 255], [0, 0, 0, 255]],
         [[0, 0, 0, 255], [0, 0, 0, 255]]], dtype=np.uint8)
    target_frame = np.array(
        [[[255, 0, 0, 255], [0, 255, 0, 255]],
         [[0, 0, 255, 255], [255, 255, 0, 255]]], dtype=np.uint8)
    mask = mv.layer.Image(mask_frame, 10)
    target = mv.layer.Image(target_frame, 10)

    alpha_matte = mv.layer.AlphaMatte(mask, target)

    output = alpha_matte(5)
    assert output is not None, "Output should not be None"

    assert isinstance(output, np.ndarray), "Output should be a numpy array"
    assert output.shape == mask_frame.shape, "Output and mask should have the same shape"
    assert np.all(output[:, :, 3] == mask_frame[:, :, 3]), "Alpha channel should be the same as mask"

    alpha_matte.blending_mode = mv.BlendingMode.MULTIPLY
    output_add = alpha_matte(5)
    assert isinstance(output_add, np.ndarray), "Output should be a numpy array when blending_mode is MULTIPLY"

    assert alpha_matte(11) is None, "Should return None when the time is out of duration"


def test_luminancd_matte():
    mask_frame = np.array(
        [[[0, 0, 0, 255], [0, 0, 0, 255]],
         [[0, 0, 0, 255], [0, 0, 0, 255]]], dtype=np.uint8)
    target_frame = np.array(
        [[[255, 0, 0, 255], [0, 255, 0, 255]],
         [[0, 0, 255, 255], [255, 255, 0, 255]]], dtype=np.uint8)
    mask = mv.layer.Image(mask_frame, 10)
    target = mv.layer.Image(target_frame, 10)
    luminance_matte = mv.layer.LuminanceMatte(mask, target)

    output = luminance_matte(5)
    assert output is not None, "Output should not be None"
    assert isinstance(output, np.ndarray), "Output should be a numpy array"
    assert output.shape == mask_frame.shape, "Output and mask should have the same shape"
    assert luminance_matte(11) is None, "Should return None when the time is out of duration"
