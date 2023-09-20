import numpy as np

import movis as mv


def test_initialization():
    drop_shadow = mv.effect.DropShadow(radius=10.0, offset=5.0, angle=45.0, color="red", opacity=0.8)

    assert drop_shadow.radius.init_value == 10.0
    assert drop_shadow.offset.init_value == 5.0
    assert drop_shadow.angle.init_value == 45.0
    assert np.array_equal(drop_shadow.color.init_value, np.array([255, 0, 0]))
    assert drop_shadow.opacity.init_value == 0.8


def test_drop_shadow_effect():
    input_image = np.full((100, 100, 4), [255, 255, 255, 255], dtype=np.uint8)
    drop_shadow = mv.effect.DropShadow(
        radius=10.0, offset=5.0, angle=45.0, color=(0, 0, 0), opacity=0.5)

    output_image = drop_shadow(input_image, time=0.0)

    H, W = output_image.shape[:2]
    assert 100 <= H
    assert 100 <= W
    x = (H - 100) // 2
    y = (W - 100) // 2
    assert np.all(output_image[y, x] == [255, 255, 255, 255])
