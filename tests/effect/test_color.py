import movis as mv
import numpy as np


def test_fill_color():
    image = np.zeros((10, 10, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    image_layer = mv.layer.Image(image)
    scene = mv.layer.Composition(size=(10, 10), duration=1.0)
    scene.add_layer(image_layer, name='layer')
    assert np.all(scene(0.0) == image)
    scene['layer'].add_effect(mv.effect.FillColor(color='#ff0000'))

    filled_image = scene(0.0)
    assert np.all(filled_image[0, 0, :] == [255, 0, 0, 255])


def test_hsl_shift():
    image = np.zeros((10, 10, 4), dtype=np.uint8)
    image[:, :, :3] = [0, 0, 255]
    image[:, :, 3] = 255
    image_layer = mv.layer.Image(image)
    scene = mv.layer.Composition(size=(10, 10), duration=1.0)
    scene.add_layer(image_layer, name='layer')
    assert np.all(scene(0.0) == image)

    scene['layer'].add_effect(mv.effect.HSLShift(hue=360.0, saturation=0.0, luminance=0.0))
    shifted_image = scene(0.0)
    assert np.all(shifted_image[0, 0, :] == [0, 0, 255, 255])

    scene['layer'].add_effect(mv.effect.HSLShift(hue=180.0, saturation=0.0, luminance=0.0))
    shifted_image = scene(0.0)
    assert np.all(shifted_image[0, 0, :] == [255, 255, 0, 255])
