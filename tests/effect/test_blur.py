import movis as mv
import numpy as np


def test_gaussian_blur():
    image = np.full((10, 10, 4), 255, dtype=np.uint8)
    image_layer = mv.layer.Image(image)
    scene = mv.layer.Composition(size=(10, 10), duration=1.0)
    scene.add_layer(image_layer, name='layer')
    assert np.all(scene(0.0) == image)
    scene['layer'].add_effect(mv.effect.GaussianBlur(radius=3.0))

    out_image = scene(0.0)
    assert np.all(out_image[0, 0, :3] == [255, 255, 255])
    assert out_image[0, 0, 3] < 255


def test_glow():
    image = np.full((10, 10, 4), 200, dtype=np.uint8)
    image_layer = mv.layer.Image(image)
    scene = mv.layer.Composition(size=(10, 10), duration=1.0)
    scene.add_layer(image_layer, name='layer')
    assert np.all(scene(0.0) == image)
    scene['layer'].add_effect(mv.effect.Glow(radius=3.0))

    out_image = scene(0.0)
    assert np.all(out_image[0, 0, :3] > [200, 200, 200])
