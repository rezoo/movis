import numpy as np

import movis as mv


def test_concatenate():
    layer1 = mv.layer.Image.from_color(size=(10, 10), color=(255, 0, 0), duration=1.0)
    layer2 = mv.layer.Image.from_color(size=(10, 10), color=(0, 255, 0), duration=2.0)
    layer3 = mv.layer.Image.from_color(size=(10, 10), color=(0, 0, 255), duration=3.0)
    scene = mv.concatenate([layer1, layer2, layer3], size=(10, 10))
    assert scene.duration == 6.0
    assert scene.layers[0].duration == 1.0
    assert scene.layers[1].duration == 2.0
    assert scene.layers[2].duration == 3.0

    assert np.all(scene(0.0)[0, 0, :] == np.array([255, 0, 0, 255]))
    assert np.all(scene(1.0)[0, 0, :] == np.array([0, 255, 0, 255]))
    assert np.all(scene(3.0)[0, 0, :] == np.array([0, 0, 255, 255]))

    scene = mv.concatenate([layer1, layer2, layer3])
    assert scene.size == (10, 10)


def test_repeat():
    img1 = mv.layer.Image.from_color(size=(10, 10), color=(255, 0, 0), duration=1.0)
    img2 = mv.layer.Image.from_color(size=(10, 10), color=(0, 255, 0), duration=1.0)
    layer = mv.concatenate([img1, img2])
    scene = mv.repeat(layer, 3, size=(10, 10))

    assert scene.duration == 6.0
    assert scene.layers[0].duration == 2.0
    assert scene.layers[1].duration == 2.0
    assert scene.layers[2].duration == 2.0

    assert np.all(scene(0.0)[0, 0, :] == np.array([255, 0, 0, 255]))
    assert np.all(scene(1.0)[0, 0, :] == np.array([0, 255, 0, 255]))
    assert np.all(scene(2.0)[0, 0, :] == np.array([255, 0, 0, 255]))
    assert np.all(scene(3.0)[0, 0, :] == np.array([0, 255, 0, 255]))
    assert np.all(scene(4.0)[0, 0, :] == np.array([255, 0, 0, 255]))
    assert np.all(scene(5.0)[0, 0, :] == np.array([0, 255, 0, 255]))

    scene = mv.repeat(layer, 3)
    assert scene.size == (10, 10)


def test_trim():
    layer1 = mv.layer.Image.from_color(size=(10, 10), color=(255, 0, 0), duration=1.0)
    layer2 = mv.layer.Image.from_color(size=(10, 10), color=(0, 255, 0), duration=2.0)
    layer3 = mv.layer.Image.from_color(size=(10, 10), color=(0, 0, 255), duration=3.0)
    layer = mv.concatenate([layer1, layer2, layer3])
    scene = mv.trim(layer, start_times=[0.0, 3.0], end_times=[1.0, 4.0], size=(10, 10))

    assert scene.duration == 2.0
    assert len(scene) == 2
    assert scene.layers[0].duration == 1.0
    assert scene.layers[1].duration == 1.0

    assert np.all(scene(0.0)[0, 0, :] == np.array([255, 0, 0, 255]))
    assert np.all(scene(1.0)[0, 0, :] == np.array([0, 0, 255, 255]))

    scene = mv.trim(layer, start_times=[0.0, 3.0], end_times=[1.0, 4.0])
    assert scene.size == (10, 10)


def test_tile():
    layer1 = mv.layer.Image.from_color(size=(10, 10), color=(255, 0, 0), duration=1.0)
    layer2 = mv.layer.Image.from_color(size=(10, 10), color=(0, 255, 0), duration=2.0)
    scene = mv.tile([layer1, layer2], rows=1, cols=2)

    assert scene.duration == 2.0
    assert scene.size == (20, 10)
    assert len(scene) == 2
    assert scene.layers[0].duration == 1.0
    assert scene.layers[1].duration == 2.0

    assert np.all(scene(0.0)[0, 0, :] == np.array([255, 0, 0, 255]))
    assert np.all(scene(0.0)[0, 10, :] == np.array([0, 255, 0, 255]))

    scene = mv.tile([layer1, layer2], rows=2, cols=1)

    assert scene.duration == 2.0
    assert scene.size == (10, 20)
    assert len(scene) == 2
    assert scene.layers[0].duration == 1.0
    assert scene.layers[1].duration == 2.0

    assert np.all(scene(0.0)[0, 0, :] == np.array([255, 0, 0, 255]))
    assert np.all(scene(0.0)[10, 0, :] == np.array([0, 255, 0, 255]))


def test_crop():
    image = np.zeros((100, 100, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    image[10:30, 10:30, 0] = 1
    layer = mv.layer.Image(image, duration=2.0)
    scene = mv.crop(layer, (10, 10, 20, 20))

    assert scene.duration == 2.0
    assert scene.size == (20, 20)
    assert len(scene) == 1

    frame = scene(0.0)
    assert np.all(frame[0, 0, :] == np.array([1, 0, 0, 255]))
    assert np.all(frame[-1, 0, :] == np.array([1, 0, 0, 255]))
    assert np.all(frame[0, -1, :] == np.array([1, 0, 0, 255]))
    assert np.all(frame[-1, -1, :] == np.array([1, 0, 0, 255]))


def test_switch():
    img1 = mv.layer.Image(np.full((100, 100, 4), 1, dtype=np.uint8), duration=5.0)
    img2 = mv.layer.Image(np.full((100, 100, 4), 2, dtype=np.uint8), duration=6.0)
    img3 = mv.layer.Image(np.full((100, 100, 4), 3, dtype=np.uint8), duration=7.0)
    scene = mv.switch([img1, img2, img3], [0.0, 1.0, 2.0, 3.0, 4.0], [0, 1, 2, 1, 0])

    assert scene.duration == 5.0
    assert scene.size == (100, 100)
    assert len(scene) == 5

    assert np.all(scene(0.0)[0, 0, :] == np.array([1, 1, 1, 1]))
    assert np.all(scene(1.0)[0, 0, :] == np.array([2, 2, 2, 2]))
    assert np.all(scene(2.0)[0, 0, :] == np.array([3, 3, 3, 3]))
    assert np.all(scene(3.0)[0, 0, :] == np.array([2, 2, 2, 2]))
    assert np.all(scene(4.0)[0, 0, :] == np.array([1, 1, 1, 1]))


def test_insert():
    img1 = mv.layer.Image(np.full((100, 100, 4), 1, dtype=np.uint8), duration=2.0)
    img2 = mv.layer.Image(np.full((100, 100, 4), 3, dtype=np.uint8), duration=3.0)
    source = mv.concatenate([img1, img2], size=img1.size)

    target = mv.layer.Image(np.full((100, 100, 4), 2, dtype=np.uint8), duration=1.0)
    scene = mv.insert(source, target, 2.0)

    assert scene.duration == 6.0
    assert scene.size == (100, 100)
    assert len(scene) == 3

    assert np.all(scene(0.0)[0, 0, :] == np.array([1, 1, 1, 1]))
    assert np.all(scene(2.0)[0, 0, :] == np.array([2, 2, 2, 2]))
    assert np.all(scene(3.0)[0, 0, :] == np.array([3, 3, 3, 3]))
    assert np.all(scene(6.0 - 1e-5)[0, 0, :] == np.array([3, 3, 3, 3]))


def test_fade_in():
    img = mv.layer.Image.from_color(size=(10, 10), color='white', duration=4.0)
    scene = mv.fade_in(img, duration=1.0)

    assert np.all(scene(0.0)[0, 0, :] == np.array([0, 0, 0, 0]))
    assert np.all(scene(1.0)[0, 0, :] == np.array([255, 255, 255, 255]))


def test_fade_out():
    img = mv.layer.Image.from_color(size=(10, 10), color='white', duration=4.0)
    scene = mv.fade_out(img, duration=1.0)

    assert np.all(scene(0.0)[0, 0, :] == np.array([255, 255, 255, 255]))
    assert np.all(scene(3.0)[0, 0, :] == np.array([255, 255, 255, 255]))
    assert np.all(scene(4.0 - 1e-5)[0, 0, :] == np.array([0, 0, 0, 0]))


def test_fade_in_out():
    img = mv.layer.Image.from_color(size=(10, 10), color='white', duration=4.0)
    scene = mv.fade_in_out(img, fade_in=1.0, fade_out=2.0)

    assert np.all(scene(0.0)[0, 0, :] == np.array([0, 0, 0, 0]))
    assert np.all(scene(1.0)[0, 0, :] == np.array([255, 255, 255, 255]))
    assert np.all(scene(2.0)[0, 0, :] == np.array([255, 255, 255, 255]))
    assert np.all(scene(4.0 - 1e-5)[0, 0, :] == np.array([0, 0, 0, 0]))
